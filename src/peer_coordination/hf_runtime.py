"""Production Hugging Face runtime adapter for bounded JSON work products.

This module keeps provider-specific request shaping outside the coordination
control plane. Each live call sends the exact Pydantic JSON Schema for the bounded
peer task and still passes the returned content through the same strict parser and
Pydantic validation used by the application.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import (
    AnalystModelOutput,
    HuggingFaceChatClient,
    ModelConfigurationError,
    SkepticModelOutput,
    StructuredModelError,
    SynthesizerModelOutput,
)
from .models import StrictModel

_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[^\s,;]+")
_HF_TOKEN_PATTERN = re.compile(r"\bhf_[A-Za-z0-9_-]{6,}\b")

_TASK_SCHEMAS: dict[str, type[StrictModel]] = {
    "analysis": AnalystModelOutput,
    "critique": SkepticModelOutput,
    "synthesis": SynthesizerModelOutput,
}


def _safe_provider_error(exc: Exception, *, token: str) -> str:
    """Return a bounded diagnostic without leaking credentials."""

    raw = " ".join(str(exc).split())
    if token:
        raw = raw.replace(token, "[redacted]")
    raw = _HF_TOKEN_PATTERN.sub("[redacted]", raw)
    raw = _BEARER_PATTERN.sub("Bearer [redacted]", raw)
    raw = raw[:500]

    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    parts = ["Hugging Face provider request failed"]
    if status is not None:
        parts.append(f"status={status}")
    if code:
        parts.append(f"code={code}")
    if raw:
        parts.append(raw)
    return ": ".join(parts)


def _schema_for_task(task_name: str) -> type[StrictModel]:
    try:
        return _TASK_SCHEMAS[task_name]
    except KeyError as exc:
        raise StructuredModelError(
            f"no provider JSON schema registered for bounded task: {task_name}"
        ) from exc


@dataclass(frozen=True)
class HuggingFaceStructuredChatClient(HuggingFaceChatClient):
    """Hugging Face chat client hardened for strict structured work products."""

    def complete_json(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on runtime install
            raise ModelConfigurationError(
                "openai package is required for live Hugging Face LLM mode"
            ) from exc

        schema = _schema_for_task(task_name)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": True,
            },
        }

        client = OpenAI(base_url=self.base_url, api_key=self.token)
        request: dict[str, object] = {
            "model": self.model_id,
            "messages": (
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": response_format,
        }

        # Only documented OpenAI-compatible fields are sent. Provider/model
        # compatibility is explicit in MODEL_ID; unsupported extensions are not
        # guessed or silently retried with weaker contracts.
        try:
            completion = client.chat.completions.create(**request)
        except Exception as exc:  # provider/network/auth failures fail closed
            raise StructuredModelError(
                _safe_provider_error(exc, token=self.token)
            ) from exc

        content = completion.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise StructuredModelError("model returned an empty chat response")
        return content
