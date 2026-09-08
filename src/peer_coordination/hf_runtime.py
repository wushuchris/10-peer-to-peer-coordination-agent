"""Production Hugging Face runtime adapter for bounded JSON work products.

This module keeps provider-specific request shaping outside the coordination
control plane. The live adapter requests provider-level JSON output and then
passes the response through the same strict parser and Pydantic validation used
by the bounded work handlers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import (
    HuggingFaceChatClient,
    ModelConfigurationError,
    StructuredModelError,
)

_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[^\s,;]+")
_HF_TOKEN_PATTERN = re.compile(r"\bhf_[A-Za-z0-9_-]{6,}\b")


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


@dataclass(frozen=True)
class HuggingFaceStructuredChatClient(HuggingFaceChatClient):
    """Hugging Face chat client hardened for strict JSON work-product calls."""

    def complete_json(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        del task_name  # task labels are local audit metadata, not provider routing.
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on runtime install
            raise ModelConfigurationError(
                "openai package is required for live Hugging Face LLM mode"
            ) from exc

        client = OpenAI(base_url=self.base_url, api_key=self.token)
        request: dict[str, object] = {
            "model": self.model_id,
            "messages": (
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }

        # Keep the live request to provider-supported OpenAI-compatible fields.
        # Model-specific chat-template arguments are intentionally excluded here:
        # unsupported provider extensions must not prevent a bounded JSON call.
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
