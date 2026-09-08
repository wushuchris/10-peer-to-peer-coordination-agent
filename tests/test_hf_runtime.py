"""Regression tests for the production Hugging Face JSON runtime adapter."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from peer_coordination.demo import DemoMode, run_demo
from peer_coordination.hf_runtime import HuggingFaceStructuredChatClient
from peer_coordination.llm import StructuredModelError


class RecordingOpenAI:
    last_request: dict[str, object] | None = None

    def __init__(self, *, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        type(self).last_request = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        )


class FailingOpenAI:
    def __init__(self, *, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        del kwargs
        raise RuntimeError("provider rejected Bearer hf_secret_runtime_token")


class FailingJsonModel:
    def complete_json(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        del task_name, system_prompt, user_prompt
        raise StructuredModelError("synthetic provider outage")


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, client_type: type) -> None:
    module = ModuleType("openai")
    module.OpenAI = client_type
    monkeypatch.setitem(sys.modules, "openai", module)


def _run_recorded_call(
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_name: str,
) -> dict[str, object]:
    _install_fake_openai(monkeypatch, RecordingOpenAI)
    RecordingOpenAI.last_request = None
    client = HuggingFaceStructuredChatClient(
        model_id="openai/gpt-oss-120b:fireworks-ai",
        token="hf_example",
    )
    client.complete_json(
        task_name=task_name,
        system_prompt="system",
        user_prompt="user",
    )
    request = RecordingOpenAI.last_request
    assert request is not None
    return request


def test_analysis_request_uses_strict_pydantic_json_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _run_recorded_call(monkeypatch, task_name="analysis")
    response_format = request["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    json_schema = response_format["json_schema"]
    assert isinstance(json_schema, dict)
    assert json_schema["name"] == "AnalystModelOutput"
    assert json_schema["strict"] is True
    schema = json_schema["schema"]
    assert isinstance(schema, dict)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert {"summary", "claims", "assumptions", "confidence"}.issubset(
        set(schema["properties"])
    )
    assert "extra_body" not in request


def test_each_bounded_task_gets_its_own_provider_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "analysis": "AnalystModelOutput",
        "critique": "SkepticModelOutput",
        "synthesis": "SynthesizerModelOutput",
    }
    for task_name, schema_name in expected.items():
        request = _run_recorded_call(monkeypatch, task_name=task_name)
        response_format = request["response_format"]
        assert isinstance(response_format, dict)
        json_schema = response_format["json_schema"]
        assert isinstance(json_schema, dict)
        assert json_schema["name"] == schema_name
        assert json_schema["strict"] is True


def test_unknown_bounded_task_fails_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch, RecordingOpenAI)
    RecordingOpenAI.last_request = None
    client = HuggingFaceStructuredChatClient(
        model_id="openai/gpt-oss-120b:fireworks-ai",
        token="hf_example",
    )

    with pytest.raises(StructuredModelError, match="no provider JSON schema registered"):
        client.complete_json(
            task_name="unknown",
            system_prompt="system",
            user_prompt="user",
        )

    assert RecordingOpenAI.last_request is None


def test_provider_failure_is_redacted_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch, FailingOpenAI)
    client = HuggingFaceStructuredChatClient(
        model_id="openai/gpt-oss-120b:fireworks-ai",
        token="hf_secret_runtime_token",
    )

    with pytest.raises(StructuredModelError) as exc_info:
        client.complete_json(
            task_name="analysis",
            system_prompt="system",
            user_prompt="user",
        )

    message = str(exc_info.value)
    assert "Hugging Face provider request failed" in message
    assert "hf_secret_runtime_token" not in message
    assert "Bearer [redacted]" in message


def test_demo_surfaces_fail_closed_handler_diagnostic() -> None:
    snapshot = run_demo(DemoMode.LLM_ASSISTED, model=FailingJsonModel())

    assert snapshot.completed is False
    assert snapshot.error is not None
    assert "Analyst" in snapshot.error
    assert "StructuredModelError" in snapshot.error
    assert "synthetic provider outage" in snapshot.error
