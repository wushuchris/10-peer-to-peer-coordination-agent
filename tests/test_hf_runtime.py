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


def test_qwen_live_request_uses_provider_supported_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch, RecordingOpenAI)
    RecordingOpenAI.last_request = None
    client = HuggingFaceStructuredChatClient(
        model_id="Qwen/Qwen3.8-27B",
        token="hf_example",
    )

    response = client.complete_json(
        task_name="analysis",
        system_prompt="system",
        user_prompt="user",
    )

    assert response == '{"ok": true}'
    request = RecordingOpenAI.last_request
    assert request is not None
    assert request["response_format"] == {"type": "json_object"}
    assert "extra_body" not in request


def test_non_qwen_live_request_uses_same_provider_supported_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch, RecordingOpenAI)
    RecordingOpenAI.last_request = None
    client = HuggingFaceStructuredChatClient(
        model_id="openai/gpt-oss-20b",
        token="hf_example",
    )

    client.complete_json(
        task_name="analysis",
        system_prompt="system",
        user_prompt="user",
    )

    request = RecordingOpenAI.last_request
    assert request is not None
    assert request["response_format"] == {"type": "json_object"}
    assert "extra_body" not in request


def test_provider_failure_is_redacted_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch, FailingOpenAI)
    client = HuggingFaceStructuredChatClient(
        model_id="Qwen/Qwen3.8-27B",
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
