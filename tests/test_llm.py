"""Tests for bounded LLM work handlers without external model calls."""

from __future__ import annotations

import json
from collections import defaultdict

import pytest

from peer_coordination.llm import (
    HuggingFaceChatClient,
    ModelConfigurationError,
    StructuredModelError,
    _json_object_text,
    build_llm_research_demo_system,
)
from peer_coordination.models import PeerStatus
from peer_coordination.research import (
    ANALYST_AGENT_ID,
    FINAL_WORK_PRODUCT_ID,
    SYNTHESIZER_AGENT_ID,
    FinalBriefArtifact,
    ResearchRecommendation,
    VerificationStatus,
    make_initial_research_request,
)


class QueueJsonModel:
    """Deterministic fake model keyed by bounded task name."""

    def __init__(self, responses: dict[str, list[dict] | list[str]]) -> None:
        self._responses: dict[str, list[str]] = defaultdict(list)
        for task, values in responses.items():
            for value in values:
                self._responses[task].append(
                    value if isinstance(value, str) else json.dumps(value)
                )
        self.calls: list[str] = []

    def complete_json(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        assert "routing" in system_prompt or "publication" in system_prompt
        assert "mission" in user_prompt
        self.calls.append(task_name)
        if not self._responses[task_name]:
            raise AssertionError(f"no fake response queued for {task_name}")
        return self._responses[task_name].pop(0)


def healthy_responses() -> dict[str, list[dict]]:
    return {
        "analysis": [
            {
                "summary": (
                    "The evidence supports market entry with explicit conditions around "
                    "certification, incumbent competition, and service coverage."
                ),
                "claims": [
                    {
                        "text": (
                            "Demand, channel interest, and modeled economics provide positive "
                            "entry signals."
                        ),
                        "evidence_ids": ["EV-001", "EV-002", "EV-003"],
                    },
                    {
                        "text": (
                            "Certification cost, incumbent concentration, and service coverage "
                            "remain material launch constraints."
                        ),
                        "evidence_ids": ["EV-004", "EV-005", "EV-006"],
                    },
                ],
                "assumptions": [
                    "The supplied synthetic evidence is the complete evidence set for this demo."
                ],
                "confidence": 0.78,
            }
        ],
        "critique": [
            {
                "review_summary": (
                    "The evidence links are structurally sound; service coverage remains a "
                    "business risk rather than a verification defect."
                ),
                "concerns": [
                    {
                        "issue": "Initial service coverage reaches only part of the target base.",
                        "evidence_ids": ["EV-006"],
                    }
                ],
            }
        ],
        "synthesis": [
            {
                "brief": (
                    "Asteria should enter the fictional Borealis market with conditions. "
                    "Demand, channel interest, and modeled economics are favorable, while "
                    "certification, incumbent concentration, and service coverage require "
                    "explicit launch mitigations."
                ),
                "evidence_ids": [
                    "EV-001",
                    "EV-002",
                    "EV-003",
                    "EV-004",
                    "EV-005",
                    "EV-006",
                ],
            }
        ],
    }


def drive_system(model: QueueJsonModel, cycles: int = 8):
    system = build_llm_research_demo_system(model)
    initiator = system.runtime.get_peer(SYNTHESIZER_AGENT_ID)
    initiator.broadcast_mission()
    system.runtime.run_round()
    system.runtime.run_round()
    initiator.broadcast_work_request(make_initial_research_request())

    for _ in range(cycles):
        system.runtime.run_round()
        system.runtime.run_execution_round()
        if FINAL_WORK_PRODUCT_ID in initiator.work_products:
            break
    return system


def test_bounded_llm_mission_produces_verified_final_brief() -> None:
    model = QueueJsonModel(healthy_responses())
    system = drive_system(model)

    final = system.runtime.get_peer(SYNTHESIZER_AGENT_ID).work_products[FINAL_WORK_PRODUCT_ID]
    artifact = FinalBriefArtifact.model_validate(final.metadata["artifact"])

    assert artifact.recommendation is ResearchRecommendation.ENTER_WITH_CONDITIONS
    assert artifact.verification_status is VerificationStatus.VERIFIED
    assert artifact.brief.startswith("Asteria should enter")
    assert final.metadata["llm"] == {"bounded": True}
    assert model.calls == ["analysis", "critique", "synthesis"]


def test_llm_skeptic_concern_cannot_override_deterministic_verification() -> None:
    responses = healthy_responses()
    responses["critique"][0]["concerns"] = [
        {
            "issue": "I would still prefer more service coverage before launch.",
            "evidence_ids": ["EV-006"],
        }
    ]
    model = QueueJsonModel(responses)
    system = drive_system(model)

    final = system.runtime.get_peer(SYNTHESIZER_AGENT_ID).work_products[FINAL_WORK_PRODUCT_ID]
    artifact = FinalBriefArtifact.model_validate(final.metadata["artifact"])
    assert artifact.verification_status is VerificationStatus.VERIFIED


def test_analyst_invented_evidence_id_fails_closed() -> None:
    responses = healthy_responses()
    responses["analysis"][0]["claims"][0]["evidence_ids"] = ["EV-999"]
    model = QueueJsonModel(responses)
    system = drive_system(model)

    analyst = system.runtime.get_peer(ANALYST_AGENT_ID)
    synth = system.runtime.get_peer(SYNTHESIZER_AGENT_ID)
    assert analyst.state.status is PeerStatus.ESCALATED
    assert FINAL_WORK_PRODUCT_ID not in synth.work_products
    assert model.calls == ["analysis"]


def test_analyst_cannot_return_application_owned_recommendation() -> None:
    responses = healthy_responses()
    responses["analysis"][0]["recommendation"] = "do_not_enter"
    model = QueueJsonModel(responses)
    system = drive_system(model)

    assert system.runtime.get_peer(ANALYST_AGENT_ID).state.status is PeerStatus.ESCALATED
    assert FINAL_WORK_PRODUCT_ID not in system.runtime.get_peer(
        SYNTHESIZER_AGENT_ID
    ).work_products


def test_synthesizer_must_preserve_exact_verified_evidence_set() -> None:
    responses = healthy_responses()
    responses["synthesis"][0]["evidence_ids"] = [
        "EV-001",
        "EV-002",
        "EV-003",
        "EV-004",
        "EV-005",
    ]
    model = QueueJsonModel(responses)
    system = drive_system(model)

    synth = system.runtime.get_peer(SYNTHESIZER_AGENT_ID)
    assert synth.state.status is PeerStatus.ESCALATED
    assert FINAL_WORK_PRODUCT_ID not in synth.work_products
    assert model.calls == ["analysis", "critique", "synthesis"]


def test_malformed_model_json_fails_closed() -> None:
    responses = healthy_responses()
    responses["analysis"] = ["not json"]
    model = QueueJsonModel(responses)
    system = drive_system(model)

    assert system.runtime.get_peer(ANALYST_AGENT_ID).state.status is PeerStatus.ESCALATED
    assert model.calls == ["analysis"]


def test_json_normalizer_accepts_one_json_code_fence() -> None:
    normalized = _json_object_text('```json\n{"summary": "ok"}\n```')
    assert json.loads(normalized) == {"summary": "ok"}


def test_json_normalizer_rejects_non_object_payload() -> None:
    with pytest.raises(StructuredModelError, match="one JSON object"):
        _json_object_text('["not", "an", "object"]')


def test_hugging_face_client_requires_runtime_secret_and_model() -> None:
    with pytest.raises(ModelConfigurationError, match="HF_TOKEN"):
        HuggingFaceChatClient.from_env(
            {"HF_TOKEN": "your_runtime_token", "MODEL_ID": "your_model_id"}
        )

    with pytest.raises(ModelConfigurationError, match="MODEL_ID"):
        HuggingFaceChatClient.from_env({"HF_TOKEN": "hf_example", "MODEL_ID": ""})


def test_hugging_face_client_requires_https_base_url() -> None:
    with pytest.raises(ModelConfigurationError, match="https"):
        HuggingFaceChatClient.from_env(
            {
                "HF_TOKEN": "hf_example",
                "MODEL_ID": "example/model",
                "HF_BASE_URL": "http://example.test/v1",
            }
        )
