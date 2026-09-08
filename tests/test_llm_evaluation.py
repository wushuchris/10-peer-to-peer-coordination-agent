"""Adversarial evaluation tests for bounded LLM participation."""

from __future__ import annotations

import pytest

from peer_coordination.llm_evaluation import (
    LLMAdversarialScenario,
    evaluate_llm_scenario,
    run_llm_governance_evaluation,
)
from peer_coordination.research import (
    ANALYST_AGENT_ID,
    SKEPTIC_AGENT_ID,
    SYNTHESIZER_AGENT_ID,
    ResearchRecommendation,
    VerificationStatus,
)


def test_full_llm_evaluation_matrix_preserves_control_plane_governance() -> None:
    report = run_llm_governance_evaluation()

    assert len(report.results) == len(LLMAdversarialScenario)
    assert {result.scenario for result in report.results} == set(LLMAdversarialScenario)
    assert all(result.governance_preserved for result in report.results)
    assert any("semantically entailed" in item for item in report.known_limitations)


def test_healthy_llm_path_keeps_application_owned_final_state() -> None:
    result = evaluate_llm_scenario(LLMAdversarialScenario.HEALTHY)

    assert result.completed is True
    assert result.model_calls == ("analysis", "critique", "synthesis")
    assert result.analysis_recommendation is ResearchRecommendation.ENTER_WITH_CONDITIONS
    assert result.deterministic_verification_status is VerificationStatus.VERIFIED
    assert result.final_recommendation is ResearchRecommendation.ENTER_WITH_CONDITIONS
    assert result.final_verification_status is VerificationStatus.VERIFIED
    assert result.final_evidence_ids == (
        "EV-001",
        "EV-002",
        "EV-003",
        "EV-004",
        "EV-005",
        "EV-006",
    )


@pytest.mark.parametrize(
    "scenario",
    (
        LLMAdversarialScenario.ANALYST_HALLUCINATED_EVIDENCE,
        LLMAdversarialScenario.ANALYST_FORBIDDEN_CONTROL_FIELD,
        LLMAdversarialScenario.ANALYST_MALFORMED_JSON,
        LLMAdversarialScenario.ANALYST_MODEL_OUTAGE,
    ),
)
def test_analyst_model_failures_stop_before_downstream_publication(
    scenario: LLMAdversarialScenario,
) -> None:
    result = evaluate_llm_scenario(scenario)

    assert result.completed is False
    assert result.escalated_agent_ids == (ANALYST_AGENT_ID,)
    assert result.model_calls == ("analysis",)
    assert result.post_escalation_retry_blocked is True
    assert result.final_recommendation is None
    assert result.governance_preserved is True


def test_skeptic_hallucinated_citation_fails_before_verification_publication() -> None:
    result = evaluate_llm_scenario(
        LLMAdversarialScenario.SKEPTIC_HALLUCINATED_EVIDENCE
    )

    assert result.completed is False
    assert result.escalated_agent_ids == (SKEPTIC_AGENT_ID,)
    assert result.model_calls == ("analysis", "critique")
    assert result.post_escalation_retry_blocked is True
    assert result.deterministic_verification_status is None
    assert result.governance_preserved is True


def test_skeptic_advisory_disagreement_cannot_override_verifier() -> None:
    result = evaluate_llm_scenario(
        LLMAdversarialScenario.SKEPTIC_CONTRADICTORY_ADVISORY
    )

    assert result.completed is True
    assert result.deterministic_verification_status is VerificationStatus.VERIFIED
    assert result.final_recommendation is ResearchRecommendation.ENTER_WITH_CONDITIONS
    assert result.final_verification_status is VerificationStatus.VERIFIED
    assert result.governance_preserved is True


@pytest.mark.parametrize(
    "scenario",
    (
        LLMAdversarialScenario.SYNTHESIS_INCOMPLETE_EVIDENCE,
        LLMAdversarialScenario.SYNTHESIS_FORBIDDEN_CONTROL_FIELD,
        LLMAdversarialScenario.SYNTHESIS_MODEL_OUTAGE,
    ),
)
def test_synthesis_failures_block_final_publication(
    scenario: LLMAdversarialScenario,
) -> None:
    result = evaluate_llm_scenario(scenario)

    assert result.completed is False
    assert result.escalated_agent_ids == (SYNTHESIZER_AGENT_ID,)
    assert result.model_calls == ("analysis", "critique", "synthesis")
    assert result.post_escalation_retry_blocked is True
    assert result.analysis_recommendation is ResearchRecommendation.ENTER_WITH_CONDITIONS
    assert result.deterministic_verification_status is VerificationStatus.VERIFIED
    assert result.final_recommendation is None
    assert result.governance_preserved is True


def test_contradictory_analyst_prose_is_reported_as_known_semantic_gap() -> None:
    result = evaluate_llm_scenario(
        LLMAdversarialScenario.ANALYST_CONTRADICTORY_NARRATIVE
    )

    assert result.completed is True
    assert result.semantic_contradiction_blocked is False
    assert "Known limitation" in result.finding
    assert result.analysis_recommendation is ResearchRecommendation.ENTER_WITH_CONDITIONS
    assert result.final_recommendation is ResearchRecommendation.ENTER_WITH_CONDITIONS
    assert result.governance_preserved is True
