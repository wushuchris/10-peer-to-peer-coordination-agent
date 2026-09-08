"""End-to-end tests for the synthetic Agent 10 research mission."""

import pytest
from pydantic import ValidationError

from peer_coordination.models import PeerRole, WorkRequestPayload
from peer_coordination.policy import PolicyAction
from peer_coordination.research import (
    ANALYSIS_CAPABILITY,
    ANALYSIS_WORK_ID,
    FINAL_WORK_PRODUCT_ID,
    SOURCE_AGENT_ID,
    SOURCE_CAPABILITY,
    SOURCE_WORK_ID,
    SYNTHESIZER_AGENT_ID,
    FinalBriefArtifact,
    ResearchRecommendation,
    VerificationStatus,
    build_research_demo_system,
    build_synthetic_corpus,
    make_initial_research_request,
    run_deterministic_research_demo,
)
from peer_coordination.runtime import WorkExecutionStatus


def establish_roles(system) -> None:
    initiator = system.runtime.get_peer(SYNTHESIZER_AGENT_ID)
    initiator.broadcast_mission()
    system.runtime.run_round()
    system.runtime.run_round()


def test_synthetic_corpus_is_small_deterministic_and_public_safe() -> None:
    corpus = build_synthetic_corpus()
    items = corpus.all_items()

    assert tuple(item.evidence_id for item in items) == (
        "EV-001",
        "EV-002",
        "EV-003",
        "EV-004",
        "EV-005",
        "EV-006",
    )
    assert all("synthetic" in item.finding.lower() for item in items)


def test_work_request_rejects_duplicate_input_products() -> None:
    with pytest.raises(ValidationError, match="input_work_product_ids must not contain duplicates"):
        WorkRequestPayload(
            work_id="work-test-001",
            requested_capability=ANALYSIS_CAPABILITY,
            summary="Analyze supplied work products.",
            input_work_product_ids=("wp-001", "wp-001"),
        )


def test_four_peers_self_select_required_roles() -> None:
    system = build_research_demo_system()
    establish_roles(system)

    claimed = {
        system.runtime.get_peer(agent_id).state.claimed_role
        for agent_id in (
            "peer-source-01",
            "peer-analyst-01",
            "peer-skeptic-01",
            "peer-synth-01",
        )
    }

    assert claimed == {
        PeerRole.SOURCE_FINDER,
        PeerRole.ANALYST,
        PeerRole.SKEPTIC,
        PeerRole.SYNTHESIZER,
    }


def test_open_source_request_is_broadcast_but_only_source_peer_accepts() -> None:
    system = build_research_demo_system()
    establish_roles(system)

    initiator = system.runtime.get_peer(SYNTHESIZER_AGENT_ID)
    initiator.broadcast_work_request(make_initial_research_request())
    turns = system.runtime.run_round()

    source = system.runtime.get_peer(SOURCE_AGENT_ID)
    assert source.state.accepted_work_ids == (SOURCE_WORK_ID,)

    non_source_decisions = [
        decision
        for turn in turns
        if turn.agent_id != SOURCE_AGENT_ID
        for decision in turn.decisions
        if decision.work_id == SOURCE_WORK_ID
    ]
    assert non_source_decisions
    assert all(decision.action is PolicyAction.IGNORE for decision in non_source_decisions)


def test_missing_named_dependency_waits_instead_of_executing() -> None:
    system = build_research_demo_system()
    establish_roles(system)

    requester = system.runtime.get_peer(SYNTHESIZER_AGENT_ID)
    requester.broadcast_work_request(
        WorkRequestPayload(
            work_id=ANALYSIS_WORK_ID,
            requested_capability=ANALYSIS_CAPABILITY,
            summary="Analyze a work product that has not arrived.",
            input_work_product_ids=("wp-missing",),
        )
    )
    system.runtime.run_round()
    executions = system.runtime.run_execution_round()

    analyst_execution = next(
        execution for execution in executions if execution.work_id == ANALYSIS_WORK_ID
    )
    assert analyst_execution.status is WorkExecutionStatus.WAITING
    assert "wp-missing" in analyst_execution.detail


def test_end_to_end_demo_produces_verified_final_brief() -> None:
    final = run_deterministic_research_demo()
    artifact = FinalBriefArtifact.model_validate(final.metadata["artifact"])

    assert final.work_product_id == FINAL_WORK_PRODUCT_ID
    assert artifact.recommendation is ResearchRecommendation.ENTER_WITH_CONDITIONS
    assert artifact.verification_status is VerificationStatus.VERIFIED
    assert artifact.evidence_ids == (
        "EV-001",
        "EV-002",
        "EV-003",
        "EV-004",
        "EV-005",
        "EV-006",
    )
    assert "certification" in artifact.brief.lower()
    assert "service coverage" in artifact.brief.lower()


def test_demo_is_deterministic_at_the_work_product_boundary() -> None:
    first = run_deterministic_research_demo()
    second = run_deterministic_research_demo()

    assert first == second
