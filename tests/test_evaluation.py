from peer_coordination.evaluation import (
    CoordinationArchitecture,
    EvaluationScenario,
    run_architecture_evaluation,
    run_centralized_evaluation,
    run_peer_to_peer_evaluation,
)
from peer_coordination.research import FINAL_WORK_PRODUCT_ID


def test_happy_path_both_architectures_complete_same_work_product() -> None:
    p2p = run_peer_to_peer_evaluation(EvaluationScenario.HAPPY_PATH)
    centralized = run_centralized_evaluation(EvaluationScenario.HAPPY_PATH)

    assert p2p.completed is True
    assert centralized.completed is True
    assert p2p.final_product_id == FINAL_WORK_PRODUCT_ID
    assert centralized.final_product_id == FINAL_WORK_PRODUCT_ID
    assert p2p.final_product_digest == centralized.final_product_digest


def test_happy_path_exposes_coordination_overhead_and_authority_difference() -> None:
    p2p = run_peer_to_peer_evaluation(EvaluationScenario.HAPPY_PATH)
    centralized = run_centralized_evaluation(EvaluationScenario.HAPPY_PATH)

    assert p2p.architecture is CoordinationArchitecture.PEER_TO_PEER
    assert centralized.architecture is CoordinationArchitecture.CENTRALIZED
    assert p2p.protocol_messages > 0
    assert p2p.message_deliveries >= p2p.protocol_messages
    assert centralized.protocol_messages == 0
    assert centralized.message_deliveries == 0
    assert p2p.central_coordinator_required is False
    assert centralized.central_coordinator_required is True
    assert p2p.audit_trace_items > 0
    assert centralized.audit_trace_items > 0


def test_central_coordinator_outage_only_breaks_centralized_architecture() -> None:
    p2p = run_peer_to_peer_evaluation(EvaluationScenario.COORDINATOR_OUTAGE)
    centralized = run_centralized_evaluation(EvaluationScenario.COORDINATOR_OUTAGE)

    assert p2p.completed is True
    assert p2p.failure_code is None
    assert centralized.completed is False
    assert centralized.failure_code == "coordinator_unavailable"
    assert centralized.work_executions == 0


def test_unavailable_analyst_fails_closed_in_both_architectures() -> None:
    p2p = run_peer_to_peer_evaluation(EvaluationScenario.UNAVAILABLE_ANALYST)
    centralized = run_centralized_evaluation(EvaluationScenario.UNAVAILABLE_ANALYST)

    assert p2p.completed is False
    assert centralized.completed is False
    assert p2p.final_product_id is None
    assert centralized.final_product_id is None
    assert p2p.failure_code == "unavailable_required_role"
    assert centralized.failure_code == "assigned_peer_unavailable"


def test_evaluation_report_covers_all_bounded_scenarios() -> None:
    report = run_architecture_evaluation()

    assert tuple(item.scenario for item in report.comparisons) == tuple(EvaluationScenario)
    assert all(item.observations for item in report.comparisons)
    assert all(item.peer_to_peer.scenario is item.scenario for item in report.comparisons)
    assert all(item.centralized.scenario is item.scenario for item in report.comparisons)


def test_evaluation_metrics_are_deterministic_across_repeated_runs() -> None:
    first = run_architecture_evaluation()
    second = run_architecture_evaluation()

    assert first == second


def test_baseline_does_not_claim_peer_to_peer_message_efficiency() -> None:
    p2p = run_peer_to_peer_evaluation(EvaluationScenario.HAPPY_PATH)
    centralized = run_centralized_evaluation(EvaluationScenario.HAPPY_PATH)

    # The centralized baseline can be mechanically leaner precisely because one
    # component owns assignment. The test records the tradeoff instead of treating
    # lower coordination overhead as proof of a superior architecture.
    assert centralized.coordination_steps == centralized.work_executions
    assert p2p.protocol_messages > centralized.protocol_messages
    assert p2p.central_coordinator_required is False
    assert centralized.central_coordinator_required is True
