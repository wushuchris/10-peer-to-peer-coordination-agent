from __future__ import annotations

from datetime import datetime, timezone

from peer_coordination.challenge import ChallengeCapablePeerAgent
from peer_coordination.models import (
    ChallengePayload,
    Mission,
    PeerProfile,
    PeerRole,
    PeerStatus,
    WorkRequestPayload,
    WorkResultPayload,
)
from peer_coordination.registry import PeerRegistry
from peer_coordination.resilience import (
    FailureContainmentRuntime,
    FailureIssueCode,
    FailurePolicy,
    FailureSeverity,
)
from peer_coordination.runtime import PeerAgent, WorkExecutionContext, WorkExecutionOutcome
from peer_coordination.transport import MessageBus


def mission(*roles: PeerRole) -> Mission:
    return Mission(
        mission_id="mission-resilience-001",
        objective="Exercise deterministic failure containment.",
        required_roles=roles,
    )


def profile(
    agent_id: str,
    *,
    role: PeerRole,
    capability: str,
    available: bool = True,
) -> PeerProfile:
    return PeerProfile(
        agent_id=agent_id,
        display_name=agent_id,
        capabilities=(capability,),
        eligible_roles=(role,),
        available=available,
    )


def issue_codes(runtime: FailureContainmentRuntime) -> set[FailureIssueCode]:
    return {issue.code for issue in runtime.health.issues}


def test_unavailable_required_role_blocks_immediately() -> None:
    m = mission(PeerRole.ANALYST)
    registry = PeerRegistry(
        (profile("peer-analyst", role=PeerRole.ANALYST, capability="analysis", available=False),)
    )
    bus = MessageBus(registry)
    peer = PeerAgent(agent_id="peer-analyst", mission=m, registry=registry, bus=bus)
    runtime = FailureContainmentRuntime(
        [peer],
        registry=registry,
        policy=FailurePolicy(no_progress_rounds=10),
    )

    result = runtime.run_cycle()

    assert result.health.blocked is True
    assert FailureIssueCode.UNAVAILABLE_REQUIRED_ROLE in issue_codes(runtime)


def test_available_but_absent_peer_becomes_unclaimed_role_failure_after_grace() -> None:
    m = mission(PeerRole.SYNTHESIZER, PeerRole.ANALYST)
    profiles = (
        profile("peer-synth", role=PeerRole.SYNTHESIZER, capability="synthesis"),
        profile("peer-ghost", role=PeerRole.ANALYST, capability="analysis"),
    )
    registry = PeerRegistry(profiles)
    bus = MessageBus(registry)
    synth = PeerAgent(agent_id="peer-synth", mission=m, registry=registry, bus=bus)
    runtime = FailureContainmentRuntime(
        [synth],
        registry=registry,
        policy=FailurePolicy(role_grace_rounds=2, no_progress_rounds=10),
    )

    first = runtime.run_cycle()
    second = runtime.run_cycle()

    assert first.health.blocked is False
    assert second.health.blocked is True
    issue = next(
        item
        for item in second.health.issues
        if item.code is FailureIssueCode.UNCLAIMED_REQUIRED_ROLE
    )
    assert issue.role is PeerRole.ANALYST


def test_persistent_duplicate_role_claim_fails_closed() -> None:
    m = mission(PeerRole.ANALYST)
    profiles = (
        profile("peer-a", role=PeerRole.ANALYST, capability="analysis"),
        profile("peer-b", role=PeerRole.ANALYST, capability="analysis"),
    )
    registry = PeerRegistry(profiles)
    bus = MessageBus(registry)
    peer_a = PeerAgent(agent_id="peer-a", mission=m, registry=registry, bus=bus)
    peer_b = PeerAgent(agent_id="peer-b", mission=m, registry=registry, bus=bus)
    peer_a.state = peer_a.state.model_copy(
        update={"claimed_role": PeerRole.ANALYST, "status": PeerStatus.ACTIVE}
    )
    peer_b.state = peer_b.state.model_copy(
        update={"claimed_role": PeerRole.ANALYST, "status": PeerStatus.ACTIVE}
    )
    runtime = FailureContainmentRuntime(
        [peer_a, peer_b],
        registry=registry,
        policy=FailurePolicy(role_conflict_rounds=1, no_progress_rounds=10),
    )

    result = runtime.run_cycle()

    issue = next(
        item for item in result.health.issues if item.code is FailureIssueCode.DUPLICATE_ROLE_CLAIM
    )
    assert issue.severity is FailureSeverity.BLOCKING
    assert result.health.blocked is True


def build_missing_dependency_runtime(
    *,
    dependency_wait_rounds: int,
    no_progress_rounds: int,
) -> tuple[FailureContainmentRuntime, PeerAgent]:
    m = mission(PeerRole.SYNTHESIZER, PeerRole.ANALYST)
    profiles = (
        profile("peer-synth", role=PeerRole.SYNTHESIZER, capability="synthesis"),
        profile("peer-analyst", role=PeerRole.ANALYST, capability="analysis"),
    )
    registry = PeerRegistry(profiles)
    bus = MessageBus(registry)
    synth = PeerAgent(agent_id="peer-synth", mission=m, registry=registry, bus=bus)
    analyst = PeerAgent(
        agent_id="peer-analyst",
        mission=m,
        registry=registry,
        bus=bus,
        handlers={
            "analysis": lambda context: WorkExecutionOutcome(
                result=WorkResultPayload(
                    work_id=context.request.work_id,
                    work_product_id=f"wp:{context.request.work_id}",
                    summary="Should never execute while its dependency is missing.",
                )
            )
        },
    )
    runtime = FailureContainmentRuntime(
        [synth, analyst],
        registry=registry,
        policy=FailurePolicy(
            role_grace_rounds=5,
            dependency_wait_rounds=dependency_wait_rounds,
            no_progress_rounds=no_progress_rounds,
        ),
    )
    runtime.run_cycle()
    runtime.run_cycle()
    synth.broadcast_work_request(
        WorkRequestPayload(
            work_id="work-needs-input",
            requested_capability="analysis",
            summary="Analyze the missing upstream product.",
            input_work_product_ids=("wp:missing",),
        )
    )
    return runtime, analyst


def test_missing_dependency_escalates_after_bounded_wait() -> None:
    runtime, analyst = build_missing_dependency_runtime(
        dependency_wait_rounds=2,
        no_progress_rounds=10,
    )

    first = runtime.run_cycle()
    second = runtime.run_cycle()

    assert first.health.blocked is False
    assert second.health.blocked is True
    issue = next(
        item for item in second.health.issues if item.code is FailureIssueCode.MISSING_DEPENDENCY
    )
    assert issue.agent_id == analyst.agent_id
    assert issue.work_id == "work-needs-input"
    assert "wp:missing" in issue.detail


def test_no_progress_blocks_before_longer_dependency_timeout() -> None:
    runtime, _ = build_missing_dependency_runtime(
        dependency_wait_rounds=10,
        no_progress_rounds=1,
    )

    runtime.run_cycle()
    result = runtime.run_cycle()

    assert result.health.blocked is True
    assert FailureIssueCode.NO_PROGRESS in {item.code for item in result.health.issues}
    assert FailureIssueCode.MISSING_DEPENDENCY not in {
        item.code for item in result.health.issues
    }


def test_unanswered_challenge_times_out_without_reassignment() -> None:
    m = mission(PeerRole.SKEPTIC)
    profiles = (
        profile("peer-skeptic", role=PeerRole.SKEPTIC, capability="verification"),
        profile("peer-ghost", role=PeerRole.ANALYST, capability="analysis"),
    )
    registry = PeerRegistry(profiles)
    bus = MessageBus(registry)
    skeptic = ChallengeCapablePeerAgent(
        agent_id="peer-skeptic",
        mission=m,
        registry=registry,
        bus=bus,
    )
    skeptic.issue_challenge(
        recipient_id="peer-ghost",
        challenge=ChallengePayload(
            challenge_id="challenge-timeout-001",
            work_id="work-analysis-001",
            issue="Respond to this deterministic review challenge.",
        ),
    )
    runtime = FailureContainmentRuntime(
        [skeptic],
        registry=registry,
        policy=FailurePolicy(
            challenge_response_rounds=2,
            role_grace_rounds=5,
            no_progress_rounds=10,
        ),
    )

    first = runtime.run_cycle()
    second = runtime.run_cycle()

    assert first.health.blocked is False
    assert second.health.blocked is True
    issue = next(
        item
        for item in second.health.issues
        if item.code is FailureIssueCode.OPEN_CHALLENGE_TIMEOUT
    )
    assert issue.challenge_id == "challenge-timeout-001"
    assert issue.agent_id == "peer-skeptic"
    assert not hasattr(runtime, "assign_work")
    assert not hasattr(runtime, "select_worker")


def test_blocked_runtime_contains_future_cycles() -> None:
    m = mission(PeerRole.ANALYST)
    registry = PeerRegistry(
        (profile("peer-analyst", role=PeerRole.ANALYST, capability="analysis", available=False),)
    )
    bus = MessageBus(registry)
    peer = PeerAgent(agent_id="peer-analyst", mission=m, registry=registry, bus=bus)
    runtime = FailureContainmentRuntime([peer], registry=registry)

    blocked = runtime.run_cycle()
    contained = runtime.run_cycle()

    assert blocked.health.blocked is True
    assert contained.contained is True
    assert contained.turns == ()
    assert contained.executions == ()
    assert contained.health == blocked.health


def test_completed_role_release_is_remembered_as_fulfilled() -> None:
    m = mission(PeerRole.SYNTHESIZER, PeerRole.ANALYST)
    profiles = (
        profile("peer-synth", role=PeerRole.SYNTHESIZER, capability="synthesis"),
        profile("peer-analyst", role=PeerRole.ANALYST, capability="analysis"),
    )
    registry = PeerRegistry(profiles)
    bus = MessageBus(registry)
    synth = PeerAgent(agent_id="peer-synth", mission=m, registry=registry, bus=bus)

    def handler(context: WorkExecutionContext) -> WorkExecutionOutcome:
        return WorkExecutionOutcome(
            result=WorkResultPayload(
                work_id=context.request.work_id,
                work_product_id="wp:analysis-done",
                summary="Deterministic analysis complete.",
            )
        )

    analyst = PeerAgent(
        agent_id="peer-analyst",
        mission=m,
        registry=registry,
        bus=bus,
        handlers={"analysis": handler},
    )
    runtime = FailureContainmentRuntime(
        [synth, analyst],
        registry=registry,
        policy=FailurePolicy(role_grace_rounds=3, no_progress_rounds=10),
    )

    runtime.run_cycle()
    runtime.run_cycle()
    synth.broadcast_work_request(
        WorkRequestPayload(
            work_id="work-analysis-done",
            requested_capability="analysis",
            summary="Complete one bounded analysis task.",
        )
    )
    runtime.run_cycle()
    assert analyst.state.status is PeerStatus.COMPLETED

    release_cycle = runtime.run_cycle()
    after_release = runtime.run_cycle()

    assert analyst.state.claimed_role is None
    assert analyst.state.status is PeerStatus.COMPLETED
    assert FailureIssueCode.UNCLAIMED_REQUIRED_ROLE not in {
        item.code for item in release_cycle.health.issues
    }
    assert FailureIssueCode.UNCLAIMED_REQUIRED_ROLE not in {
        item.code for item in after_release.health.issues
    }
