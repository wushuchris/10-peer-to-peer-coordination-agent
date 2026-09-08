"""Tests for Agent 10 deterministic local coordination policy."""

from peer_coordination.models import (
    Mission,
    PeerProfile,
    PeerRole,
    PeerState,
    PeerStatus,
    WorkRequestPayload,
)
from peer_coordination.policy import LocalCoordinationPolicy, PolicyAction


def make_mission() -> Mission:
    return Mission(
        mission_id="mission-001",
        objective="Prepare an evidence-grounded market expansion recommendation.",
        required_roles=(
            PeerRole.SOURCE_FINDER,
            PeerRole.ANALYST,
            PeerRole.SKEPTIC,
            PeerRole.SYNTHESIZER,
        ),
    )


def make_profile(
    *,
    agent_id: str = "peer-analyst-01",
    available: bool = True,
    eligible_roles: tuple[PeerRole, ...] = (PeerRole.ANALYST,),
    capabilities: tuple[str, ...] = ("analysis",),
) -> PeerProfile:
    return PeerProfile(
        agent_id=agent_id,
        display_name=agent_id,
        capabilities=capabilities,
        eligible_roles=eligible_roles,
        available=available,
    )


def make_state(
    *,
    agent_id: str = "peer-analyst-01",
    mission_id: str = "mission-001",
    status: PeerStatus = PeerStatus.IDLE,
    claimed_role: PeerRole | None = None,
) -> PeerState:
    return PeerState(
        agent_id=agent_id,
        mission_id=mission_id,
        status=status,
        claimed_role=claimed_role,
    )


def test_unclaimed_eligible_role_is_claimed() -> None:
    decision = LocalCoordinationPolicy().decide_role(
        profile=make_profile(),
        state=make_state(),
        mission=make_mission(),
        observed_claims={},
    )

    assert decision.action is PolicyAction.CLAIM_ROLE
    assert decision.role is PeerRole.ANALYST


def test_claimed_role_is_not_stolen() -> None:
    decision = LocalCoordinationPolicy().decide_role(
        profile=make_profile(),
        state=make_state(),
        mission=make_mission(),
        observed_claims={PeerRole.ANALYST: ("peer-analyst-02",)},
    )

    assert decision.action is PolicyAction.WAIT


def test_peer_with_no_required_eligible_role_ignores_mission() -> None:
    decision = LocalCoordinationPolicy().decide_role(
        profile=make_profile(eligible_roles=(PeerRole.ANALYST,)),
        state=make_state(),
        mission=Mission(
            mission_id="mission-001",
            objective="Find evidence.",
            required_roles=(PeerRole.SOURCE_FINDER,),
        ),
        observed_claims={},
    )

    assert decision.action is PolicyAction.IGNORE


def test_existing_valid_role_is_retained() -> None:
    decision = LocalCoordinationPolicy().decide_role(
        profile=make_profile(),
        state=make_state(status=PeerStatus.ACTIVE, claimed_role=PeerRole.ANALYST),
        mission=make_mission(),
        observed_claims={PeerRole.ANALYST: ("peer-analyst-01",)},
    )

    assert decision.action is PolicyAction.WAIT
    assert decision.role is PeerRole.ANALYST


def test_duplicate_role_claim_uses_deterministic_tie_breaker() -> None:
    decision = LocalCoordinationPolicy().decide_role(
        profile=make_profile(agent_id="peer-analyst-02"),
        state=make_state(
            agent_id="peer-analyst-02",
            status=PeerStatus.ACTIVE,
            claimed_role=PeerRole.ANALYST,
        ),
        mission=make_mission(),
        observed_claims={PeerRole.ANALYST: ("peer-analyst-01", "peer-analyst-02")},
    )

    assert decision.action is PolicyAction.RELEASE_ROLE
    assert decision.role is PeerRole.ANALYST
    assert "peer-analyst-01" in decision.reason


def test_duplicate_role_claim_winner_retains_role() -> None:
    decision = LocalCoordinationPolicy().decide_role(
        profile=make_profile(agent_id="peer-analyst-01"),
        state=make_state(
            agent_id="peer-analyst-01",
            status=PeerStatus.ACTIVE,
            claimed_role=PeerRole.ANALYST,
        ),
        mission=make_mission(),
        observed_claims={PeerRole.ANALYST: ("peer-analyst-02",)},
    )

    assert decision.action is PolicyAction.WAIT
    assert decision.role is PeerRole.ANALYST


def test_unavailable_peer_releases_existing_role() -> None:
    decision = LocalCoordinationPolicy().decide_role(
        profile=make_profile(available=False),
        state=make_state(status=PeerStatus.ACTIVE, claimed_role=PeerRole.ANALYST),
        mission=make_mission(),
        observed_claims={},
    )

    assert decision.action is PolicyAction.RELEASE_ROLE


def test_completed_peer_releases_existing_role() -> None:
    decision = LocalCoordinationPolicy().decide_role(
        profile=make_profile(),
        state=make_state(status=PeerStatus.COMPLETED, claimed_role=PeerRole.ANALYST),
        mission=make_mission(),
        observed_claims={},
    )

    assert decision.action is PolicyAction.RELEASE_ROLE


def test_identity_mismatch_escalates() -> None:
    decision = LocalCoordinationPolicy().decide_role(
        profile=make_profile(agent_id="peer-analyst-01"),
        state=make_state(agent_id="peer-other"),
        mission=make_mission(),
        observed_claims={},
    )

    assert decision.action is PolicyAction.ESCALATE


def test_mission_mismatch_escalates() -> None:
    decision = LocalCoordinationPolicy().decide_role(
        profile=make_profile(),
        state=make_state(mission_id="mission-other"),
        mission=make_mission(),
        observed_claims={},
    )

    assert decision.action is PolicyAction.ESCALATE


def test_matching_capability_work_request_is_accepted() -> None:
    request = WorkRequestPayload(
        work_id="work-001",
        requested_capability="analysis",
        summary="Assess the supplied evidence.",
    )
    decision = LocalCoordinationPolicy().decide_work_request(
        profile=make_profile(),
        state=make_state(status=PeerStatus.ACTIVE, claimed_role=PeerRole.ANALYST),
        mission=make_mission(),
        request=request,
    )

    assert decision.action is PolicyAction.ACCEPT_WORK
    assert decision.work_id == "work-001"


def test_nonmatching_capability_work_request_is_ignored() -> None:
    request = WorkRequestPayload(
        work_id="work-001",
        requested_capability="source_retrieval",
        summary="Retrieve evidence.",
    )
    decision = LocalCoordinationPolicy().decide_work_request(
        profile=make_profile(),
        state=make_state(status=PeerStatus.ACTIVE, claimed_role=PeerRole.ANALYST),
        mission=make_mission(),
        request=request,
    )

    assert decision.action is PolicyAction.IGNORE


def test_roleless_peer_waits_on_matching_work_request() -> None:
    request = WorkRequestPayload(
        work_id="work-001",
        requested_capability="analysis",
        summary="Assess the supplied evidence.",
    )
    decision = LocalCoordinationPolicy().decide_work_request(
        profile=make_profile(),
        state=make_state(status=PeerStatus.IDLE, claimed_role=None),
        mission=make_mission(),
        request=request,
    )

    assert decision.action is PolicyAction.WAIT


def test_completed_peer_ignores_new_work() -> None:
    request = WorkRequestPayload(
        work_id="work-001",
        requested_capability="analysis",
        summary="Assess the supplied evidence.",
    )
    decision = LocalCoordinationPolicy().decide_work_request(
        profile=make_profile(),
        state=make_state(status=PeerStatus.COMPLETED, claimed_role=PeerRole.ANALYST),
        mission=make_mission(),
        request=request,
    )

    assert decision.action is PolicyAction.IGNORE
