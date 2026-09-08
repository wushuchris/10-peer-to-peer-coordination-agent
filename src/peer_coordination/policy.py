"""Deterministic local coordination rules for Agent 10 peers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .models import Mission, PeerProfile, PeerRole, PeerState, PeerStatus, WorkRequestPayload


class PolicyAction(str, Enum):
    """Bounded local actions a peer may recommend."""

    CLAIM_ROLE = "claim_role"
    RELEASE_ROLE = "release_role"
    ACCEPT_WORK = "accept_work"
    IGNORE = "ignore"
    WAIT = "wait"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class PolicyDecision:
    """Auditable output of one local policy evaluation."""

    action: PolicyAction
    reason: str
    role: PeerRole | None = None
    work_id: str | None = None


class LocalCoordinationPolicy:
    """Pure deterministic decisions made from one peer's local view.

    The policy never assigns work to another peer and never mutates shared state.
    It only decides what the current peer should do based on its own registered
    capabilities, local state, mission requirements, and observed role claims.
    """

    def decide_role(
        self,
        *,
        profile: PeerProfile,
        state: PeerState,
        mission: Mission,
        observed_claims: Mapping[PeerRole, tuple[str, ...]],
    ) -> PolicyDecision:
        """Decide whether this peer should claim, retain, release, or wait."""

        identity_error = self._validate_local_identity(profile, state, mission)
        if identity_error is not None:
            return identity_error

        if state.status is PeerStatus.ESCALATED:
            return PolicyDecision(
                action=PolicyAction.WAIT,
                reason="peer is already escalated and should not make a new role decision",
            )

        if state.status is PeerStatus.COMPLETED:
            if state.claimed_role is not None:
                return PolicyDecision(
                    action=PolicyAction.RELEASE_ROLE,
                    role=state.claimed_role,
                    reason="completed peer should release its active role claim",
                )
            return PolicyDecision(
                action=PolicyAction.IGNORE,
                reason="completed peer has no further role action",
            )

        if not profile.available:
            if state.claimed_role is not None:
                return PolicyDecision(
                    action=PolicyAction.RELEASE_ROLE,
                    role=state.claimed_role,
                    reason="unavailable peer must release its current role claim",
                )
            return PolicyDecision(
                action=PolicyAction.WAIT,
                reason="unavailable peer cannot claim mission work",
            )

        if state.claimed_role is not None:
            role = state.claimed_role
            if role not in mission.required_roles or role not in profile.eligible_roles:
                return PolicyDecision(
                    action=PolicyAction.RELEASE_ROLE,
                    role=role,
                    reason="current role is no longer required or locally eligible",
                )

            claimants = set(observed_claims.get(role, ()))
            claimants.add(profile.agent_id)
            winner = min(claimants)
            if winner != profile.agent_id:
                return PolicyDecision(
                    action=PolicyAction.RELEASE_ROLE,
                    role=role,
                    reason=(
                        f"duplicate role claim observed; deterministic tie-breaker "
                        f"keeps {winner}"
                    ),
                )

            return PolicyDecision(
                action=PolicyAction.WAIT,
                role=role,
                reason="retain current role; no conflicting higher-priority claimant observed",
            )

        for role in mission.required_roles:
            if role not in profile.eligible_roles:
                continue

            claimants = tuple(
                agent_id
                for agent_id in observed_claims.get(role, ())
                if agent_id != profile.agent_id
            )
            if claimants:
                continue

            return PolicyDecision(
                action=PolicyAction.CLAIM_ROLE,
                role=role,
                reason=f"role {role.value} is required, locally eligible, and observed unclaimed",
            )

        eligible_required_roles = tuple(
            role for role in mission.required_roles if role in profile.eligible_roles
        )
        if not eligible_required_roles:
            return PolicyDecision(
                action=PolicyAction.IGNORE,
                reason="peer has no eligible role required by this mission",
            )

        return PolicyDecision(
            action=PolicyAction.WAIT,
            reason="all locally eligible mission roles are currently claimed by peers",
        )

    def decide_work_request(
        self,
        *,
        profile: PeerProfile,
        state: PeerState,
        mission: Mission,
        request: WorkRequestPayload,
    ) -> PolicyDecision:
        """Decide whether this peer should accept an already-delivered work request."""

        identity_error = self._validate_local_identity(profile, state, mission)
        if identity_error is not None:
            return identity_error

        if not profile.available:
            return PolicyDecision(
                action=PolicyAction.IGNORE,
                work_id=request.work_id,
                reason="unavailable peer does not accept new work",
            )

        if state.status in {PeerStatus.COMPLETED, PeerStatus.ESCALATED}:
            return PolicyDecision(
                action=PolicyAction.IGNORE,
                work_id=request.work_id,
                reason=f"peer status {state.status.value} does not accept new work",
            )

        if state.claimed_role is None:
            return PolicyDecision(
                action=PolicyAction.WAIT,
                work_id=request.work_id,
                reason="peer has not yet established a mission role",
            )

        if request.requested_capability not in profile.capabilities:
            return PolicyDecision(
                action=PolicyAction.IGNORE,
                work_id=request.work_id,
                reason=(
                    f"requested capability {request.requested_capability} is not "
                    "declared by this peer"
                ),
            )

        return PolicyDecision(
            action=PolicyAction.ACCEPT_WORK,
            work_id=request.work_id,
            reason="request matches a declared capability of this active mission peer",
        )

    @staticmethod
    def _validate_local_identity(
        profile: PeerProfile,
        state: PeerState,
        mission: Mission,
    ) -> PolicyDecision | None:
        if profile.agent_id != state.agent_id:
            return PolicyDecision(
                action=PolicyAction.ESCALATE,
                reason="local profile and state refer to different peer identities",
            )
        if state.mission_id != mission.mission_id:
            return PolicyDecision(
                action=PolicyAction.ESCALATE,
                reason="local state mission does not match the mission being evaluated",
            )
        return None
