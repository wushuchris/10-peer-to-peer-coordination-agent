"""Passive failure detection and fail-closed containment for Agent 10.

This module observes an existing peer network without assigning roles, selecting
workers, or repairing the mission. It turns repeated lack of progress into typed,
auditable coordination issues and stops further execution once a blocking failure
is established.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import PeerRole, PeerStatus, StrictModel
from .registry import PeerRegistry
from .runtime import PeerAgent, PeerNetworkRuntime, TurnResult, WorkExecutionResult, WorkExecutionStatus


class FailureIssueCode(str, Enum):
    UNAVAILABLE_REQUIRED_ROLE = "unavailable_required_role"
    UNCLAIMED_REQUIRED_ROLE = "unclaimed_required_role"
    DUPLICATE_ROLE_CLAIM = "duplicate_role_claim"
    MISSING_DEPENDENCY = "missing_dependency"
    OPEN_CHALLENGE_TIMEOUT = "open_challenge_timeout"
    NO_PROGRESS = "no_progress"


class FailureSeverity(str, Enum):
    WARNING = "warning"
    BLOCKING = "blocking"


class CoordinationIssue(StrictModel):
    """One deterministic health finding produced by passive observation."""

    code: FailureIssueCode
    severity: FailureSeverity
    detail: str
    agent_id: str | None = None
    role: PeerRole | None = None
    work_id: str | None = None
    challenge_id: str | None = None


class CoordinationHealth(StrictModel):
    """Snapshot of network health after one mechanical coordination cycle."""

    cycle: int
    stalled_rounds: int
    blocked: bool
    issues: tuple[CoordinationIssue, ...] = ()


@dataclass(frozen=True)
class FailurePolicy:
    """Bounded deterministic thresholds for declaring coordination failures."""

    role_grace_rounds: int = 3
    role_conflict_rounds: int = 2
    dependency_wait_rounds: int = 3
    challenge_response_rounds: int = 3
    no_progress_rounds: int = 3

    def __post_init__(self) -> None:
        for name, value in (
            ("role_grace_rounds", self.role_grace_rounds),
            ("role_conflict_rounds", self.role_conflict_rounds),
            ("dependency_wait_rounds", self.dependency_wait_rounds),
            ("challenge_response_rounds", self.challenge_response_rounds),
            ("no_progress_rounds", self.no_progress_rounds),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")


@dataclass(frozen=True)
class ContainedCycleResult:
    """One process/execution cycle plus the passive health result."""

    turns: tuple[TurnResult, ...]
    executions: tuple[WorkExecutionResult, ...]
    health: CoordinationHealth
    contained: bool = False


class FailureContainmentRuntime:
    """Mechanical runtime wrapper that detects failure and then fails closed.

    The wrapper owns no semantic routing logic. It never calls registry discovery to
    choose a worker, never changes a peer's role, and never mutates peer state. It
    only schedules the already-existing PeerNetworkRuntime, observes deterministic
    state, and prevents additional cycles after a blocking condition is established.
    """

    def __init__(
        self,
        peers: tuple[PeerAgent, ...] | list[PeerAgent],
        *,
        registry: PeerRegistry,
        policy: FailurePolicy | None = None,
    ) -> None:
        peer_map: dict[str, PeerAgent] = {}
        for peer in peers:
            if peer.agent_id in peer_map:
                raise ValueError(f"duplicate peer runtime: {peer.agent_id}")
            peer_map[peer.agent_id] = peer
        if not peer_map:
            raise ValueError("failure containment requires at least one peer")

        missions = {peer.mission.mission_id for peer in peer_map.values()}
        if len(missions) != 1:
            raise ValueError("all contained peers must share one mission_id")

        self._peers = peer_map
        self.registry = registry
        self.policy = policy or FailurePolicy()
        self.runtime = PeerNetworkRuntime(tuple(peer_map.values()))
        self.mission = next(iter(peer_map.values())).mission

        self._cycle = 0
        self._blocked = False
        self._last_health = CoordinationHealth(cycle=0, stalled_rounds=0, blocked=False)
        self._last_fingerprint: tuple | None = None
        self._stalled_rounds = 0
        self._fulfilled_roles: dict[str, set[PeerRole]] = {
            agent_id: set() for agent_id in peer_map
        }
        self._role_conflict_rounds: dict[PeerRole, int] = {}
        self._dependency_wait_rounds: dict[tuple[str, str], int] = {}
        self._open_challenge_rounds: dict[tuple[str, str], int] = {}

    @property
    def blocked(self) -> bool:
        return self._blocked

    @property
    def health(self) -> CoordinationHealth:
        return self._last_health

    def get_peer(self, agent_id: str) -> PeerAgent:
        try:
            return self._peers[agent_id]
        except KeyError as exc:
            raise KeyError(f"unknown contained peer: {agent_id}") from exc

    def run_cycle(self) -> ContainedCycleResult:
        """Advance one mechanical cycle unless a prior issue already blocked it."""

        if self._blocked:
            return ContainedCycleResult(
                turns=(),
                executions=(),
                health=self._last_health,
                contained=True,
            )

        turns = self.runtime.run_round()
        self._record_fulfilled_roles(turns)
        executions = self.runtime.run_execution_round()
        self._cycle += 1

        health = self._inspect(executions)
        self._last_health = health
        if health.blocked:
            self._blocked = True

        return ContainedCycleResult(
            turns=turns,
            executions=executions,
            health=health,
            contained=False,
        )

    def _record_fulfilled_roles(self, turns: tuple[TurnResult, ...]) -> None:
        """Preserve completed-role history after a peer broadcasts ROLE_RELEASE."""

        for turn in turns:
            peer = self._peers[turn.agent_id]
            if peer.state.status is not PeerStatus.COMPLETED:
                continue
            for decision in turn.decisions:
                if decision.action.value == "release_role" and decision.role is not None:
                    self._fulfilled_roles[peer.agent_id].add(decision.role)

    def _inspect(
        self,
        executions: tuple[WorkExecutionResult, ...],
    ) -> CoordinationHealth:
        issues: list[CoordinationIssue] = []

        active_claimants: dict[PeerRole, list[str]] = {}
        fulfilled: set[PeerRole] = set()
        for agent_id, peer in self._peers.items():
            if peer.state.claimed_role is not None:
                active_claimants.setdefault(peer.state.claimed_role, []).append(agent_id)
            fulfilled.update(self._fulfilled_roles[agent_id])

        covered_roles = set(active_claimants) | fulfilled

        for role in self.mission.required_roles:
            if role in covered_roles:
                continue
            eligible_available = self.registry.discover(role=role, available_only=True)
            if not eligible_available:
                issues.append(
                    CoordinationIssue(
                        code=FailureIssueCode.UNAVAILABLE_REQUIRED_ROLE,
                        severity=FailureSeverity.BLOCKING,
                        role=role,
                        detail=(
                            f"required role {role.value} is uncovered and no registered "
                            "available peer is eligible"
                        ),
                    )
                )
            elif self._cycle >= self.policy.role_grace_rounds:
                issues.append(
                    CoordinationIssue(
                        code=FailureIssueCode.UNCLAIMED_REQUIRED_ROLE,
                        severity=FailureSeverity.BLOCKING,
                        role=role,
                        detail=(
                            f"required role {role.value} remained unclaimed for "
                            f"{self._cycle} coordination cycles"
                        ),
                    )
                )

        current_conflicts = {
            role: tuple(sorted(agent_ids))
            for role, agent_ids in active_claimants.items()
            if len(agent_ids) > 1
        }
        for role in set(self._role_conflict_rounds) | set(current_conflicts):
            if role in current_conflicts:
                rounds = self._role_conflict_rounds.get(role, 0) + 1
                self._role_conflict_rounds[role] = rounds
                severity = (
                    FailureSeverity.BLOCKING
                    if rounds >= self.policy.role_conflict_rounds
                    else FailureSeverity.WARNING
                )
                issues.append(
                    CoordinationIssue(
                        code=FailureIssueCode.DUPLICATE_ROLE_CLAIM,
                        severity=severity,
                        role=role,
                        detail=(
                            f"role {role.value} has persistent claimants "
                            f"{', '.join(current_conflicts[role])} for {rounds} cycle(s)"
                        ),
                    )
                )
            else:
                self._role_conflict_rounds.pop(role, None)

        waiting_now: set[tuple[str, str]] = set()
        for execution in executions:
            key = (execution.agent_id, execution.work_id)
            if execution.status is WorkExecutionStatus.WAITING:
                waiting_now.add(key)
                rounds = self._dependency_wait_rounds.get(key, 0) + 1
                self._dependency_wait_rounds[key] = rounds
                if rounds >= self.policy.dependency_wait_rounds:
                    issues.append(
                        CoordinationIssue(
                            code=FailureIssueCode.MISSING_DEPENDENCY,
                            severity=FailureSeverity.BLOCKING,
                            agent_id=execution.agent_id,
                            work_id=execution.work_id,
                            detail=(
                                f"work {execution.work_id} remained unable to execute for "
                                f"{rounds} cycle(s): {execution.detail}"
                            ),
                        )
                    )
        for key in tuple(self._dependency_wait_rounds):
            if key not in waiting_now:
                self._dependency_wait_rounds.pop(key, None)

        open_now: set[tuple[str, str]] = set()
        for agent_id, peer in self._peers.items():
            ledger = getattr(peer, "challenge_ledger", None)
            if ledger is None:
                continue
            for challenge_id in ledger.open_challenge_ids:
                key = (agent_id, challenge_id)
                open_now.add(key)
                rounds = self._open_challenge_rounds.get(key, 0) + 1
                self._open_challenge_rounds[key] = rounds
                if rounds >= self.policy.challenge_response_rounds:
                    issues.append(
                        CoordinationIssue(
                            code=FailureIssueCode.OPEN_CHALLENGE_TIMEOUT,
                            severity=FailureSeverity.BLOCKING,
                            agent_id=agent_id,
                            challenge_id=challenge_id,
                            detail=(
                                f"challenge {challenge_id} remained unresolved for "
                                f"{rounds} coordination cycle(s)"
                            ),
                        )
                    )
        for key in tuple(self._open_challenge_rounds):
            if key not in open_now:
                self._open_challenge_rounds.pop(key, None)

        fingerprint = self._progress_fingerprint()
        if self._last_fingerprint is None or fingerprint != self._last_fingerprint:
            self._stalled_rounds = 0
        else:
            self._stalled_rounds += 1
        self._last_fingerprint = fingerprint

        if (
            self._stalled_rounds >= self.policy.no_progress_rounds
            and not self._mission_is_quiescent_success()
        ):
            issues.append(
                CoordinationIssue(
                    code=FailureIssueCode.NO_PROGRESS,
                    severity=FailureSeverity.BLOCKING,
                    detail=(
                        f"network state made no observable progress for "
                        f"{self._stalled_rounds} consecutive cycle(s)"
                    ),
                )
            )

        issues.sort(
            key=lambda issue: (
                issue.code.value,
                issue.role.value if issue.role is not None else "",
                issue.agent_id or "",
                issue.work_id or "",
                issue.challenge_id or "",
            )
        )
        blocked = any(issue.severity is FailureSeverity.BLOCKING for issue in issues)
        return CoordinationHealth(
            cycle=self._cycle,
            stalled_rounds=self._stalled_rounds,
            blocked=blocked,
            issues=tuple(issues),
        )

    def _progress_fingerprint(self) -> tuple:
        peers: list[tuple] = []
        for agent_id in sorted(self._peers):
            peer = self._peers[agent_id]
            ledger = getattr(peer, "challenge_ledger", None)
            challenge_state = ()
            if ledger is not None:
                challenge_state = (
                    tuple(ledger.open_challenge_ids),
                    tuple(ledger.responded_challenge_ids),
                    tuple(ledger.resolved_challenge_ids),
                    tuple(ledger.disputed_challenge_ids),
                )
            peers.append(
                (
                    agent_id,
                    peer.state.status.value,
                    peer.state.claimed_role.value if peer.state.claimed_role else None,
                    tuple(peer.state.received_message_ids),
                    tuple(peer.state.accepted_work_ids),
                    tuple(peer.state.completed_work_ids),
                    tuple(sorted(peer.work_products)),
                    tuple(sorted(role.value for role in self._fulfilled_roles[agent_id])),
                    challenge_state,
                )
            )
        return tuple(peers)

    def _mission_is_quiescent_success(self) -> bool:
        covered: set[PeerRole] = set()
        pending_work = False
        open_challenge = False

        for agent_id, peer in self._peers.items():
            if peer.state.claimed_role is not None:
                covered.add(peer.state.claimed_role)
            covered.update(self._fulfilled_roles[agent_id])
            if set(peer.state.accepted_work_ids) - set(peer.state.completed_work_ids):
                pending_work = True
            ledger = getattr(peer, "challenge_ledger", None)
            if ledger is not None and ledger.open_challenge_ids:
                open_challenge = True

        return (
            set(self.mission.required_roles).issubset(covered)
            and not pending_work
            and not open_challenge
        )
