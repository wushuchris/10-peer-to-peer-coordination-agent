"""Independent peer runtime for Agent 10 decentralized coordination."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from .models import (
    BROADCAST_RECIPIENT,
    CapabilityAdvertisementPayload,
    EscalationPayload,
    MessageEnvelope,
    MessageType,
    Mission,
    MissionAnnouncementPayload,
    PeerProfile,
    PeerRole,
    PeerState,
    PeerStatus,
    RoleClaimPayload,
    RoleReleasePayload,
    WorkRequestPayload,
    WorkResultPayload,
)
from .policy import LocalCoordinationPolicy, PolicyAction, PolicyDecision
from .registry import PeerRegistry
from .transport import MessageBus

Clock = Callable[[], datetime]


@dataclass(frozen=True)
class TurnResult:
    """Auditable summary of one peer's processing turn."""

    agent_id: str
    processed_message_ids: tuple[str, ...]
    emitted_message_ids: tuple[str, ...]
    decisions: tuple[PolicyDecision, ...]


@dataclass(frozen=True)
class WorkExecutionContext:
    """Read-only inputs supplied to one local work handler invocation."""

    mission: Mission
    profile: PeerProfile
    state: PeerState
    request: WorkRequestPayload
    input_products: tuple[WorkResultPayload, ...]


@dataclass(frozen=True)
class WorkExecutionOutcome:
    """Validated handler output that the peer runtime may publish."""

    result: WorkResultPayload
    next_request: WorkRequestPayload | None = None
    mark_peer_completed: bool = True


WorkHandler = Callable[[WorkExecutionContext], WorkExecutionOutcome]


class WorkExecutionStatus(str, Enum):
    EXECUTED = "executed"
    WAITING = "waiting"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class WorkExecutionResult:
    """Auditable result of one peer attempting locally accepted work."""

    agent_id: str
    work_id: str
    status: WorkExecutionStatus
    emitted_message_ids: tuple[str, ...]
    detail: str


class PeerAgent:
    """One independently stateful peer connected to the passive message bus.

    The peer drains only its own inbox, updates only its own local state, makes
    decisions through LocalCoordinationPolicy, and emits typed protocol messages.
    Work executes only after policy acceptance and only through a local handler
    explicitly registered for the peer's declared capability.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        mission: Mission,
        registry: PeerRegistry,
        bus: MessageBus,
        policy: LocalCoordinationPolicy | None = None,
        handlers: Mapping[str, WorkHandler] | None = None,
        clock: Clock | None = None,
    ) -> None:
        profile = registry.get(agent_id)
        self.agent_id = profile.agent_id
        self.mission = mission
        self.registry = registry
        self.bus = bus
        self.policy = policy or LocalCoordinationPolicy()
        self.handlers = dict(handlers or {})
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.state = PeerState(agent_id=self.agent_id, mission_id=mission.mission_id)
        self._observed_claims: dict[PeerRole, set[str]] = {}
        self._accepted_requests: dict[str, MessageEnvelope] = {}
        self._work_products: dict[str, WorkResultPayload] = {}
        self._message_sequence = 0

    @property
    def profile(self) -> PeerProfile:
        """Return the peer's current registered profile, including availability."""

        return self.registry.get(self.agent_id)

    @property
    def observed_claims(self) -> dict[PeerRole, tuple[str, ...]]:
        """Return a copy of the peer's local role-claim observations."""

        return {
            role: tuple(sorted(agent_ids))
            for role, agent_ids in self._observed_claims.items()
            if agent_ids
        }

    @property
    def work_products(self) -> dict[str, WorkResultPayload]:
        """Return a copy of work products this peer has produced or observed."""

        return dict(self._work_products)

    def process_turn(self) -> TurnResult:
        """Process this peer's inbox and make one bounded local role decision."""

        inbox = self.bus.drain(self.agent_id)
        processed: list[str] = []
        emitted: list[str] = []
        decisions: list[PolicyDecision] = []

        for envelope in inbox:
            processed.append(envelope.message_id)
            self._record_received(envelope)

            if envelope.mission_id != self.mission.mission_id:
                decision = PolicyDecision(
                    action=PolicyAction.ESCALATE,
                    reason="received message for a different mission",
                )
                decisions.append(decision)
                emitted.extend(self._apply_decision(decision))
                continue

            payload = envelope.payload
            if isinstance(payload, MissionAnnouncementPayload):
                if payload.mission != self.mission:
                    decision = PolicyDecision(
                        action=PolicyAction.ESCALATE,
                        reason="mission announcement conflicts with local mission contract",
                    )
                    decisions.append(decision)
                    emitted.extend(self._apply_decision(decision))
                    continue

            elif isinstance(payload, CapabilityAdvertisementPayload):
                # Transport already verified the profile against the registry.
                pass

            elif isinstance(payload, RoleClaimPayload):
                self._observed_claims.setdefault(payload.role, set()).add(envelope.sender_id)

            elif isinstance(payload, RoleReleasePayload):
                self._observed_claims.setdefault(payload.role, set()).discard(envelope.sender_id)

            elif isinstance(payload, WorkResultPayload):
                conflict = self._record_work_product(payload)
                if conflict is not None:
                    decisions.append(conflict)
                    emitted.extend(self._apply_decision(conflict))

            elif isinstance(payload, WorkRequestPayload):
                decision = self.policy.decide_work_request(
                    profile=self.profile,
                    state=self.state,
                    mission=self.mission,
                    request=payload,
                )
                if decision.action is PolicyAction.ACCEPT_WORK:
                    existing = self._accepted_requests.get(payload.work_id)
                    if existing is not None and existing.payload != payload:
                        decision = PolicyDecision(
                            action=PolicyAction.ESCALATE,
                            work_id=payload.work_id,
                            reason="conflicting work requests reuse the same work_id",
                        )
                    else:
                        self._accepted_requests[payload.work_id] = envelope
                decisions.append(decision)
                emitted.extend(self._apply_decision(decision))

        if self.state.status is not PeerStatus.ESCALATED:
            role_decision = self.policy.decide_role(
                profile=self.profile,
                state=self.state,
                mission=self.mission,
                observed_claims=self.observed_claims,
            )
            decisions.append(role_decision)
            emitted.extend(self._apply_decision(role_decision))

        return TurnResult(
            agent_id=self.agent_id,
            processed_message_ids=tuple(processed),
            emitted_message_ids=tuple(emitted),
            decisions=tuple(decisions),
        )

    def execute_accepted_work(self) -> tuple[WorkExecutionResult, ...]:
        """Give this peer a chance to execute only work it locally accepted.

        Missing dependencies cause a local wait. Missing handlers, conflicting
        outputs, or handler failures fail closed into protocol escalation.
        """

        results: list[WorkExecutionResult] = []
        for work_id in self.state.accepted_work_ids:
            if work_id in self.state.completed_work_ids:
                continue

            request_envelope = self._accepted_requests.get(work_id)
            if request_envelope is None or not isinstance(
                request_envelope.payload, WorkRequestPayload
            ):
                results.append(
                    self._escalate_execution(
                        work_id,
                        "accepted work has no matching local request envelope",
                    )
                )
                continue

            request = request_envelope.payload
            missing = tuple(
                product_id
                for product_id in request.input_work_product_ids
                if product_id not in self._work_products
            )
            if missing:
                results.append(
                    WorkExecutionResult(
                        agent_id=self.agent_id,
                        work_id=work_id,
                        status=WorkExecutionStatus.WAITING,
                        emitted_message_ids=(),
                        detail=f"waiting for input work products: {', '.join(missing)}",
                    )
                )
                continue

            handler = self.handlers.get(request.requested_capability)
            if handler is None:
                results.append(
                    self._escalate_execution(
                        work_id,
                        (
                            "no local handler registered for accepted capability "
                            f"{request.requested_capability}"
                        ),
                    )
                )
                continue

            context = WorkExecutionContext(
                mission=self.mission.model_copy(deep=True),
                profile=self.profile.model_copy(deep=True),
                state=self.state.model_copy(deep=True),
                request=request.model_copy(deep=True),
                input_products=tuple(
                    self._work_products[product_id].model_copy(deep=True)
                    for product_id in request.input_work_product_ids
                ),
            )

            try:
                outcome = handler(context)
                self._validate_work_outcome(request, outcome)
            except Exception as exc:  # local failure containment boundary
                results.append(
                    self._escalate_execution(
                        work_id,
                        f"local work handler failed closed: {type(exc).__name__}: {exc}",
                    )
                )
                continue

            conflict = self._record_work_product(outcome.result)
            if conflict is not None:
                results.append(self._escalate_execution(work_id, conflict.reason))
                continue

            result_envelope = self._build_envelope(
                recipient_id=BROADCAST_RECIPIENT,
                message_type=MessageType.WORK_RESULT,
                payload=outcome.result,
                correlation_id=request_envelope.message_id,
            )
            self.bus.send(result_envelope)
            emitted = [result_envelope.message_id]

            if outcome.next_request is not None:
                next_envelope = self._build_envelope(
                    recipient_id=BROADCAST_RECIPIENT,
                    message_type=MessageType.WORK_REQUEST,
                    payload=outcome.next_request,
                    correlation_id=result_envelope.message_id,
                )
                self.bus.send(next_envelope)
                emitted.append(next_envelope.message_id)

            completed = list(self.state.completed_work_ids)
            completed.append(work_id)
            updates: dict[str, object] = {"completed_work_ids": tuple(completed)}
            if outcome.mark_peer_completed:
                updates["status"] = PeerStatus.COMPLETED
            self.state = self.state.model_copy(update=updates)

            results.append(
                WorkExecutionResult(
                    agent_id=self.agent_id,
                    work_id=work_id,
                    status=WorkExecutionStatus.EXECUTED,
                    emitted_message_ids=tuple(emitted),
                    detail=f"published work product {outcome.result.work_product_id}",
                )
            )

        return tuple(results)

    def broadcast_mission(self) -> str:
        """Broadcast the shared mission without assigning any role or task."""

        envelope = self._build_envelope(
            recipient_id=BROADCAST_RECIPIENT,
            message_type=MessageType.MISSION_ANNOUNCEMENT,
            payload=MissionAnnouncementPayload(mission=self.mission),
        )
        self.bus.send(envelope)
        return envelope.message_id

    def broadcast_work_request(self, request: WorkRequestPayload) -> str:
        """Broadcast an open request; recipients decide locally whether to accept."""

        envelope = self._build_envelope(
            recipient_id=BROADCAST_RECIPIENT,
            message_type=MessageType.WORK_REQUEST,
            payload=request,
        )
        self.bus.send(envelope)
        return envelope.message_id

    def _record_received(self, envelope: MessageEnvelope) -> None:
        known_peer_ids = list(self.state.known_peer_ids)
        if envelope.sender_id != self.agent_id and envelope.sender_id not in known_peer_ids:
            known_peer_ids.append(envelope.sender_id)

        received_message_ids = list(self.state.received_message_ids)
        if envelope.message_id not in received_message_ids:
            received_message_ids.append(envelope.message_id)

        self.state = self.state.model_copy(
            update={
                "known_peer_ids": tuple(known_peer_ids),
                "received_message_ids": tuple(received_message_ids),
            }
        )

    def _record_work_product(self, payload: WorkResultPayload) -> PolicyDecision | None:
        existing = self._work_products.get(payload.work_product_id)
        if existing is not None and existing != payload:
            return PolicyDecision(
                action=PolicyAction.ESCALATE,
                work_id=payload.work_id,
                reason=(
                    "conflicting work results reuse work_product_id "
                    f"{payload.work_product_id}"
                ),
            )
        self._work_products[payload.work_product_id] = payload
        return None

    def _apply_decision(self, decision: PolicyDecision) -> tuple[str, ...]:
        if decision.action is PolicyAction.CLAIM_ROLE:
            if decision.role is None:
                raise ValueError("CLAIM_ROLE decision requires a role")
            envelope = self._build_envelope(
                recipient_id=BROADCAST_RECIPIENT,
                message_type=MessageType.ROLE_CLAIM,
                payload=RoleClaimPayload(role=decision.role, reason=decision.reason),
            )
            self.bus.send(envelope)
            self._observed_claims.setdefault(decision.role, set()).add(self.agent_id)
            self.state = self.state.model_copy(
                update={"claimed_role": decision.role, "status": PeerStatus.ACTIVE}
            )
            return (envelope.message_id,)

        if decision.action is PolicyAction.RELEASE_ROLE:
            if decision.role is None:
                raise ValueError("RELEASE_ROLE decision requires a role")
            envelope = self._build_envelope(
                recipient_id=BROADCAST_RECIPIENT,
                message_type=MessageType.ROLE_RELEASE,
                payload=RoleReleasePayload(role=decision.role, reason=decision.reason),
            )
            self.bus.send(envelope)
            self._observed_claims.setdefault(decision.role, set()).discard(self.agent_id)
            self.state = self.state.model_copy(
                update={"claimed_role": None, "status": PeerStatus.WAITING}
            )
            return (envelope.message_id,)

        if decision.action is PolicyAction.ACCEPT_WORK:
            if decision.work_id is None:
                raise ValueError("ACCEPT_WORK decision requires a work_id")
            accepted = list(self.state.accepted_work_ids)
            if decision.work_id not in accepted:
                accepted.append(decision.work_id)
            self.state = self.state.model_copy(
                update={"accepted_work_ids": tuple(accepted), "status": PeerStatus.ACTIVE}
            )
            return ()

        if decision.action is PolicyAction.ESCALATE:
            if self.state.status is PeerStatus.ESCALATED:
                return ()
            self.state = self.state.model_copy(update={"status": PeerStatus.ESCALATED})
            envelope = self._build_envelope(
                recipient_id=BROADCAST_RECIPIENT,
                message_type=MessageType.ESCALATION,
                payload=EscalationPayload(reason=decision.reason),
            )
            self.bus.send(envelope)
            return (envelope.message_id,)

        if decision.action is PolicyAction.WAIT:
            if self.state.claimed_role is None and self.state.status not in {
                PeerStatus.COMPLETED,
                PeerStatus.ESCALATED,
            }:
                self.state = self.state.model_copy(update={"status": PeerStatus.WAITING})
            return ()

        # IGNORE intentionally leaves local state unchanged.
        return ()

    def _escalate_execution(self, work_id: str, reason: str) -> WorkExecutionResult:
        decision = PolicyDecision(
            action=PolicyAction.ESCALATE,
            work_id=work_id,
            reason=reason,
        )
        emitted = self._apply_decision(decision)
        return WorkExecutionResult(
            agent_id=self.agent_id,
            work_id=work_id,
            status=WorkExecutionStatus.ESCALATED,
            emitted_message_ids=emitted,
            detail=reason,
        )

    @staticmethod
    def _validate_work_outcome(
        request: WorkRequestPayload,
        outcome: WorkExecutionOutcome,
    ) -> None:
        if outcome.result.work_id != request.work_id:
            raise ValueError("work result must reference the accepted work_id")
        if outcome.next_request is not None:
            if outcome.next_request.work_id == request.work_id:
                raise ValueError("next work request must use a new work_id")
            if outcome.result.work_product_id not in outcome.next_request.input_work_product_ids:
                raise ValueError(
                    "next work request must depend on the work product just produced"
                )

    def _build_envelope(
        self,
        *,
        recipient_id: str,
        message_type: MessageType,
        payload,
        correlation_id: str | None = None,
    ) -> MessageEnvelope:
        self._message_sequence += 1
        return MessageEnvelope(
            message_id=f"{self.agent_id}:msg:{self._message_sequence:04d}",
            mission_id=self.mission.mission_id,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            correlation_id=correlation_id,
            created_at=self.clock(),
            payload=payload,
        )


class PeerNetworkRuntime:
    """Mechanical scheduler for a collection of independent peers.

    The runtime only gives each peer stable-order opportunities to process its
    inbox or execute locally accepted work. It does not inspect capabilities,
    assign roles, select workers, or mutate peer states directly.
    """

    def __init__(self, peers: Iterable[PeerAgent]) -> None:
        peer_map: dict[str, PeerAgent] = {}
        for peer in peers:
            if peer.agent_id in peer_map:
                raise ValueError(f"duplicate peer runtime: {peer.agent_id}")
            peer_map[peer.agent_id] = peer
        self._peers = peer_map

    def run_round(self) -> tuple[TurnResult, ...]:
        return tuple(self._peers[agent_id].process_turn() for agent_id in sorted(self._peers))

    def run_execution_round(self) -> tuple[WorkExecutionResult, ...]:
        return tuple(
            execution
            for agent_id in sorted(self._peers)
            for execution in self._peers[agent_id].execute_accepted_work()
        )

    def get_peer(self, agent_id: str) -> PeerAgent:
        try:
            return self._peers[agent_id]
        except KeyError as exc:
            raise KeyError(f"unknown peer runtime: {agent_id}") from exc
