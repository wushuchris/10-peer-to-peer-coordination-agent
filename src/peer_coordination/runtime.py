"""Independent peer runtime for Agent 10 decentralized coordination."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

from .models import (
    BROADCAST_RECIPIENT,
    CapabilityAdvertisementPayload,
    EscalationPayload,
    MessageEnvelope,
    MessageType,
    Mission,
    MissionAnnouncementPayload,
    PeerRole,
    PeerState,
    PeerStatus,
    RoleClaimPayload,
    RoleReleasePayload,
    WorkRequestPayload,
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


class PeerAgent:
    """One independently stateful peer connected to the passive message bus.

    The peer drains only its own inbox, updates only its own local state, makes
    decisions through LocalCoordinationPolicy, and emits typed protocol messages.
    It does not mutate another peer or ask the runtime to assign work.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        mission: Mission,
        registry: PeerRegistry,
        bus: MessageBus,
        policy: LocalCoordinationPolicy | None = None,
        clock: Clock | None = None,
    ) -> None:
        profile = registry.get(agent_id)
        self.agent_id = profile.agent_id
        self.mission = mission
        self.registry = registry
        self.bus = bus
        self.policy = policy or LocalCoordinationPolicy()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.state = PeerState(agent_id=self.agent_id, mission_id=mission.mission_id)
        self._observed_claims: dict[PeerRole, set[str]] = {}
        self._message_sequence = 0

    @property
    def profile(self):
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
                # The transport has already verified that the advertised profile
                # matches the registry. Recording the sender is sufficient here.
                pass

            elif isinstance(payload, RoleClaimPayload):
                self._observed_claims.setdefault(payload.role, set()).add(envelope.sender_id)

            elif isinstance(payload, RoleReleasePayload):
                self._observed_claims.setdefault(payload.role, set()).discard(envelope.sender_id)

            elif isinstance(payload, WorkRequestPayload):
                decision = self.policy.decide_work_request(
                    profile=self.profile,
                    state=self.state,
                    mission=self.mission,
                    request=payload,
                )
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

    def broadcast_mission(self) -> str:
        """Broadcast the shared mission without assigning any role or task."""

        envelope = self._build_envelope(
            recipient_id=BROADCAST_RECIPIENT,
            message_type=MessageType.MISSION_ANNOUNCEMENT,
            payload=MissionAnnouncementPayload(mission=self.mission),
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

    def _build_envelope(
        self,
        *,
        recipient_id: str,
        message_type: MessageType,
        payload,
    ) -> MessageEnvelope:
        self._message_sequence += 1
        return MessageEnvelope(
            message_id=f"{self.agent_id}:msg:{self._message_sequence:04d}",
            mission_id=self.mission.mission_id,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            created_at=self.clock(),
            payload=payload,
        )


class PeerNetworkRuntime:
    """Mechanical turn scheduler for a collection of independent peers.

    The runtime only turns the clock by calling each peer in stable ID order.
    It does not inspect capabilities, assign roles, select workers, or mutate peer
    states directly.
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

    def get_peer(self, agent_id: str) -> PeerAgent:
        try:
            return self._peers[agent_id]
        except KeyError as exc:
            raise KeyError(f"unknown peer runtime: {agent_id}") from exc
