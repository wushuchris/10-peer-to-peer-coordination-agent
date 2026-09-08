"""Passive validated message transport for Agent 10."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import (
    BROADCAST_RECIPIENT,
    CapabilityAdvertisementPayload,
    MessageEnvelope,
    RoleClaimPayload,
    RoleReleasePayload,
)
from .registry import PeerRegistry, RegistryError


class TransportStatus(str, Enum):
    DELIVERED = "delivered"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"


class TransportError(ValueError):
    """Raised when a structurally valid message violates transport context."""


@dataclass(frozen=True)
class DeliveryReceipt:
    """Outcome of one transport attempt."""

    message_id: str
    status: TransportStatus
    recipients: tuple[str, ...]


@dataclass(frozen=True)
class TransportAuditEvent:
    """Append-only record of transport behavior."""

    message_id: str
    message_type: str
    correlation_id: str | None
    status: TransportStatus
    sender_id: str
    recipient_id: str
    recipients: tuple[str, ...]
    detail: str


class MessageBus:
    """Validate, deliver, deduplicate, and audit peer messages.

    The bus deliberately does not inspect work semantics to select a capable
    recipient. Broadcast messages go to all available peers except the sender;
    peers will later decide locally whether they should act.
    """

    def __init__(self, registry: PeerRegistry) -> None:
        self._registry = registry
        self._inboxes: dict[str, list[MessageEnvelope]] = {}
        self._delivered_message_ids: set[str] = set()
        self._audit_log: list[TransportAuditEvent] = []

    @property
    def audit_log(self) -> tuple[TransportAuditEvent, ...]:
        return tuple(self._audit_log)

    def peek(self, agent_id: str) -> tuple[MessageEnvelope, ...]:
        self._registry.get(agent_id)
        return tuple(self._inboxes.get(agent_id, ()))

    def drain(self, agent_id: str) -> tuple[MessageEnvelope, ...]:
        self._registry.get(agent_id)
        messages = tuple(self._inboxes.get(agent_id, ()))
        self._inboxes[agent_id] = []
        return messages

    def send(self, envelope: MessageEnvelope) -> DeliveryReceipt:
        if envelope.message_id in self._delivered_message_ids:
            event = TransportAuditEvent(
                message_id=envelope.message_id,
                message_type=envelope.message_type.value,
                correlation_id=envelope.correlation_id,
                status=TransportStatus.DUPLICATE,
                sender_id=envelope.sender_id,
                recipient_id=envelope.recipient_id,
                recipients=(),
                detail="message_id was already delivered; duplicate was not redelivered",
            )
            self._audit_log.append(event)
            return DeliveryReceipt(
                message_id=envelope.message_id,
                status=TransportStatus.DUPLICATE,
                recipients=(),
            )

        try:
            self._validate_context(envelope)
            recipients = self._resolve_recipients(envelope)
        except (RegistryError, TransportError) as exc:
            self._audit_log.append(
                TransportAuditEvent(
                    message_id=envelope.message_id,
                    message_type=envelope.message_type.value,
                    correlation_id=envelope.correlation_id,
                    status=TransportStatus.REJECTED,
                    sender_id=envelope.sender_id,
                    recipient_id=envelope.recipient_id,
                    recipients=(),
                    detail=str(exc),
                )
            )
            if isinstance(exc, TransportError):
                raise
            raise TransportError(str(exc)) from exc

        for agent_id in recipients:
            self._inboxes.setdefault(agent_id, []).append(envelope)

        self._delivered_message_ids.add(envelope.message_id)
        self._audit_log.append(
            TransportAuditEvent(
                message_id=envelope.message_id,
                message_type=envelope.message_type.value,
                correlation_id=envelope.correlation_id,
                status=TransportStatus.DELIVERED,
                sender_id=envelope.sender_id,
                recipient_id=envelope.recipient_id,
                recipients=recipients,
                detail=f"delivered to {len(recipients)} peer(s)",
            )
        )
        return DeliveryReceipt(
            message_id=envelope.message_id,
            status=TransportStatus.DELIVERED,
            recipients=recipients,
        )

    def _validate_context(self, envelope: MessageEnvelope) -> None:
        # Identity must always be registered. An unavailable peer is allowed to
        # release an existing claim so the network can converge cleanly, but it
        # cannot claim new work or send other active-work messages.
        self._registry.get(envelope.sender_id)
        if not isinstance(envelope.payload, RoleReleasePayload):
            self._registry.require_available(envelope.sender_id)

        if envelope.recipient_id != BROADCAST_RECIPIENT:
            if envelope.recipient_id == envelope.sender_id:
                raise TransportError("direct peer messages cannot target the sender itself")
            self._registry.require_available(envelope.recipient_id)

        if isinstance(envelope.payload, RoleClaimPayload):
            self._registry.require_role_eligibility(
                envelope.sender_id,
                envelope.payload.role,
            )

        if isinstance(envelope.payload, CapabilityAdvertisementPayload):
            self._registry.require_matching_advertisement(envelope.payload.profile)

    def _resolve_recipients(self, envelope: MessageEnvelope) -> tuple[str, ...]:
        if envelope.recipient_id != BROADCAST_RECIPIENT:
            return (envelope.recipient_id,)

        return tuple(
            profile.agent_id
            for profile in self._registry.discover(
                available_only=True,
                exclude_agent_id=envelope.sender_id,
            )
        )
