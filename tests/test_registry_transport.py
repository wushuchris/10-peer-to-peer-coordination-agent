"""Tests for deterministic peer discovery and passive message transport."""

from datetime import datetime, timezone

import pytest

from peer_coordination.models import (
    BROADCAST_RECIPIENT,
    CapabilityAdvertisementPayload,
    MessageEnvelope,
    MessageType,
    PeerProfile,
    PeerRole,
    RoleClaimPayload,
    StatusPayload,
    PeerStatus,
    WorkRequestPayload,
)
from peer_coordination.registry import (
    DuplicatePeerError,
    IneligibleRoleError,
    PeerRegistry,
    UnknownPeerError,
)
from peer_coordination.transport import MessageBus, TransportError, TransportStatus


def profile(
    agent_id: str,
    role: PeerRole,
    capability: str,
    *,
    available: bool = True,
) -> PeerProfile:
    return PeerProfile(
        agent_id=agent_id,
        display_name=agent_id,
        capabilities=(capability,),
        eligible_roles=(role,),
        available=available,
    )


def make_registry() -> PeerRegistry:
    return PeerRegistry(
        (
            profile("peer-source", PeerRole.SOURCE_FINDER, "source_retrieval"),
            profile("peer-analyst", PeerRole.ANALYST, "analysis"),
            profile("peer-skeptic", PeerRole.SKEPTIC, "verification"),
            profile("peer-synth", PeerRole.SYNTHESIZER, "synthesis"),
        )
    )


def envelope(
    *,
    message_id: str = "msg-001",
    sender_id: str = "peer-source",
    recipient_id: str = "peer-analyst",
    message_type: MessageType = MessageType.STATUS,
    payload: object | None = None,
) -> MessageEnvelope:
    if payload is None:
        payload = StatusPayload(status=PeerStatus.ACTIVE, detail="Working")
    return MessageEnvelope(
        message_id=message_id,
        mission_id="mission-001",
        sender_id=sender_id,
        recipient_id=recipient_id,
        message_type=message_type,
        created_at=datetime.now(timezone.utc),
        payload=payload,
    )


def test_registry_rejects_duplicate_peer_ids() -> None:
    registry = PeerRegistry()
    peer = profile("peer-analyst", PeerRole.ANALYST, "analysis")
    registry.register(peer)

    with pytest.raises(DuplicatePeerError, match="peer already registered"):
        registry.register(peer)


def test_registry_discovers_by_capability_and_role() -> None:
    registry = make_registry()

    by_capability = registry.discover(capability="analysis")
    by_role = registry.discover(role=PeerRole.SKEPTIC)

    assert tuple(peer.agent_id for peer in by_capability) == ("peer-analyst",)
    assert tuple(peer.agent_id for peer in by_role) == ("peer-skeptic",)


def test_registry_excludes_unavailable_peers_by_default() -> None:
    registry = make_registry()
    registry.set_availability("peer-analyst", False)

    available = registry.discover()
    all_peers = registry.discover(available_only=False)

    assert "peer-analyst" not in {peer.agent_id for peer in available}
    assert "peer-analyst" in {peer.agent_id for peer in all_peers}


def test_registry_rejects_unknown_peer_and_ineligible_role() -> None:
    registry = make_registry()

    with pytest.raises(UnknownPeerError, match="unknown peer"):
        registry.get("peer-missing")

    with pytest.raises(IneligibleRoleError, match="not eligible"):
        registry.require_role_eligibility("peer-source", PeerRole.ANALYST)


def test_direct_message_delivers_only_to_declared_recipient() -> None:
    bus = MessageBus(make_registry())

    receipt = bus.send(envelope())

    assert receipt.status is TransportStatus.DELIVERED
    assert receipt.recipients == ("peer-analyst",)
    assert len(bus.peek("peer-analyst")) == 1
    assert bus.peek("peer-skeptic") == ()


def test_broadcast_transport_does_not_semantically_route_work() -> None:
    bus = MessageBus(make_registry())
    request = envelope(
        message_id="msg-work",
        recipient_id=BROADCAST_RECIPIENT,
        message_type=MessageType.WORK_REQUEST,
        payload=WorkRequestPayload(
            work_id="work-001",
            requested_capability="analysis",
            summary="Analyze the supplied evidence.",
        ),
    )

    receipt = bus.send(request)

    # The bus does not pick only the analyst. Every available peer except the
    # sender receives the broadcast and will later decide locally whether to act.
    assert receipt.recipients == ("peer-analyst", "peer-skeptic", "peer-synth")
    assert len(bus.peek("peer-analyst")) == 1
    assert len(bus.peek("peer-skeptic")) == 1
    assert len(bus.peek("peer-synth")) == 1


def test_duplicate_message_is_not_redelivered() -> None:
    bus = MessageBus(make_registry())
    message = envelope()

    first = bus.send(message)
    duplicate = bus.send(message)

    assert first.status is TransportStatus.DELIVERED
    assert duplicate.status is TransportStatus.DUPLICATE
    assert len(bus.peek("peer-analyst")) == 1
    assert tuple(event.status for event in bus.audit_log) == (
        TransportStatus.DELIVERED,
        TransportStatus.DUPLICATE,
    )


def test_unknown_sender_is_rejected_and_audited() -> None:
    bus = MessageBus(make_registry())
    message = envelope(sender_id="peer-intruder")

    with pytest.raises(TransportError, match="unknown peer"):
        bus.send(message)

    assert bus.peek("peer-analyst") == ()
    assert bus.audit_log[-1].status is TransportStatus.REJECTED


def test_unknown_or_unavailable_recipient_is_rejected() -> None:
    registry = make_registry()
    bus = MessageBus(registry)

    with pytest.raises(TransportError, match="unknown peer"):
        bus.send(envelope(recipient_id="peer-missing"))

    registry.set_availability("peer-analyst", False)
    with pytest.raises(TransportError, match="unavailable"):
        bus.send(envelope(message_id="msg-002"))


def test_ineligible_role_claim_is_rejected_contextually() -> None:
    bus = MessageBus(make_registry())
    claim = envelope(
        message_type=MessageType.ROLE_CLAIM,
        payload=RoleClaimPayload(
            role=PeerRole.ANALYST,
            reason="I want to analyze the mission.",
        ),
    )

    with pytest.raises(TransportError, match="not eligible"):
        bus.send(claim)

    assert bus.audit_log[-1].status is TransportStatus.REJECTED


def test_capability_advertisement_cannot_expand_registered_authority() -> None:
    bus = MessageBus(make_registry())
    forged_profile = profile("peer-source", PeerRole.ANALYST, "analysis")
    advertisement = envelope(
        recipient_id=BROADCAST_RECIPIENT,
        message_type=MessageType.CAPABILITY_ADVERTISEMENT,
        payload=CapabilityAdvertisementPayload(profile=forged_profile),
    )

    with pytest.raises(TransportError, match="does not match registered profile"):
        bus.send(advertisement)


def test_drain_returns_messages_once_and_clears_inbox() -> None:
    bus = MessageBus(make_registry())
    bus.send(envelope())

    drained = bus.drain("peer-analyst")

    assert len(drained) == 1
    assert bus.peek("peer-analyst") == ()


def test_direct_message_to_self_is_rejected() -> None:
    bus = MessageBus(make_registry())

    with pytest.raises(TransportError, match="cannot target the sender itself"):
        bus.send(envelope(recipient_id="peer-source"))
