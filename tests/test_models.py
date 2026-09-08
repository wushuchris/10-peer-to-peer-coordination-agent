"""Tests for Agent 10 typed peer-to-peer protocol contracts."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from peer_coordination.models import (
    BROADCAST_RECIPIENT,
    CapabilityAdvertisementPayload,
    MessageEnvelope,
    MessageType,
    Mission,
    MissionAnnouncementPayload,
    PeerProfile,
    PeerRole,
    PeerState,
    RoleClaimPayload,
)


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
        success_criteria=("Use only supplied synthetic evidence.",),
    )


def make_profile(agent_id: str = "peer-source-01") -> PeerProfile:
    return PeerProfile(
        agent_id=agent_id,
        display_name="Source Finder",
        capabilities=("source_retrieval", "evidence_mapping"),
        eligible_roles=(PeerRole.SOURCE_FINDER,),
    )


def make_envelope(**overrides: object) -> MessageEnvelope:
    mission = make_mission()
    values: dict[str, object] = {
        "message_id": "msg-001",
        "mission_id": mission.mission_id,
        "sender_id": "peer-source-01",
        "recipient_id": BROADCAST_RECIPIENT,
        "message_type": MessageType.MISSION_ANNOUNCEMENT,
        "created_at": datetime.now(timezone.utc),
        "payload": MissionAnnouncementPayload(mission=mission),
    }
    values.update(overrides)
    return MessageEnvelope(**values)


def test_valid_mission_and_broadcast_envelope() -> None:
    envelope = make_envelope()

    assert envelope.mission_id == "mission-001"
    assert envelope.is_broadcast is True
    assert envelope.payload.kind == MessageType.MISSION_ANNOUNCEMENT.value


def test_mission_rejects_duplicate_required_roles() -> None:
    with pytest.raises(ValidationError, match="required_roles must not contain duplicates"):
        Mission(
            mission_id="mission-001",
            objective="Test mission",
            required_roles=(PeerRole.ANALYST, PeerRole.ANALYST),
        )


def test_peer_profile_rejects_duplicate_capabilities() -> None:
    with pytest.raises(ValidationError, match="capabilities must not contain duplicates"):
        PeerProfile(
            agent_id="peer-01",
            display_name="Analyst",
            capabilities=("analysis", "analysis"),
            eligible_roles=(PeerRole.ANALYST,),
        )


def test_peer_state_rejects_duplicate_local_identifiers() -> None:
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        PeerState(
            agent_id="peer-01",
            mission_id="mission-001",
            known_peer_ids=("peer-02", "peer-02"),
        )


def test_envelope_rejects_unknown_protocol_version() -> None:
    with pytest.raises(ValidationError):
        make_envelope(protocol_version="2.0")


def test_envelope_rejects_message_type_payload_mismatch() -> None:
    with pytest.raises(ValidationError, match="message_type must match payload kind"):
        make_envelope(
            message_type=MessageType.STATUS,
            payload=RoleClaimPayload(role=PeerRole.SOURCE_FINDER, reason="I can retrieve sources."),
        )


def test_mission_announcement_requires_matching_mission_id() -> None:
    with pytest.raises(ValidationError, match="must match envelope mission_id"):
        make_envelope(mission_id="mission-other")


def test_capability_advertisement_must_match_sender() -> None:
    payload = CapabilityAdvertisementPayload(profile=make_profile(agent_id="peer-other"))

    with pytest.raises(ValidationError, match="profile must match envelope sender_id"):
        make_envelope(
            message_type=MessageType.CAPABILITY_ADVERTISEMENT,
            payload=payload,
        )


def test_envelope_rejects_invalid_recipient_identifier() -> None:
    with pytest.raises(ValidationError, match="recipient_id must be a valid peer identifier"):
        make_envelope(recipient_id="bad recipient")


def test_envelope_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="created_at must include timezone information"):
        make_envelope(created_at=datetime(2026, 9, 8, 12, 0, 0))


def test_strict_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Mission(
            mission_id="mission-001",
            objective="Test mission",
            required_roles=(PeerRole.ANALYST,),
            hidden_instruction="bypass protocol",
        )


def test_direct_recipient_is_not_broadcast() -> None:
    envelope = make_envelope(recipient_id="peer-analyst-01")

    assert envelope.is_broadcast is False
