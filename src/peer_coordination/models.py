"""Typed protocol contracts for Agent 10 peer-to-peer coordination."""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

PROTOCOL_VERSION = "1.0"
BROADCAST_RECIPIENT = "*"

Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
LongText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
Capability = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    ),
]

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


class StrictModel(BaseModel):
    """Base model that rejects unknown fields and trims strings."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PeerRole(str, Enum):
    SOURCE_FINDER = "source_finder"
    ANALYST = "analyst"
    SKEPTIC = "skeptic"
    SYNTHESIZER = "synthesizer"


class PeerStatus(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    WAITING = "waiting"
    COMPLETED = "completed"
    ESCALATED = "escalated"


class ChallengeResolution(str, Enum):
    ACCEPTED = "accepted"
    REVISED = "revised"
    DISPUTED = "disputed"


class MessageType(str, Enum):
    MISSION_ANNOUNCEMENT = "mission_announcement"
    CAPABILITY_ADVERTISEMENT = "capability_advertisement"
    ROLE_CLAIM = "role_claim"
    ROLE_RELEASE = "role_release"
    WORK_REQUEST = "work_request"
    WORK_RESULT = "work_result"
    CHALLENGE = "challenge"
    CHALLENGE_RESPONSE = "challenge_response"
    STATUS = "status"
    ESCALATION = "escalation"


class Mission(StrictModel):
    """Shared mission contract propagated to participating peers."""

    mission_id: Identifier
    objective: LongText
    required_roles: tuple[PeerRole, ...] = Field(min_length=1)
    success_criteria: tuple[ShortText, ...] = ()

    @field_validator("required_roles")
    @classmethod
    def required_roles_must_be_unique(cls, value: tuple[PeerRole, ...]) -> tuple[PeerRole, ...]:
        if len(value) != len(set(value)):
            raise ValueError("required_roles must not contain duplicates")
        return value


class PeerProfile(StrictModel):
    """Discoverable identity and declared capabilities for one peer."""

    agent_id: Identifier
    display_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
    capabilities: tuple[Capability, ...] = Field(min_length=1)
    eligible_roles: tuple[PeerRole, ...] = Field(min_length=1)
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    available: bool = True

    @field_validator("capabilities")
    @classmethod
    def capabilities_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("capabilities must not contain duplicates")
        return value

    @field_validator("eligible_roles")
    @classmethod
    def eligible_roles_must_be_unique(cls, value: tuple[PeerRole, ...]) -> tuple[PeerRole, ...]:
        if len(value) != len(set(value)):
            raise ValueError("eligible_roles must not contain duplicates")
        return value


class PeerState(StrictModel):
    """One peer's explicit local view of a mission."""

    agent_id: Identifier
    mission_id: Identifier
    status: PeerStatus = PeerStatus.IDLE
    claimed_role: PeerRole | None = None
    known_peer_ids: tuple[Identifier, ...] = ()
    received_message_ids: tuple[Identifier, ...] = ()
    completed_work_ids: tuple[Identifier, ...] = ()

    @field_validator("known_peer_ids", "received_message_ids", "completed_work_ids")
    @classmethod
    def identifiers_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("local state identifier collections must not contain duplicates")
        return value


class MissionAnnouncementPayload(StrictModel):
    kind: Literal["mission_announcement"] = MessageType.MISSION_ANNOUNCEMENT.value
    mission: Mission


class CapabilityAdvertisementPayload(StrictModel):
    kind: Literal["capability_advertisement"] = MessageType.CAPABILITY_ADVERTISEMENT.value
    profile: PeerProfile


class RoleClaimPayload(StrictModel):
    kind: Literal["role_claim"] = MessageType.ROLE_CLAIM.value
    role: PeerRole
    reason: ShortText


class RoleReleasePayload(StrictModel):
    kind: Literal["role_release"] = MessageType.ROLE_RELEASE.value
    role: PeerRole
    reason: ShortText


class WorkRequestPayload(StrictModel):
    kind: Literal["work_request"] = MessageType.WORK_REQUEST.value
    work_id: Identifier
    requested_capability: Capability
    summary: ShortText


class WorkResultPayload(StrictModel):
    kind: Literal["work_result"] = MessageType.WORK_RESULT.value
    work_id: Identifier
    work_product_id: Identifier
    summary: LongText
    evidence_ids: tuple[Identifier, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("evidence_ids must not contain duplicates")
        return value


class ChallengePayload(StrictModel):
    kind: Literal["challenge"] = MessageType.CHALLENGE.value
    challenge_id: Identifier
    work_id: Identifier
    issue: LongText
    evidence_ids: tuple[Identifier, ...] = ()


class ChallengeResponsePayload(StrictModel):
    kind: Literal["challenge_response"] = MessageType.CHALLENGE_RESPONSE.value
    challenge_id: Identifier
    response: LongText
    resolution: ChallengeResolution
    evidence_ids: tuple[Identifier, ...] = ()


class StatusPayload(StrictModel):
    kind: Literal["status"] = MessageType.STATUS.value
    status: PeerStatus
    detail: ShortText


class EscalationPayload(StrictModel):
    kind: Literal["escalation"] = MessageType.ESCALATION.value
    reason: LongText
    related_message_ids: tuple[Identifier, ...] = ()


MessagePayload = Annotated[
    MissionAnnouncementPayload
    | CapabilityAdvertisementPayload
    | RoleClaimPayload
    | RoleReleasePayload
    | WorkRequestPayload
    | WorkResultPayload
    | ChallengePayload
    | ChallengeResponsePayload
    | StatusPayload
    | EscalationPayload,
    Field(discriminator="kind"),
]


class MessageEnvelope(StrictModel):
    """Validated transport envelope for every peer-to-peer protocol message."""

    message_id: Identifier
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    mission_id: Identifier
    sender_id: Identifier
    recipient_id: str
    message_type: MessageType
    correlation_id: Identifier | None = None
    created_at: datetime
    payload: MessagePayload

    @field_validator("recipient_id")
    @classmethod
    def recipient_must_be_peer_or_broadcast(cls, value: str) -> str:
        value = value.strip()
        if value == BROADCAST_RECIPIENT:
            return value
        if not _IDENTIFIER_PATTERN.fullmatch(value):
            raise ValueError("recipient_id must be a valid peer identifier or '*'")
        return value

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include timezone information")
        return value

    @model_validator(mode="after")
    def envelope_must_match_payload(self) -> "MessageEnvelope":
        if self.message_type.value != self.payload.kind:
            raise ValueError("message_type must match payload kind")

        if isinstance(self.payload, MissionAnnouncementPayload):
            if self.payload.mission.mission_id != self.mission_id:
                raise ValueError("mission announcement payload must match envelope mission_id")

        if isinstance(self.payload, CapabilityAdvertisementPayload):
            if self.payload.profile.agent_id != self.sender_id:
                raise ValueError("capability advertisement profile must match envelope sender_id")

        return self

    @property
    def is_broadcast(self) -> bool:
        return self.recipient_id == BROADCAST_RECIPIENT
