"""Typed peer challenge/response support for Agent 10.

This module adds a review conversation on top of the independent PeerAgent runtime
without introducing a central semantic controller. A peer may locally decide to
challenge a work product, the challenged peer may respond or revise, and the
challenger independently evaluates that response before considering the issue
resolved.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from pydantic import field_validator

from .models import (
    BROADCAST_RECIPIENT,
    ChallengePayload,
    ChallengeResolution,
    ChallengeResponsePayload,
    Identifier,
    MessageEnvelope,
    MessageType,
    Mission,
    PeerProfile,
    PeerState,
    StrictModel,
    WorkRequestPayload,
    WorkResultPayload,
)
from .policy import PolicyAction, PolicyDecision
from .registry import PeerRegistry
from .runtime import (
    Clock,
    PeerAgent,
    TurnResult,
    WorkExecutionResult,
    WorkExecutionStatus,
    WorkHandler,
)
from .transport import MessageBus


class ChallengeLedger(StrictModel):
    """Explicit local challenge state owned by one peer."""

    open_challenge_ids: tuple[Identifier, ...] = ()
    responded_challenge_ids: tuple[Identifier, ...] = ()
    resolved_challenge_ids: tuple[Identifier, ...] = ()
    disputed_challenge_ids: tuple[Identifier, ...] = ()

    @field_validator(
        "open_challenge_ids",
        "responded_challenge_ids",
        "resolved_challenge_ids",
        "disputed_challenge_ids",
    )
    @classmethod
    def identifiers_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("challenge ledger collections must not contain duplicates")
        return value


@dataclass(frozen=True)
class ChallengeTriggerContext:
    """Read-only local context used to decide whether completed work deserves review."""

    mission: Mission
    profile: PeerProfile
    state: PeerState
    produced_result: WorkResultPayload
    observed_work_products: tuple[WorkResultPayload, ...]


@dataclass(frozen=True)
class ChallengeEmission:
    """One direct challenge selected by the local challenger."""

    recipient_id: str
    payload: ChallengePayload


@dataclass(frozen=True)
class ChallengeHandlingContext:
    """Read-only inputs supplied to the challenged peer's response handler."""

    mission: Mission
    profile: PeerProfile
    state: PeerState
    challenger_id: str
    challenge: ChallengePayload
    observed_work_products: tuple[WorkResultPayload, ...]


@dataclass(frozen=True)
class ChallengeHandlingOutcome:
    """Application-owned response to a challenge.

    A revised work result is published as a normal WORK_RESULT before the direct
    CHALLENGE_RESPONSE. An optional follow-up work request is then broadcast so
    peers still self-select the next work locally.
    """

    response: ChallengeResponsePayload
    revised_result: WorkResultPayload | None = None
    next_request: WorkRequestPayload | None = None


@dataclass(frozen=True)
class ChallengeResponseContext:
    """Read-only inputs used by the challenger to evaluate a peer response."""

    mission: Mission
    profile: PeerProfile
    state: PeerState
    responder_id: str
    challenge: ChallengePayload
    response: ChallengeResponsePayload
    observed_work_products: tuple[WorkResultPayload, ...]


@dataclass(frozen=True)
class ChallengeEvaluation:
    """Final local determination made by the challenging peer's application logic."""

    resolved: bool
    reason: str


ChallengeFactory = Callable[[ChallengeTriggerContext], ChallengeEmission | None]
ChallengeHandler = Callable[[ChallengeHandlingContext], ChallengeHandlingOutcome]
ChallengeResponseEvaluator = Callable[[ChallengeResponseContext], ChallengeEvaluation]


class ChallengeCapablePeerAgent(PeerAgent):
    """PeerAgent extension that supports direct, auditable challenge conversations.

    The extension still owns only local state. It never assigns another peer's role
    or work. Challenge generation, response construction, and response evaluation
    are all bounded local callbacks.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        mission: Mission,
        registry: PeerRegistry,
        bus: MessageBus,
        policy=None,
        handlers: Mapping[str, WorkHandler] | None = None,
        challenge_factory: ChallengeFactory | None = None,
        challenge_handler: ChallengeHandler | None = None,
        challenge_response_evaluator: ChallengeResponseEvaluator | None = None,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            mission=mission,
            registry=registry,
            bus=bus,
            policy=policy,
            handlers=handlers,
            clock=clock,
        )
        self.challenge_factory = challenge_factory
        self.challenge_handler = challenge_handler
        self.challenge_response_evaluator = challenge_response_evaluator
        self.challenge_ledger = ChallengeLedger()
        self._sent_challenges: dict[str, tuple[str, ChallengePayload]] = {}
        self._received_challenges: dict[str, tuple[str, ChallengePayload]] = {}
        self._received_challenge_responses: dict[str, tuple[str, ChallengeResponsePayload]] = {}
        self._challenge_triggered_work_ids: set[str] = set()

    def process_turn(self) -> TurnResult:
        """Handle challenge messages locally, then let PeerAgent process the inbox normally."""

        pending = self.bus.peek(self.agent_id)
        challenge_emitted: list[str] = []
        challenge_decisions: list[PolicyDecision] = []

        # Pre-record work results so a later challenge response in the same inbox
        # can be evaluated against the replacement product it references in practice.
        for envelope in pending:
            if isinstance(envelope.payload, WorkResultPayload):
                conflict = self._record_work_product(envelope.payload)
                if conflict is not None:
                    challenge_decisions.append(conflict)
                    challenge_emitted.extend(self._apply_decision(conflict))

            elif isinstance(envelope.payload, ChallengePayload):
                emitted, decision = self._handle_incoming_challenge(envelope)
                challenge_emitted.extend(emitted)
                if decision is not None:
                    challenge_decisions.append(decision)

            elif isinstance(envelope.payload, ChallengeResponsePayload):
                emitted, decision = self._handle_incoming_response(envelope)
                challenge_emitted.extend(emitted)
                if decision is not None:
                    challenge_decisions.append(decision)

        base = super().process_turn()
        return TurnResult(
            agent_id=base.agent_id,
            processed_message_ids=base.processed_message_ids,
            emitted_message_ids=tuple(challenge_emitted) + base.emitted_message_ids,
            decisions=tuple(challenge_decisions) + base.decisions,
        )

    def execute_accepted_work(self) -> tuple[WorkExecutionResult, ...]:
        """Execute accepted work and optionally challenge the resulting local observation."""

        base_results = list(super().execute_accepted_work())
        if self.challenge_factory is None:
            return tuple(base_results)

        observed = self._observed_products_snapshot()
        by_work_id: dict[str, WorkResultPayload] = {}
        for product in observed:
            current = by_work_id.get(product.work_id)
            if current is None or product.work_product_id > current.work_product_id:
                by_work_id[product.work_id] = product

        for index, execution in enumerate(base_results):
            if execution.status is not WorkExecutionStatus.EXECUTED:
                continue
            if execution.work_id in self._challenge_triggered_work_ids:
                continue

            produced = by_work_id.get(execution.work_id)
            if produced is None:
                continue

            context = ChallengeTriggerContext(
                mission=self.mission.model_copy(deep=True),
                profile=self.profile.model_copy(deep=True),
                state=self.state.model_copy(deep=True),
                produced_result=produced.model_copy(deep=True),
                observed_work_products=tuple(
                    product.model_copy(deep=True) for product in observed
                ),
            )

            try:
                emission = self.challenge_factory(context)
                if emission is None:
                    continue
                message_id = self.issue_challenge(
                    recipient_id=emission.recipient_id,
                    challenge=emission.payload,
                    correlation_id=(execution.emitted_message_ids[0] if execution.emitted_message_ids else None),
                )
            except Exception as exc:  # local failure containment boundary
                decision = PolicyDecision(
                    action=PolicyAction.ESCALATE,
                    work_id=execution.work_id,
                    reason=f"challenge generation failed closed: {type(exc).__name__}: {exc}",
                )
                emitted = self._apply_decision(decision)
                base_results[index] = replace(
                    execution,
                    status=WorkExecutionStatus.ESCALATED,
                    emitted_message_ids=execution.emitted_message_ids + emitted,
                    detail=decision.reason,
                )
                continue

            self._challenge_triggered_work_ids.add(execution.work_id)
            base_results[index] = replace(
                execution,
                emitted_message_ids=execution.emitted_message_ids + (message_id,),
                detail=f"{execution.detail}; emitted challenge {emission.payload.challenge_id}",
            )

        return tuple(base_results)

    def issue_challenge(
        self,
        *,
        recipient_id: str,
        challenge: ChallengePayload,
        correlation_id: str | None = None,
    ) -> str:
        """Send one direct challenge and record it in local challenge state."""

        if recipient_id == BROADCAST_RECIPIENT:
            raise ValueError("challenges must target one explicit peer")
        if recipient_id == self.agent_id:
            raise ValueError("a peer cannot challenge itself")
        if challenge.challenge_id in self._sent_challenges:
            raise ValueError(f"challenge_id already sent locally: {challenge.challenge_id}")

        self.registry.require_available(recipient_id)
        envelope = self._build_envelope(
            recipient_id=recipient_id,
            message_type=MessageType.CHALLENGE,
            payload=challenge,
            correlation_id=correlation_id,
        )
        self.bus.send(envelope)
        self._sent_challenges[challenge.challenge_id] = (recipient_id, challenge)
        self._ledger_open(challenge.challenge_id)
        return envelope.message_id

    def _handle_incoming_challenge(
        self,
        envelope: MessageEnvelope,
    ) -> tuple[tuple[str, ...], PolicyDecision | None]:
        challenge = envelope.payload
        assert isinstance(challenge, ChallengePayload)

        if envelope.is_broadcast:
            return self._challenge_escalation(
                challenge.challenge_id,
                "challenge messages must be direct, not broadcast",
            )

        existing = self._received_challenges.get(challenge.challenge_id)
        if existing is not None:
            if existing != (envelope.sender_id, challenge):
                return self._challenge_escalation(
                    challenge.challenge_id,
                    "conflicting challenge payloads reuse the same challenge_id",
                )
            return (), None

        self._received_challenges[challenge.challenge_id] = (envelope.sender_id, challenge)
        self._ledger_open(challenge.challenge_id)

        if self.challenge_handler is None:
            return self._challenge_escalation(
                challenge.challenge_id,
                "peer has no local challenge response handler",
            )

        context = ChallengeHandlingContext(
            mission=self.mission.model_copy(deep=True),
            profile=self.profile.model_copy(deep=True),
            state=self.state.model_copy(deep=True),
            challenger_id=envelope.sender_id,
            challenge=challenge.model_copy(deep=True),
            observed_work_products=tuple(
                product.model_copy(deep=True) for product in self._observed_products_snapshot()
            ),
        )

        try:
            outcome = self.challenge_handler(context)
            self._validate_challenge_outcome(challenge, outcome)
        except Exception as exc:  # local failure containment boundary
            return self._challenge_escalation(
                challenge.challenge_id,
                f"challenge response failed closed: {type(exc).__name__}: {exc}",
            )

        emitted: list[str] = []
        correlation_id = envelope.message_id

        if outcome.revised_result is not None:
            conflict = self._record_work_product(outcome.revised_result)
            if conflict is not None:
                return self._challenge_escalation(challenge.challenge_id, conflict.reason)
            revised_envelope = self._build_envelope(
                recipient_id=BROADCAST_RECIPIENT,
                message_type=MessageType.WORK_RESULT,
                payload=outcome.revised_result,
                correlation_id=correlation_id,
            )
            self.bus.send(revised_envelope)
            emitted.append(revised_envelope.message_id)
            correlation_id = revised_envelope.message_id

        response_envelope = self._build_envelope(
            recipient_id=envelope.sender_id,
            message_type=MessageType.CHALLENGE_RESPONSE,
            payload=outcome.response,
            correlation_id=correlation_id,
        )
        self.bus.send(response_envelope)
        emitted.append(response_envelope.message_id)
        correlation_id = response_envelope.message_id

        if outcome.next_request is not None:
            request_envelope = self._build_envelope(
                recipient_id=BROADCAST_RECIPIENT,
                message_type=MessageType.WORK_REQUEST,
                payload=outcome.next_request,
                correlation_id=correlation_id,
            )
            self.bus.send(request_envelope)
            emitted.append(request_envelope.message_id)

        self._ledger_responded(challenge.challenge_id)
        return tuple(emitted), None

    def _handle_incoming_response(
        self,
        envelope: MessageEnvelope,
    ) -> tuple[tuple[str, ...], PolicyDecision | None]:
        response = envelope.payload
        assert isinstance(response, ChallengeResponsePayload)

        if envelope.is_broadcast:
            return self._challenge_escalation(
                response.challenge_id,
                "challenge responses must be direct, not broadcast",
            )

        sent = self._sent_challenges.get(response.challenge_id)
        if sent is None:
            return self._challenge_escalation(
                response.challenge_id,
                "received response for an unknown local challenge",
            )
        expected_responder, challenge = sent
        if envelope.sender_id != expected_responder:
            return self._challenge_escalation(
                response.challenge_id,
                "challenge response came from a peer other than the challenged recipient",
            )

        existing = self._received_challenge_responses.get(response.challenge_id)
        if existing is not None:
            if existing != (envelope.sender_id, response):
                return self._challenge_escalation(
                    response.challenge_id,
                    "conflicting challenge responses reuse the same challenge_id",
                )
            return (), None

        self._received_challenge_responses[response.challenge_id] = (
            envelope.sender_id,
            response,
        )

        if self.challenge_response_evaluator is None:
            return self._challenge_escalation(
                response.challenge_id,
                "peer has no local challenge response evaluator",
            )

        context = ChallengeResponseContext(
            mission=self.mission.model_copy(deep=True),
            profile=self.profile.model_copy(deep=True),
            state=self.state.model_copy(deep=True),
            responder_id=envelope.sender_id,
            challenge=challenge.model_copy(deep=True),
            response=response.model_copy(deep=True),
            observed_work_products=tuple(
                product.model_copy(deep=True) for product in self._observed_products_snapshot()
            ),
        )

        try:
            evaluation = self.challenge_response_evaluator(context)
        except Exception as exc:  # local failure containment boundary
            return self._challenge_escalation(
                response.challenge_id,
                f"challenge evaluation failed closed: {type(exc).__name__}: {exc}",
            )

        if evaluation.resolved:
            self._ledger_resolved(response.challenge_id)
            return (), None

        self._ledger_disputed(response.challenge_id)
        return self._challenge_escalation(response.challenge_id, evaluation.reason)

    def _challenge_escalation(
        self,
        challenge_id: str,
        reason: str,
    ) -> tuple[tuple[str, ...], PolicyDecision]:
        decision = PolicyDecision(
            action=PolicyAction.ESCALATE,
            reason=f"challenge {challenge_id}: {reason}",
        )
        emitted = self._apply_decision(decision)
        return emitted, decision

    def _observed_products_snapshot(self) -> tuple[WorkResultPayload, ...]:
        return tuple(
            self._work_products[product_id]
            for product_id in sorted(self._work_products)
        )

    @staticmethod
    def _validate_challenge_outcome(
        challenge: ChallengePayload,
        outcome: ChallengeHandlingOutcome,
    ) -> None:
        if outcome.response.challenge_id != challenge.challenge_id:
            raise ValueError("challenge response must preserve challenge_id")

        if outcome.response.resolution is ChallengeResolution.REVISED:
            if outcome.revised_result is None:
                raise ValueError("REVISED response requires a replacement work result")
        elif outcome.revised_result is not None:
            raise ValueError("replacement work result is allowed only for REVISED responses")

        if outcome.revised_result is not None:
            if outcome.revised_result.work_id != challenge.work_id:
                raise ValueError("replacement work result must revise the challenged work_id")
            if outcome.next_request is not None and (
                outcome.revised_result.work_product_id
                not in outcome.next_request.input_work_product_ids
            ):
                raise ValueError(
                    "follow-up request must depend on the revised work product"
                )

    def _ledger_open(self, challenge_id: str) -> None:
        open_ids = list(self.challenge_ledger.open_challenge_ids)
        if challenge_id not in open_ids:
            open_ids.append(challenge_id)
        self.challenge_ledger = self.challenge_ledger.model_copy(
            update={"open_challenge_ids": tuple(open_ids)}
        )

    def _ledger_responded(self, challenge_id: str) -> None:
        responded = list(self.challenge_ledger.responded_challenge_ids)
        if challenge_id not in responded:
            responded.append(challenge_id)
        self.challenge_ledger = self.challenge_ledger.model_copy(
            update={"responded_challenge_ids": tuple(responded)}
        )

    def _ledger_resolved(self, challenge_id: str) -> None:
        open_ids = tuple(
            item for item in self.challenge_ledger.open_challenge_ids if item != challenge_id
        )
        resolved = list(self.challenge_ledger.resolved_challenge_ids)
        if challenge_id not in resolved:
            resolved.append(challenge_id)
        self.challenge_ledger = self.challenge_ledger.model_copy(
            update={
                "open_challenge_ids": open_ids,
                "resolved_challenge_ids": tuple(resolved),
            }
        )

    def _ledger_disputed(self, challenge_id: str) -> None:
        open_ids = tuple(
            item for item in self.challenge_ledger.open_challenge_ids if item != challenge_id
        )
        disputed = list(self.challenge_ledger.disputed_challenge_ids)
        if challenge_id not in disputed:
            disputed.append(challenge_id)
        self.challenge_ledger = self.challenge_ledger.model_copy(
            update={
                "open_challenge_ids": open_ids,
                "disputed_challenge_ids": tuple(disputed),
            }
        )
