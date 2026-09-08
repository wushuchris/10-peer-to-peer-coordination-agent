"""Presentation-safe service layer for the Agent 10 Gradio demo.

The UI consumes these helpers instead of reimplementing coordination logic. Both
execution modes drive the same peer runtime; the only difference is whether bounded
work handlers are deterministic or model-assisted.
"""

from __future__ import annotations

from enum import Enum

from .evaluation import run_architecture_evaluation
from .hf_runtime import HuggingFaceStructuredChatClient
from .llm import JsonChatModel, ModelConfigurationError, build_llm_research_demo_system
from .llm_evaluation import run_llm_governance_evaluation
from .models import BROADCAST_RECIPIENT, PeerStatus, StrictModel, WorkResultPayload
from .research import (
    FINAL_WORK_PRODUCT_ID,
    SYNTHESIZER_AGENT_ID,
    FinalBriefArtifact,
    build_research_demo_system,
    make_initial_research_request,
)
from .runtime import WorkExecutionStatus


class DemoMode(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM_ASSISTED = "llm_assisted"


class DemoPeerRow(StrictModel):
    agent_id: str
    display_name: str
    role: str
    capability: str
    final_status: str
    accepted_work: str
    completed_work: str
    known_peers: int


class DemoEvidenceRow(StrictModel):
    evidence_id: str
    title: str
    signal: str
    weight: int
    finding: str


class DemoWorkProductRow(StrictModel):
    work_product_id: str
    work_id: str
    artifact_type: str
    evidence_ids: str
    summary: str


class DemoMessageRow(StrictModel):
    message_id: str
    message_type: str
    correlation_id: str
    sender: str
    recipient: str
    delivered_to: str
    status: str


class DemoConversationRow(StrictModel):
    """Human-readable rendering of one real protocol transport event."""

    step: int
    speaker: str
    audience: str
    statement: str
    protocol_event: str
    message_id: str
    correlation_id: str


class DemoSnapshot(StrictModel):
    mode: DemoMode
    completed: bool
    recommendation: str | None = None
    verification_status: str | None = None
    final_brief: str | None = None
    final_evidence_ids: tuple[str, ...] = ()
    error: str | None = None
    peers: tuple[DemoPeerRow, ...]
    evidence: tuple[DemoEvidenceRow, ...]
    work_products: tuple[DemoWorkProductRow, ...]
    messages: tuple[DemoMessageRow, ...]
    conversation: tuple[DemoConversationRow, ...]


def _drive_system(
    system,
    *,
    max_cycles: int = 12,
) -> tuple[WorkResultPayload | None, tuple[str, ...]]:
    if max_cycles < 1:
        raise ValueError("max_cycles must be at least 1")

    failure_details: list[str] = []
    initiator = system.runtime.get_peer(SYNTHESIZER_AGENT_ID)
    initiator.broadcast_mission()
    system.runtime.run_round()
    system.runtime.run_round()
    initiator.broadcast_work_request(make_initial_research_request())

    for _ in range(max_cycles):
        system.runtime.run_round()
        executions = system.runtime.run_execution_round()
        for execution in executions:
            if execution.status is WorkExecutionStatus.ESCALATED:
                failure_details.append(f"{execution.agent_id}: {execution.detail}")

        final = initiator.work_products.get(FINAL_WORK_PRODUCT_ID)
        if final is not None:
            return final, tuple(failure_details)
        if any(
            system.runtime.get_peer(profile.agent_id).state.status is PeerStatus.ESCALATED
            for profile in system.registry.all_profiles()
        ):
            return None, tuple(failure_details)
    return None, tuple(failure_details)


def _friendly_capability(value: str) -> str:
    return value.replace("_", " ")


def _conversation_statement(
    *,
    message_type: str,
    speaker_role: str,
    speaker_capability: str,
    request_number: int,
    delivered: bool,
) -> str:
    """Translate protocol mechanics into business-readable speech.

    The wording is intentionally generic: it describes what the observed event
    means without inventing payload facts that are not present in the audit log.
    """

    if not delivered:
        return "This coordination message was not delivered, so no peer acted on it."

    stage_labels = (
        "source evidence",
        "market analysis",
        "independent verification",
        "final decision brief",
    )
    stage = stage_labels[min(request_number, len(stage_labels) - 1)]

    statements = {
        "mission_announcement": (
            "Team, Asteria Robotics needs a market-entry decision. I’m sharing the mission "
            "so each peer can decide locally how to contribute."
        ),
        "capability_advertisement": (
            f"Here’s what I can contribute: {_friendly_capability(speaker_capability)}."
        ),
        "role_claim": f"I’ll take responsibility for the {speaker_role} role on this mission.",
        "role_release": (
            f"My {speaker_role} responsibilities are complete, so I’m releasing that role."
        ),
        "work_request": (
            f"The mission now needs {stage}. I’m asking the peer network to respond based "
            "on capability rather than assigning a worker centrally."
        ),
        "work_result": (
            f"I completed my {speaker_role} work and shared the resulting work product "
            "with the peer network."
        ),
        "challenge": (
            "I found something that needs to be challenged before the mission can safely continue."
        ),
        "challenge_response": (
            "I’m responding to the challenge with a correction or explanation for independent review."
        ),
        "status": "I’m sharing my current coordination status with the peer network.",
        "escalation": (
            "I can’t safely continue this work under the current conditions, so I’m escalating."
        ),
    }
    return statements.get(
        message_type,
        "I’m sharing a coordination event with the peer network.",
    )


def _conversation_rows(system) -> tuple[DemoConversationRow, ...]:
    profiles = {profile.agent_id: profile for profile in system.registry.all_profiles()}
    request_number = 0
    rows: list[DemoConversationRow] = []

    for step, event in enumerate(system.bus.audit_log, start=1):
        profile = profiles[event.sender_id]
        role = (
            profile.eligible_roles[0].value.replace("_", " ").title()
            if profile.eligible_roles
            else "Peer"
        )
        capability = profile.capabilities[0] if profile.capabilities else "coordination"
        if event.recipient_id == BROADCAST_RECIPIENT:
            audience = "Peer network"
        else:
            audience = profiles[event.recipient_id].display_name

        delivered = event.status.value == "delivered"
        statement = _conversation_statement(
            message_type=event.message_type,
            speaker_role=role,
            speaker_capability=capability,
            request_number=request_number,
            delivered=delivered,
        )
        rows.append(
            DemoConversationRow(
                step=step,
                speaker=profile.display_name,
                audience=audience,
                statement=statement,
                protocol_event=event.message_type,
                message_id=event.message_id,
                correlation_id=event.correlation_id or "—",
            )
        )
        if delivered and event.message_type == "work_request":
            request_number += 1

    return tuple(rows)


def run_demo(
    mode: DemoMode | str = DemoMode.DETERMINISTIC,
    *,
    model: JsonChatModel | None = None,
    max_cycles: int = 12,
) -> DemoSnapshot:
    """Run one public-safe market-entry mission and return a UI-ready snapshot."""

    resolved_mode = DemoMode(mode)
    try:
        if resolved_mode is DemoMode.LLM_ASSISTED:
            live_model = (
                model
                if model is not None
                else HuggingFaceStructuredChatClient.from_env()
            )
            system = build_llm_research_demo_system(live_model)
        else:
            system = build_research_demo_system()

        final, failure_details = _drive_system(system, max_cycles=max_cycles)
        error = None
        artifact: FinalBriefArtifact | None = None
        if final is not None:
            artifact = FinalBriefArtifact.model_validate(final.metadata.get("artifact"))
        else:
            escalated = tuple(
                profile.display_name
                for profile in system.registry.all_profiles()
                if system.runtime.get_peer(profile.agent_id).state.status
                is PeerStatus.ESCALATED
            )
            if escalated:
                error = f"Mission failed closed after peer escalation: {', '.join(escalated)}."
                if failure_details:
                    error += " Diagnostic: " + " | ".join(failure_details)
            else:
                error = "Mission did not produce a final brief within the bounded cycle limit."

        peers = tuple(
            DemoPeerRow(
                agent_id=profile.agent_id,
                display_name=profile.display_name,
                role=", ".join(role.value for role in profile.eligible_roles),
                capability=", ".join(profile.capabilities),
                final_status=system.runtime.get_peer(profile.agent_id).state.status.value,
                accepted_work=", ".join(
                    system.runtime.get_peer(profile.agent_id).state.accepted_work_ids
                )
                or "—",
                completed_work=", ".join(
                    system.runtime.get_peer(profile.agent_id).state.completed_work_ids
                )
                or "—",
                known_peers=len(
                    system.runtime.get_peer(profile.agent_id).state.known_peer_ids
                ),
            )
            for profile in system.registry.all_profiles()
        )

        evidence = tuple(
            DemoEvidenceRow(
                evidence_id=item.evidence_id,
                title=item.title,
                signal=item.signal.value,
                weight=item.weight,
                finding=item.finding,
            )
            for item in system.corpus.all_items()
        )

        unique_products: dict[str, WorkResultPayload] = {}
        for profile in system.registry.all_profiles():
            for product_id, product in system.runtime.get_peer(profile.agent_id).work_products.items():
                unique_products.setdefault(product_id, product)
        work_products = tuple(
            DemoWorkProductRow(
                work_product_id=product.work_product_id,
                work_id=product.work_id,
                artifact_type=str(product.metadata.get("artifact_type", "unknown")),
                evidence_ids=", ".join(product.evidence_ids) or "—",
                summary=product.summary,
            )
            for product in sorted(unique_products.values(), key=lambda item: item.work_product_id)
        )

        messages = tuple(
            DemoMessageRow(
                message_id=event.message_id,
                message_type=event.message_type,
                correlation_id=event.correlation_id or "—",
                sender=event.sender_id,
                recipient=event.recipient_id,
                delivered_to=", ".join(event.recipients) or "—",
                status=event.status.value,
            )
            for event in system.bus.audit_log
        )
        conversation = _conversation_rows(system)

        return DemoSnapshot(
            mode=resolved_mode,
            completed=artifact is not None,
            recommendation=(artifact.recommendation.value if artifact is not None else None),
            verification_status=(
                artifact.verification_status.value if artifact is not None else None
            ),
            final_brief=(artifact.brief if artifact is not None else None),
            final_evidence_ids=(artifact.evidence_ids if artifact is not None else ()),
            error=error,
            peers=peers,
            evidence=evidence,
            work_products=work_products,
            messages=messages,
            conversation=conversation,
        )
    except ModelConfigurationError as exc:
        raise ModelConfigurationError(
            "LLM-assisted mode is not configured for this deployment. " + str(exc)
        ) from exc


def architecture_comparison_rows() -> tuple[list[list[object]], tuple[str, ...]]:
    """Return UI-ready rows for the centralized-versus-P2P evaluation."""

    report = run_architecture_evaluation()
    rows: list[list[object]] = []
    observations: list[str] = []
    for comparison in report.comparisons:
        observations.extend(comparison.observations)
        for metrics in (comparison.peer_to_peer, comparison.centralized):
            rows.append(
                [
                    comparison.scenario.value,
                    metrics.architecture.value,
                    metrics.completed,
                    metrics.coordination_steps,
                    metrics.protocol_messages,
                    metrics.message_deliveries,
                    metrics.work_executions,
                    metrics.central_coordinator_required,
                    metrics.failure_code or "—",
                ]
            )
    return rows, tuple(dict.fromkeys(observations))


def llm_governance_rows() -> tuple[list[list[object]], tuple[str, ...]]:
    """Return UI-ready rows for the offline adversarial LLM evaluation matrix."""

    report = run_llm_governance_evaluation()
    rows = [
        [
            result.scenario.value,
            result.expected_completion,
            result.completed,
            ", ".join(result.escalated_agent_ids) or "—",
            " → ".join(result.model_calls) or "—",
            result.governance_preserved,
            (
                "n/a"
                if result.semantic_contradiction_blocked is None
                else result.semantic_contradiction_blocked
            ),
            result.finding,
        ]
        for result in report.results
    ]
    return rows, report.known_limitations


def llm_runtime_status() -> str:
    """Report live-model readiness without ever returning secret material."""

    try:
        client = HuggingFaceStructuredChatClient.from_env()
    except ModelConfigurationError:
        return "Not configured — deterministic mode remains fully available."
    return f"Configured for model: {client.model_id}"
