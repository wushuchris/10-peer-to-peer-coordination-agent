"""Deterministic evaluation harness for peer-to-peer versus centralized coordination.

The comparison keeps the fictional research mission, evidence, handlers, and typed
work products constant. Only coordination authority changes:

* Peer-to-peer: peers self-select work through the engineered protocol.
* Centralized: one coordinator explicitly maps capabilities to workers and invokes
  the same local handlers in sequence.

The harness measures observable tradeoffs instead of assuming either architecture
is universally better.
"""

from __future__ import annotations

import hashlib
from enum import Enum

from .models import StrictModel, WorkRequestPayload, WorkResultPayload
from .research import (
    ANALYSIS_CAPABILITY,
    ANALYST_AGENT_ID,
    FINAL_WORK_PRODUCT_ID,
    SOURCE_AGENT_ID,
    SOURCE_CAPABILITY,
    SKEPTIC_AGENT_ID,
    SYNTHESIS_CAPABILITY,
    SYNTHESIZER_AGENT_ID,
    VERIFICATION_CAPABILITY,
    build_research_demo_system,
    make_initial_research_request,
)
from .resilience import FailureContainmentRuntime, FailurePolicy, FailureSeverity
from .runtime import PeerAgent, WorkExecutionContext, WorkExecutionStatus
from .transport import TransportStatus


class CoordinationArchitecture(str, Enum):
    PEER_TO_PEER = "peer_to_peer"
    CENTRALIZED = "centralized"


class EvaluationScenario(str, Enum):
    HAPPY_PATH = "happy_path"
    COORDINATOR_OUTAGE = "coordinator_outage"
    UNAVAILABLE_ANALYST = "unavailable_analyst"


class EvaluationMetrics(StrictModel):
    """Comparable observable metrics for one architecture/scenario run."""

    architecture: CoordinationArchitecture
    scenario: EvaluationScenario
    completed: bool
    final_product_id: str | None = None
    final_product_digest: str | None = None
    coordination_steps: int
    protocol_messages: int
    message_deliveries: int
    work_executions: int
    audit_trace_items: int
    central_coordinator_required: bool
    failure_code: str | None = None
    failure_detail: str | None = None


class ArchitectureComparison(StrictModel):
    scenario: EvaluationScenario
    peer_to_peer: EvaluationMetrics
    centralized: EvaluationMetrics
    observations: tuple[str, ...]


class EvaluationReport(StrictModel):
    comparisons: tuple[ArchitectureComparison, ...]


def _digest_result(result: WorkResultPayload | None) -> str | None:
    if result is None:
        return None
    payload = result.model_dump_json(exclude_none=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _research_peers(system) -> tuple[PeerAgent, ...]:
    return tuple(
        system.runtime.get_peer(agent_id)
        for agent_id in (
            SOURCE_AGENT_ID,
            ANALYST_AGENT_ID,
            SKEPTIC_AGENT_ID,
            SYNTHESIZER_AGENT_ID,
        )
    )


def run_peer_to_peer_evaluation(
    scenario: EvaluationScenario,
    *,
    max_cycles: int = 12,
) -> EvaluationMetrics:
    """Evaluate the decentralized architecture with passive failure containment."""

    if max_cycles < 1:
        raise ValueError("max_cycles must be at least 1")

    system = build_research_demo_system()
    if scenario is EvaluationScenario.UNAVAILABLE_ANALYST:
        system.registry.set_availability(ANALYST_AGENT_ID, False)

    contained = FailureContainmentRuntime(
        _research_peers(system),
        registry=system.registry,
        policy=FailurePolicy(
            role_grace_rounds=3,
            role_conflict_rounds=2,
            dependency_wait_rounds=3,
            challenge_response_rounds=3,
            no_progress_rounds=3,
        ),
    )
    initiator = contained.get_peer(SYNTHESIZER_AGENT_ID)

    # A centralized-coordinator outage is intentionally a no-op here because the
    # peer-to-peer architecture has no such semantic control-plane component.
    initiator.broadcast_mission()

    coordination_steps = 0
    work_executions = 0
    trace_items = 0

    def advance() -> None:
        nonlocal coordination_steps, work_executions, trace_items
        cycle = contained.run_cycle()
        coordination_steps += 1
        work_executions += sum(
            execution.status is WorkExecutionStatus.EXECUTED
            for execution in cycle.executions
        )
        trace_items += sum(len(turn.decisions) for turn in cycle.turns)
        trace_items += len(cycle.executions) + len(cycle.health.issues)

    # Two observation cycles allow role claims to propagate before work begins.
    advance()
    if not contained.blocked:
        advance()

    if not contained.blocked:
        initiator.broadcast_work_request(make_initial_research_request())

    final: WorkResultPayload | None = None
    for _ in range(max_cycles):
        if contained.blocked:
            break
        advance()
        final = contained.get_peer(SYNTHESIZER_AGENT_ID).work_products.get(
            FINAL_WORK_PRODUCT_ID
        )
        if final is not None:
            break

    audit = system.bus.audit_log
    delivered_events = tuple(
        event for event in audit if event.status is TransportStatus.DELIVERED
    )
    trace_items += len(audit)

    blocking = tuple(
        issue
        for issue in contained.health.issues
        if issue.severity is FailureSeverity.BLOCKING
    )
    first_issue = blocking[0] if blocking else None

    return EvaluationMetrics(
        architecture=CoordinationArchitecture.PEER_TO_PEER,
        scenario=scenario,
        completed=final is not None,
        final_product_id=final.work_product_id if final is not None else None,
        final_product_digest=_digest_result(final),
        coordination_steps=coordination_steps,
        protocol_messages=len(delivered_events),
        message_deliveries=sum(len(event.recipients) for event in delivered_events),
        work_executions=work_executions,
        audit_trace_items=trace_items,
        central_coordinator_required=False,
        failure_code=first_issue.code.value if first_issue is not None else None,
        failure_detail=first_issue.detail if first_issue is not None else None,
    )


class CentralizedResearchCoordinator:
    """Agent-8-style deterministic allocator used only as a comparison baseline.

    The coordinator owns the capability-to-worker mapping and directly invokes the
    same work handlers used by the peer implementation. If the coordinator is down,
    no semantic coordination can occur.
    """

    _CAPABILITY_OWNER = {
        SOURCE_CAPABILITY: SOURCE_AGENT_ID,
        ANALYSIS_CAPABILITY: ANALYST_AGENT_ID,
        VERIFICATION_CAPABILITY: SKEPTIC_AGENT_ID,
        SYNTHESIS_CAPABILITY: SYNTHESIZER_AGENT_ID,
    }

    def __init__(self, *, coordinator_available: bool = True) -> None:
        self.system = build_research_demo_system()
        self.coordinator_available = coordinator_available
        self.products: dict[str, WorkResultPayload] = {}
        self.trace: list[str] = []
        self.work_executions = 0

    def run(self) -> tuple[WorkResultPayload | None, str | None, str | None]:
        if not self.coordinator_available:
            self.trace.append("coordinator unavailable; centralized execution cannot start")
            return None, "coordinator_unavailable", self.trace[-1]

        request: WorkRequestPayload | None = make_initial_research_request()
        while request is not None:
            owner = self._CAPABILITY_OWNER.get(request.requested_capability)
            if owner is None:
                detail = f"central allocator has no owner for capability {request.requested_capability}"
                self.trace.append(detail)
                return None, "unassigned_capability", detail

            profile = self.system.registry.get(owner)
            if not profile.available:
                detail = f"centrally assigned peer is unavailable: {owner}"
                self.trace.append(detail)
                return None, "assigned_peer_unavailable", detail

            peer = self.system.runtime.get_peer(owner)
            handler = peer.handlers.get(request.requested_capability)
            if handler is None:
                detail = (
                    f"centrally assigned peer {owner} has no handler for "
                    f"{request.requested_capability}"
                )
                self.trace.append(detail)
                return None, "missing_handler", detail

            missing = tuple(
                product_id
                for product_id in request.input_work_product_ids
                if product_id not in self.products
            )
            if missing:
                detail = f"centralized step is missing dependencies: {', '.join(missing)}"
                self.trace.append(detail)
                return None, "missing_dependency", detail

            context = WorkExecutionContext(
                mission=self.system.mission.model_copy(deep=True),
                profile=profile.model_copy(deep=True),
                state=peer.state.model_copy(deep=True),
                request=request.model_copy(deep=True),
                input_products=tuple(
                    self.products[product_id].model_copy(deep=True)
                    for product_id in request.input_work_product_ids
                ),
            )

            try:
                outcome = handler(context)
                PeerAgent._validate_work_outcome(request, outcome)
            except Exception as exc:
                detail = f"centralized handler failed closed: {type(exc).__name__}: {exc}"
                self.trace.append(detail)
                return None, "handler_failure", detail

            existing = self.products.get(outcome.result.work_product_id)
            if existing is not None and existing != outcome.result:
                detail = f"conflicting centralized work product: {outcome.result.work_product_id}"
                self.trace.append(detail)
                return None, "work_product_conflict", detail

            self.products[outcome.result.work_product_id] = outcome.result
            self.work_executions += 1
            self.trace.append(
                f"assigned {request.work_id} ({request.requested_capability}) to {owner}; "
                f"published {outcome.result.work_product_id}"
            )
            request = outcome.next_request

        final = self.products.get(FINAL_WORK_PRODUCT_ID)
        if final is None:
            detail = "centralized plan ended without the required final work product"
            self.trace.append(detail)
            return None, "missing_final_product", detail
        return final, None, None


def run_centralized_evaluation(
    scenario: EvaluationScenario,
) -> EvaluationMetrics:
    """Evaluate the same research work under a centralized semantic coordinator."""

    coordinator = CentralizedResearchCoordinator(
        coordinator_available=scenario is not EvaluationScenario.COORDINATOR_OUTAGE
    )
    if scenario is EvaluationScenario.UNAVAILABLE_ANALYST:
        coordinator.system.registry.set_availability(ANALYST_AGENT_ID, False)

    final, failure_code, failure_detail = coordinator.run()
    return EvaluationMetrics(
        architecture=CoordinationArchitecture.CENTRALIZED,
        scenario=scenario,
        completed=final is not None,
        final_product_id=final.work_product_id if final is not None else None,
        final_product_digest=_digest_result(final),
        coordination_steps=coordinator.work_executions,
        protocol_messages=0,
        message_deliveries=0,
        work_executions=coordinator.work_executions,
        audit_trace_items=len(coordinator.trace),
        central_coordinator_required=True,
        failure_code=failure_code,
        failure_detail=failure_detail,
    )


def _observations(
    scenario: EvaluationScenario,
    p2p: EvaluationMetrics,
    centralized: EvaluationMetrics,
) -> tuple[str, ...]:
    notes: list[str] = []
    if p2p.completed and centralized.completed:
        notes.append("Both architectures completed the same deterministic mission.")
    elif p2p.completed != centralized.completed:
        survivor = "peer-to-peer" if p2p.completed else "centralized"
        notes.append(f"Only the {survivor} architecture completed this scenario.")
    else:
        notes.append("Neither architecture completed this failure scenario.")

    if scenario is EvaluationScenario.HAPPY_PATH:
        notes.append(
            "Peer-to-peer coordination incurs explicit protocol-message and fan-out overhead; "
            "the centralized baseline replaces that overhead with direct allocator decisions."
        )
    if scenario is EvaluationScenario.COORDINATOR_OUTAGE:
        notes.append(
            "The centralized baseline has a semantic single point of dependency; the peer-to-peer "
            "system has no central coordinator to lose."
        )
    if scenario is EvaluationScenario.UNAVAILABLE_ANALYST:
        notes.append(
            "Both architectures fail closed when the only Analyst is unavailable, but they detect "
            "the failure through different control structures rather than inventing a replacement."
        )
    notes.append(
        "Both architectures retain deterministic trace evidence, but authority is distributed in "
        "the peer protocol and concentrated in the centralized allocator."
    )
    return tuple(notes)


def run_architecture_evaluation() -> EvaluationReport:
    """Run the bounded deterministic comparison suite for Agent 10."""

    comparisons: list[ArchitectureComparison] = []
    for scenario in EvaluationScenario:
        p2p = run_peer_to_peer_evaluation(scenario)
        centralized = run_centralized_evaluation(scenario)
        comparisons.append(
            ArchitectureComparison(
                scenario=scenario,
                peer_to_peer=p2p,
                centralized=centralized,
                observations=_observations(scenario, p2p, centralized),
            )
        )
    return EvaluationReport(comparisons=tuple(comparisons))
