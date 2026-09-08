"""Deterministic challenge/revision scenario for the Agent 10 research mission.

The first analysis draft intentionally omits one declared evidence ID from its
WORK_RESULT envelope while retaining the correct underlying analysis artifact.
The Skeptic detects that protocol-quality defect, challenges the Analyst directly,
and independently validates the revision before accepting a re-verification pass.
"""

from __future__ import annotations

from typing import Literal

from .challenge import (
    ChallengeCapablePeerAgent,
    ChallengeEmission,
    ChallengeEvaluation,
    ChallengeHandlingContext,
    ChallengeHandlingOutcome,
    ChallengeResponseContext,
    ChallengeTriggerContext,
)
from .models import (
    ChallengePayload,
    ChallengeResolution,
    ChallengeResponsePayload,
    WorkRequestPayload,
    WorkResultPayload,
)
from .registry import PeerRegistry
from .research import (
    ANALYSIS_CAPABILITY,
    ANALYSIS_WORK_ID,
    ANALYST_AGENT_ID,
    FINAL_WORK_PRODUCT_ID,
    SKEPTIC_AGENT_ID,
    SOURCE_AGENT_ID,
    SYNTHESIZER_AGENT_ID,
    VERIFICATION_CAPABILITY,
    EvidenceItem,
    ResearchDemoSystem,
    VerificationArtifact,
    VerificationStatus,
    _analysis_handler,
    _recommendation_for_score,
    _source_handler,
    _synthesis_handler,
    _verification_handler,
    build_research_mission,
    build_research_profiles,
    build_synthetic_corpus,
    make_initial_research_request,
)
from .runtime import PeerAgent, PeerNetworkRuntime, WorkExecutionContext, WorkExecutionOutcome
from .transport import MessageBus

CHALLENGE_ID = "challenge-analysis-001"
REVISED_ANALYSIS_PRODUCT_ID = "wp:work-analysis-001:r1"
VERIFICATION_RECHECK_WORK_ID = "work-verification-002"
RevisionMode = Literal["fix", "false_fix", "dispute"]


def _analysis_with_declared_evidence_gap(
    context: WorkExecutionContext,
) -> WorkExecutionOutcome:
    """Create a correct analysis artifact with one envelope-level evidence omission."""

    base = _analysis_handler(context)
    if len(base.result.evidence_ids) < 2:
        raise ValueError("challenge demo requires multiple evidence IDs")

    flawed = base.result.model_copy(
        update={
            "summary": (
                "First-pass analysis is substantively complete but its declared evidence "
                "set contains a correctable omission for peer review."
            ),
            "evidence_ids": base.result.evidence_ids[:-1],
        }
    )
    return WorkExecutionOutcome(
        result=flawed,
        next_request=base.next_request,
        mark_peer_completed=True,
    )


def _verification_with_review_window(
    context: WorkExecutionContext,
) -> WorkExecutionOutcome:
    """Keep the Skeptic active when verification requires a challenge/revision."""

    base = _verification_handler(context)
    artifact = VerificationArtifact.model_validate(base.result.metadata.get("artifact"))
    if artifact.status is VerificationStatus.REQUIRES_REVISION:
        return WorkExecutionOutcome(
            result=base.result,
            next_request=None,
            mark_peer_completed=False,
        )
    return base


def _challenge_from_failed_verification(
    context: ChallengeTriggerContext,
) -> ChallengeEmission | None:
    if context.produced_result.metadata.get("artifact_type") != "verification":
        return None

    artifact = VerificationArtifact.model_validate(
        context.produced_result.metadata.get("artifact")
    )
    if artifact.status is VerificationStatus.VERIFIED:
        return None
    if not artifact.issues:
        raise ValueError("revision-required verification must name at least one issue")

    return ChallengeEmission(
        recipient_id=ANALYST_AGENT_ID,
        payload=ChallengePayload(
            challenge_id=CHALLENGE_ID,
            work_id=ANALYSIS_WORK_ID,
            issue="; ".join(artifact.issues),
            evidence_ids=context.produced_result.evidence_ids,
        ),
    )


def _analysis_challenge_handler(mode: RevisionMode):
    def handle(context: ChallengeHandlingContext) -> ChallengeHandlingOutcome:
        if context.challenge.challenge_id != CHALLENGE_ID:
            raise ValueError("unexpected challenge identifier")
        if context.challenge.work_id != ANALYSIS_WORK_ID:
            raise ValueError("challenge does not target the analysis work item")

        if mode == "dispute":
            return ChallengeHandlingOutcome(
                response=ChallengeResponsePayload(
                    challenge_id=context.challenge.challenge_id,
                    response=(
                        "The Analyst disputes the challenge and publishes no replacement "
                        "work product."
                    ),
                    resolution=ChallengeResolution.DISPUTED,
                    evidence_ids=context.challenge.evidence_ids,
                )
            )

        source = next(
            (
                product
                for product in context.observed_work_products
                if product.metadata.get("artifact_type") == "retrieval"
            ),
            None,
        )
        original = next(
            (
                product
                for product in context.observed_work_products
                if product.metadata.get("artifact_type") == "analysis"
                and product.work_id == ANALYSIS_WORK_ID
                and product.work_product_id != REVISED_ANALYSIS_PRODUCT_ID
            ),
            None,
        )
        if source is None or original is None:
            raise ValueError("Analyst cannot revise without local retrieval and analysis products")

        revised_evidence_ids = (
            source.evidence_ids
            if mode == "fix"
            else source.evidence_ids[:-1]
        )
        revised = original.model_copy(
            update={
                "work_product_id": REVISED_ANALYSIS_PRODUCT_ID,
                "summary": (
                    "Revised analysis republishes the same deterministic conclusions with "
                    "a corrected declared evidence set."
                ),
                "evidence_ids": revised_evidence_ids,
            }
        )
        response = ChallengeResponsePayload(
            challenge_id=context.challenge.challenge_id,
            response=(
                "The Analyst accepts the declared-evidence challenge and publishes a "
                "replacement analysis work product for independent re-verification."
            ),
            resolution=ChallengeResolution.REVISED,
            evidence_ids=revised_evidence_ids,
        )
        recheck = WorkRequestPayload(
            work_id=VERIFICATION_RECHECK_WORK_ID,
            requested_capability=VERIFICATION_CAPABILITY,
            summary=(
                "Re-verify the revised analysis against the original retrieval evidence "
                "before any final synthesis is allowed."
            ),
            input_work_product_ids=(
                source.work_product_id,
                revised.work_product_id,
            ),
        )
        return ChallengeHandlingOutcome(
            response=response,
            revised_result=revised,
            next_request=recheck,
        )

    return handle


def _evaluate_analysis_revision(
    context: ChallengeResponseContext,
) -> ChallengeEvaluation:
    """Independently determine whether the Analyst actually fixed the challenged defect."""

    if context.response.resolution is not ChallengeResolution.REVISED:
        return ChallengeEvaluation(
            resolved=False,
            reason="Analyst did not provide a revision for the challenged work product",
        )

    source = next(
        (
            product
            for product in context.observed_work_products
            if product.metadata.get("artifact_type") == "retrieval"
        ),
        None,
    )
    revised = next(
        (
            product
            for product in context.observed_work_products
            if product.work_product_id == REVISED_ANALYSIS_PRODUCT_ID
            and product.work_id == context.challenge.work_id
        ),
        None,
    )
    if source is None or revised is None:
        return ChallengeEvaluation(
            resolved=False,
            reason="claimed revision is not available in the Skeptic's local work-product view",
        )

    if set(revised.evidence_ids) != set(source.evidence_ids):
        return ChallengeEvaluation(
            resolved=False,
            reason="revised analysis still does not declare the complete retrieval evidence set",
        )
    if set(context.response.evidence_ids) != set(source.evidence_ids):
        return ChallengeEvaluation(
            resolved=False,
            reason="challenge response evidence references do not match the corrected revision",
        )

    raw_evidence = source.metadata.get("evidence")
    if not isinstance(raw_evidence, list):
        return ChallengeEvaluation(
            resolved=False,
            reason="retrieval evidence records are unavailable for deterministic recheck",
        )
    evidence = tuple(EvidenceItem.model_validate(item) for item in raw_evidence)
    expected_score = sum(item.weight for item in evidence)

    from .research import AnalysisArtifact

    analysis = AnalysisArtifact.model_validate(revised.metadata.get("artifact"))
    if analysis.signal_score != expected_score:
        return ChallengeEvaluation(
            resolved=False,
            reason="revised analysis score does not match the deterministic evidence score",
        )
    if analysis.recommendation is not _recommendation_for_score(expected_score):
        return ChallengeEvaluation(
            resolved=False,
            reason="revised analysis recommendation does not match deterministic policy",
        )

    return ChallengeEvaluation(
        resolved=True,
        reason="replacement analysis corrects the declared-evidence defect",
    )


def build_challenge_research_demo_system(
    revision_mode: RevisionMode = "fix",
) -> ResearchDemoSystem:
    if revision_mode not in {"fix", "false_fix", "dispute"}:
        raise ValueError("revision_mode must be fix, false_fix, or dispute")

    mission = build_research_mission()
    corpus = build_synthetic_corpus()
    registry = PeerRegistry(build_research_profiles())
    bus = MessageBus(registry)

    peers = (
        PeerAgent(
            agent_id=SOURCE_AGENT_ID,
            mission=mission,
            registry=registry,
            bus=bus,
            handlers={"source_retrieval": _source_handler(corpus)},
        ),
        ChallengeCapablePeerAgent(
            agent_id=ANALYST_AGENT_ID,
            mission=mission,
            registry=registry,
            bus=bus,
            handlers={ANALYSIS_CAPABILITY: _analysis_with_declared_evidence_gap},
            challenge_handler=_analysis_challenge_handler(revision_mode),
        ),
        ChallengeCapablePeerAgent(
            agent_id=SKEPTIC_AGENT_ID,
            mission=mission,
            registry=registry,
            bus=bus,
            handlers={VERIFICATION_CAPABILITY: _verification_with_review_window},
            challenge_factory=_challenge_from_failed_verification,
            challenge_response_evaluator=_evaluate_analysis_revision,
        ),
        PeerAgent(
            agent_id=SYNTHESIZER_AGENT_ID,
            mission=mission,
            registry=registry,
            bus=bus,
            handlers={"synthesis": _synthesis_handler},
        ),
    )
    return ResearchDemoSystem(
        mission=mission,
        corpus=corpus,
        registry=registry,
        bus=bus,
        runtime=PeerNetworkRuntime(peers),
    )


def run_challenge_research_demo(max_cycles: int = 12) -> WorkResultPayload:
    """Run the research mission through challenge, revision, re-verification, and synthesis."""

    if max_cycles < 1:
        raise ValueError("max_cycles must be at least 1")

    system = build_challenge_research_demo_system("fix")
    initiator = system.runtime.get_peer(SYNTHESIZER_AGENT_ID)
    initiator.broadcast_mission()
    system.runtime.run_round()
    system.runtime.run_round()
    initiator.broadcast_work_request(make_initial_research_request())

    for _ in range(max_cycles):
        system.runtime.run_round()
        system.runtime.run_execution_round()
        final = system.runtime.get_peer(SYNTHESIZER_AGENT_ID).work_products.get(
            FINAL_WORK_PRODUCT_ID
        )
        if final is not None:
            return final

    raise RuntimeError("challenge research mission did not produce a verified final brief")
