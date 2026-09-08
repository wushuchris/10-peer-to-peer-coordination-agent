"""Synthetic research mission and deterministic peer work handlers for Agent 10.

The scenario is intentionally fictional and public-safe. It demonstrates peer-to-peer
work propagation without relying on an LLM or a central task allocator.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from pydantic import Field, field_validator

from .models import (
    Identifier,
    LongText,
    Mission,
    PeerProfile,
    PeerRole,
    ShortText,
    StrictModel,
    WorkRequestPayload,
    WorkResultPayload,
)
from .registry import PeerRegistry
from .runtime import (
    PeerAgent,
    PeerNetworkRuntime,
    WorkExecutionContext,
    WorkExecutionOutcome,
)
from .transport import MessageBus

SOURCE_AGENT_ID = "peer-source-01"
ANALYST_AGENT_ID = "peer-analyst-01"
SKEPTIC_AGENT_ID = "peer-skeptic-01"
SYNTHESIZER_AGENT_ID = "peer-synth-01"

SOURCE_CAPABILITY = "source_retrieval"
ANALYSIS_CAPABILITY = "analysis"
VERIFICATION_CAPABILITY = "verification"
SYNTHESIS_CAPABILITY = "synthesis"

SOURCE_WORK_ID = "work-source-001"
ANALYSIS_WORK_ID = "work-analysis-001"
VERIFICATION_WORK_ID = "work-verification-001"
SYNTHESIS_WORK_ID = "work-synthesis-001"
FINAL_WORK_PRODUCT_ID = f"wp:{SYNTHESIS_WORK_ID}"


class EvidenceSignal(str, Enum):
    POSITIVE = "positive"
    RISK = "risk"


class ResearchRecommendation(str, Enum):
    ENTER_WITH_CONDITIONS = "enter_with_conditions"
    GATHER_MORE_EVIDENCE = "gather_more_evidence"
    DO_NOT_ENTER = "do_not_enter"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    REQUIRES_REVISION = "requires_revision"


class EvidenceItem(StrictModel):
    """One synthetic evidence record used by the demo mission."""

    evidence_id: Identifier
    title: ShortText
    finding: LongText
    topics: tuple[str, ...] = Field(min_length=1)
    signal: EvidenceSignal
    weight: int = Field(ge=-5, le=5)

    @field_validator("topics")
    @classmethod
    def topics_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(topic.strip().lower() for topic in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("evidence topics must not contain duplicates")
        return normalized


class ResearchClaim(StrictModel):
    claim_id: Identifier
    text: LongText
    evidence_ids: tuple[Identifier, ...] = Field(min_length=1)


class AnalysisArtifact(StrictModel):
    recommendation: ResearchRecommendation
    signal_score: int
    claims: tuple[ResearchClaim, ...] = Field(min_length=1)
    positive_evidence_ids: tuple[Identifier, ...]
    risk_evidence_ids: tuple[Identifier, ...]


class VerificationArtifact(StrictModel):
    status: VerificationStatus
    checked_claim_ids: tuple[Identifier, ...]
    verified_evidence_ids: tuple[Identifier, ...]
    issues: tuple[ShortText, ...] = ()


class FinalBriefArtifact(StrictModel):
    title: ShortText
    recommendation: ResearchRecommendation
    verification_status: VerificationStatus
    brief: LongText
    evidence_ids: tuple[Identifier, ...]


class EvidenceCorpus:
    """Small deterministic evidence store available only to the Source Finder."""

    def __init__(self, items: Iterable[EvidenceItem]) -> None:
        self._items: dict[str, EvidenceItem] = {}
        for item in items:
            if item.evidence_id in self._items:
                raise ValueError(f"duplicate evidence_id: {item.evidence_id}")
            self._items[item.evidence_id] = item

    def all_items(self) -> tuple[EvidenceItem, ...]:
        return tuple(self._items[evidence_id] for evidence_id in sorted(self._items))

    def search(self, text: str) -> tuple[EvidenceItem, ...]:
        """Return evidence whose declared topics appear in the request text."""

        tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
        matches = tuple(
            item
            for item in self.all_items()
            if any(topic in tokens for topic in item.topics)
        )
        return matches


@dataclass(frozen=True)
class ResearchDemoSystem:
    mission: Mission
    corpus: EvidenceCorpus
    registry: PeerRegistry
    bus: MessageBus
    runtime: PeerNetworkRuntime


def build_research_mission() -> Mission:
    return Mission(
        mission_id="mission-market-entry-001",
        objective=(
            "Determine whether fictional Asteria Robotics should enter the fictional "
            "Borealis industrial automation market using only the supplied synthetic evidence."
        ),
        required_roles=(
            PeerRole.SOURCE_FINDER,
            PeerRole.ANALYST,
            PeerRole.SKEPTIC,
            PeerRole.SYNTHESIZER,
        ),
        success_criteria=(
            "Use only synthetic evidence distributed through protocol work products.",
            "Link substantive conclusions to evidence identifiers.",
            "Require deterministic verification before a final brief is published.",
        ),
    )


def build_synthetic_corpus() -> EvidenceCorpus:
    """Create the fictional public-safe evidence set for the demo."""

    return EvidenceCorpus(
        (
            EvidenceItem(
                evidence_id="EV-001",
                title="Demand growth",
                finding=(
                    "A synthetic buyer survey projects 18% unit-demand growth over the next "
                    "12 months in the Borealis industrial automation category."
                ),
                topics=("demand", "market"),
                signal=EvidenceSignal.POSITIVE,
                weight=3,
            ),
            EvidenceItem(
                evidence_id="EV-002",
                title="Channel readiness",
                finding=(
                    "Seven of ten fictional distributors in the synthetic channel study said "
                    "they would consider a six-month Asteria pilot."
                ),
                topics=("partner", "channel"),
                signal=EvidenceSignal.POSITIVE,
                weight=2,
            ),
            EvidenceItem(
                evidence_id="EV-003",
                title="Modeled unit economics",
                finding=(
                    "The synthetic entry model estimates a 24% contribution margin after "
                    "freight and distributor economics at the target selling price."
                ),
                topics=("economics", "margin"),
                signal=EvidenceSignal.POSITIVE,
                weight=2,
            ),
            EvidenceItem(
                evidence_id="EV-004",
                title="Certification burden",
                finding=(
                    "Synthetic regulatory assumptions require nine months and approximately "
                    "$1.8 million of upfront certification and localization work."
                ),
                topics=("regulation", "compliance"),
                signal=EvidenceSignal.RISK,
                weight=-2,
            ),
            EvidenceItem(
                evidence_id="EV-005",
                title="Incumbent concentration",
                finding=(
                    "The synthetic competitor brief assigns 68% combined market share to the "
                    "two largest incumbent suppliers, both using multi-year discount contracts."
                ),
                topics=("competition", "market"),
                signal=EvidenceSignal.RISK,
                weight=-2,
            ),
            EvidenceItem(
                evidence_id="EV-006",
                title="Service coverage gap",
                finding=(
                    "The synthetic customer requirements brief calls for 48-hour local service, "
                    "while Asteria's modeled launch network initially covers 60% of target accounts."
                ),
                topics=("operations", "service"),
                signal=EvidenceSignal.RISK,
                weight=-1,
            ),
        )
    )


def build_research_profiles() -> tuple[PeerProfile, ...]:
    return (
        PeerProfile(
            agent_id=SOURCE_AGENT_ID,
            display_name="Source Finder",
            capabilities=(SOURCE_CAPABILITY,),
            eligible_roles=(PeerRole.SOURCE_FINDER,),
        ),
        PeerProfile(
            agent_id=ANALYST_AGENT_ID,
            display_name="Analyst",
            capabilities=(ANALYSIS_CAPABILITY,),
            eligible_roles=(PeerRole.ANALYST,),
        ),
        PeerProfile(
            agent_id=SKEPTIC_AGENT_ID,
            display_name="Skeptic / Verifier",
            capabilities=(VERIFICATION_CAPABILITY,),
            eligible_roles=(PeerRole.SKEPTIC,),
        ),
        PeerProfile(
            agent_id=SYNTHESIZER_AGENT_ID,
            display_name="Synthesizer",
            capabilities=(SYNTHESIS_CAPABILITY,),
            eligible_roles=(PeerRole.SYNTHESIZER,),
        ),
    )


def make_initial_research_request() -> WorkRequestPayload:
    return WorkRequestPayload(
        work_id=SOURCE_WORK_ID,
        requested_capability=SOURCE_CAPABILITY,
        summary=(
            "Collect synthetic demand market partner channel economics margin regulation "
            "compliance competition operations and service evidence for the mission."
        ),
    )


def _recommendation_for_score(score: int) -> ResearchRecommendation:
    if score >= 2:
        return ResearchRecommendation.ENTER_WITH_CONDITIONS
    if score >= 0:
        return ResearchRecommendation.GATHER_MORE_EVIDENCE
    return ResearchRecommendation.DO_NOT_ENTER


def _source_handler(corpus: EvidenceCorpus):
    def handle(context: WorkExecutionContext) -> WorkExecutionOutcome:
        query = f"{context.mission.objective} {context.request.summary}"
        evidence = corpus.search(query)
        if not evidence:
            raise ValueError("source retrieval found no matching synthetic evidence")

        evidence_ids = tuple(item.evidence_id for item in evidence)
        product_id = f"wp:{context.request.work_id}"
        result = WorkResultPayload(
            work_id=context.request.work_id,
            work_product_id=product_id,
            summary=f"Retrieved {len(evidence)} synthetic evidence items for downstream analysis.",
            evidence_ids=evidence_ids,
            metadata={
                "artifact_type": "retrieval",
                "evidence": [item.model_dump(mode="json") for item in evidence],
            },
        )
        next_request = WorkRequestPayload(
            work_id=ANALYSIS_WORK_ID,
            requested_capability=ANALYSIS_CAPABILITY,
            summary="Analyze the retrieved evidence and produce an evidence-linked recommendation.",
            input_work_product_ids=(product_id,),
        )
        return WorkExecutionOutcome(result=result, next_request=next_request)

    return handle


def _analysis_handler(context: WorkExecutionContext) -> WorkExecutionOutcome:
    if len(context.input_products) != 1:
        raise ValueError("analysis requires exactly one retrieval work product")
    source = context.input_products[0]
    if source.metadata.get("artifact_type") != "retrieval":
        raise ValueError("analysis input is not a retrieval artifact")

    raw_evidence = source.metadata.get("evidence")
    if not isinstance(raw_evidence, list):
        raise ValueError("retrieval artifact does not contain evidence records")
    evidence = tuple(EvidenceItem.model_validate(item) for item in raw_evidence)
    evidence_ids = tuple(item.evidence_id for item in evidence)
    if evidence_ids != source.evidence_ids:
        raise ValueError("retrieval evidence records do not match declared evidence_ids")

    positive_ids = tuple(
        item.evidence_id for item in evidence if item.signal is EvidenceSignal.POSITIVE
    )
    risk_ids = tuple(item.evidence_id for item in evidence if item.signal is EvidenceSignal.RISK)
    score = sum(item.weight for item in evidence)
    recommendation = _recommendation_for_score(score)

    claims = (
        ResearchClaim(
            claim_id="CLM-001",
            text=(
                f"The supplied evidence contains {len(positive_ids)} positive market-entry "
                "signals spanning demand, channel readiness, and modeled economics."
            ),
            evidence_ids=positive_ids,
        ),
        ResearchClaim(
            claim_id="CLM-002",
            text=(
                f"The supplied evidence contains {len(risk_ids)} material entry constraints "
                "covering compliance, competition, and service coverage."
            ),
            evidence_ids=risk_ids,
        ),
        ResearchClaim(
            claim_id="CLM-003",
            text=(
                f"The deterministic weighted evidence score is {score}, which maps to the "
                f"recommendation {recommendation.value}."
            ),
            evidence_ids=evidence_ids,
        ),
    )
    artifact = AnalysisArtifact(
        recommendation=recommendation,
        signal_score=score,
        claims=claims,
        positive_evidence_ids=positive_ids,
        risk_evidence_ids=risk_ids,
    )
    product_id = f"wp:{context.request.work_id}"
    result = WorkResultPayload(
        work_id=context.request.work_id,
        work_product_id=product_id,
        summary=(
            f"Deterministic analysis recommends {recommendation.value} with evidence score {score}."
        ),
        evidence_ids=evidence_ids,
        metadata={
            "artifact_type": "analysis",
            "artifact": artifact.model_dump(mode="json"),
        },
    )
    next_request = WorkRequestPayload(
        work_id=VERIFICATION_WORK_ID,
        requested_capability=VERIFICATION_CAPABILITY,
        summary="Verify the analysis claims, evidence references, score, and recommendation.",
        input_work_product_ids=(source.work_product_id, product_id),
    )
    return WorkExecutionOutcome(result=result, next_request=next_request)


def _verification_handler(context: WorkExecutionContext) -> WorkExecutionOutcome:
    if len(context.input_products) != 2:
        raise ValueError("verification requires retrieval and analysis work products")
    products = {product.metadata.get("artifact_type"): product for product in context.input_products}
    source = products.get("retrieval")
    analysis_result = products.get("analysis")
    if source is None or analysis_result is None:
        raise ValueError("verification inputs must contain retrieval and analysis artifacts")

    raw_evidence = source.metadata.get("evidence")
    if not isinstance(raw_evidence, list):
        raise ValueError("retrieval artifact does not contain evidence records")
    evidence = tuple(EvidenceItem.model_validate(item) for item in raw_evidence)
    source_ids = tuple(item.evidence_id for item in evidence)
    source_id_set = set(source_ids)

    raw_analysis = analysis_result.metadata.get("artifact")
    analysis = AnalysisArtifact.model_validate(raw_analysis)
    issues: list[str] = []

    for claim in analysis.claims:
        if not set(claim.evidence_ids).issubset(source_id_set):
            issues.append(f"claim {claim.claim_id} references evidence absent from retrieval")

    expected_score = sum(item.weight for item in evidence)
    if analysis.signal_score != expected_score:
        issues.append("analysis signal score does not equal the deterministic evidence score")

    expected_recommendation = _recommendation_for_score(expected_score)
    if analysis.recommendation is not expected_recommendation:
        issues.append("analysis recommendation does not match the deterministic score rule")

    if set(analysis_result.evidence_ids) != source_id_set:
        issues.append("analysis result does not declare the complete retrieved evidence set")

    status = (
        VerificationStatus.REQUIRES_REVISION if issues else VerificationStatus.VERIFIED
    )
    artifact = VerificationArtifact(
        status=status,
        checked_claim_ids=tuple(claim.claim_id for claim in analysis.claims),
        verified_evidence_ids=source_ids if not issues else (),
        issues=tuple(issues),
    )
    product_id = f"wp:{context.request.work_id}"
    result = WorkResultPayload(
        work_id=context.request.work_id,
        work_product_id=product_id,
        summary=(
            "Verification passed with no deterministic evidence issues."
            if status is VerificationStatus.VERIFIED
            else f"Verification requires revision: {'; '.join(issues)}"
        ),
        evidence_ids=source_ids,
        metadata={
            "artifact_type": "verification",
            "artifact": artifact.model_dump(mode="json"),
        },
    )

    next_request = None
    if status is VerificationStatus.VERIFIED:
        next_request = WorkRequestPayload(
            work_id=SYNTHESIS_WORK_ID,
            requested_capability=SYNTHESIS_CAPABILITY,
            summary="Synthesize a final brief only from the verified upstream work products.",
            input_work_product_ids=(
                source.work_product_id,
                analysis_result.work_product_id,
                product_id,
            ),
        )
    return WorkExecutionOutcome(result=result, next_request=next_request)


def _synthesis_handler(context: WorkExecutionContext) -> WorkExecutionOutcome:
    if len(context.input_products) != 3:
        raise ValueError("synthesis requires retrieval, analysis, and verification products")
    products = {product.metadata.get("artifact_type"): product for product in context.input_products}
    source = products.get("retrieval")
    analysis_result = products.get("analysis")
    verification_result = products.get("verification")
    if source is None or analysis_result is None or verification_result is None:
        raise ValueError("synthesis inputs are incomplete")

    analysis = AnalysisArtifact.model_validate(analysis_result.metadata.get("artifact"))
    verification = VerificationArtifact.model_validate(
        verification_result.metadata.get("artifact")
    )
    if verification.status is not VerificationStatus.VERIFIED:
        raise ValueError("final publication is blocked until verification passes")

    evidence_ids = verification.verified_evidence_ids
    brief = (
        "Recommendation: enter the fictional Borealis market with conditions. "
        f"The deterministic evidence score is {analysis.signal_score}. Positive signals "
        f"are supported by {', '.join(analysis.positive_evidence_ids)}; material risks are "
        f"supported by {', '.join(analysis.risk_evidence_ids)}. Proceed only with explicit "
        "plans for certification, incumbent competition, and service coverage. The skeptic "
        "verified every published claim against the synthetic evidence set."
    )
    artifact = FinalBriefArtifact(
        title="Asteria Robotics — Borealis Market Entry Brief",
        recommendation=analysis.recommendation,
        verification_status=verification.status,
        brief=brief,
        evidence_ids=evidence_ids,
    )
    result = WorkResultPayload(
        work_id=context.request.work_id,
        work_product_id=FINAL_WORK_PRODUCT_ID,
        summary=brief,
        evidence_ids=evidence_ids,
        metadata={
            "artifact_type": "final_brief",
            "artifact": artifact.model_dump(mode="json"),
        },
    )
    return WorkExecutionOutcome(result=result)


def build_research_demo_system() -> ResearchDemoSystem:
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
            handlers={SOURCE_CAPABILITY: _source_handler(corpus)},
        ),
        PeerAgent(
            agent_id=ANALYST_AGENT_ID,
            mission=mission,
            registry=registry,
            bus=bus,
            handlers={ANALYSIS_CAPABILITY: _analysis_handler},
        ),
        PeerAgent(
            agent_id=SKEPTIC_AGENT_ID,
            mission=mission,
            registry=registry,
            bus=bus,
            handlers={VERIFICATION_CAPABILITY: _verification_handler},
        ),
        PeerAgent(
            agent_id=SYNTHESIZER_AGENT_ID,
            mission=mission,
            registry=registry,
            bus=bus,
            handlers={SYNTHESIS_CAPABILITY: _synthesis_handler},
        ),
    )
    return ResearchDemoSystem(
        mission=mission,
        corpus=corpus,
        registry=registry,
        bus=bus,
        runtime=PeerNetworkRuntime(peers),
    )


def run_deterministic_research_demo(max_cycles: int = 8) -> WorkResultPayload:
    """Run the fictional mission mechanically until a verified final brief exists.

    The Synthesizer peer acts only as the initial requester because it needs the final
    brief. Every work request is broadcast; recipients still self-select locally.
    The loop never chooses a worker or assigns a role.
    """

    if max_cycles < 1:
        raise ValueError("max_cycles must be at least 1")

    system = build_research_demo_system()
    initiator = system.runtime.get_peer(SYNTHESIZER_AGENT_ID)
    initiator.broadcast_mission()

    # Give all peers time to observe the mission and propagate their own role claims.
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

    raise RuntimeError("deterministic research mission did not produce a final brief")
