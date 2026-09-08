from peer_coordination.challenge import ChallengeCapablePeerAgent
from peer_coordination.challenge_demo import (
    CHALLENGE_ID,
    REVISED_ANALYSIS_PRODUCT_ID,
    VERIFICATION_RECHECK_WORK_ID,
    build_challenge_research_demo_system,
    run_challenge_research_demo,
)
from peer_coordination.models import PeerStatus
from peer_coordination.research import (
    FINAL_WORK_PRODUCT_ID,
    SKEPTIC_AGENT_ID,
    SYNTHESIZER_AGENT_ID,
    ANALYST_AGENT_ID,
    VerificationArtifact,
    VerificationStatus,
    make_initial_research_request,
)


def _prime(system):
    initiator = system.runtime.get_peer(SYNTHESIZER_AGENT_ID)
    initiator.broadcast_mission()
    system.runtime.run_round()
    system.runtime.run_round()
    initiator.broadcast_work_request(make_initial_research_request())


def _run_cycles(system, count=12):
    for _ in range(count):
        system.runtime.run_round()
        system.runtime.run_execution_round()


def test_valid_revision_resolves_challenge_and_allows_final_publication():
    system = build_challenge_research_demo_system("fix")
    _prime(system)
    _run_cycles(system)

    analyst = system.runtime.get_peer(ANALYST_AGENT_ID)
    skeptic = system.runtime.get_peer(SKEPTIC_AGENT_ID)
    synthesizer = system.runtime.get_peer(SYNTHESIZER_AGENT_ID)

    assert isinstance(analyst, ChallengeCapablePeerAgent)
    assert isinstance(skeptic, ChallengeCapablePeerAgent)
    assert CHALLENGE_ID in analyst.challenge_ledger.responded_challenge_ids
    assert CHALLENGE_ID in skeptic.challenge_ledger.resolved_challenge_ids
    assert CHALLENGE_ID not in skeptic.challenge_ledger.disputed_challenge_ids

    assert REVISED_ANALYSIS_PRODUCT_ID in skeptic.work_products
    assert f"wp:{VERIFICATION_RECHECK_WORK_ID}" in skeptic.work_products
    recheck = skeptic.work_products[f"wp:{VERIFICATION_RECHECK_WORK_ID}"]
    artifact = VerificationArtifact.model_validate(recheck.metadata.get("artifact"))
    assert artifact.status is VerificationStatus.VERIFIED

    final = synthesizer.work_products.get(FINAL_WORK_PRODUCT_ID)
    assert final is not None
    assert REVISED_ANALYSIS_PRODUCT_ID in synthesizer.work_products


def test_false_revised_label_is_rejected_by_independent_skeptic_evaluation():
    system = build_challenge_research_demo_system("false_fix")
    _prime(system)
    _run_cycles(system)

    skeptic = system.runtime.get_peer(SKEPTIC_AGENT_ID)
    synthesizer = system.runtime.get_peer(SYNTHESIZER_AGENT_ID)

    assert isinstance(skeptic, ChallengeCapablePeerAgent)
    assert CHALLENGE_ID in skeptic.challenge_ledger.disputed_challenge_ids
    assert CHALLENGE_ID not in skeptic.challenge_ledger.resolved_challenge_ids
    assert skeptic.state.status is PeerStatus.ESCALATED
    assert FINAL_WORK_PRODUCT_ID not in synthesizer.work_products


def test_disputed_challenge_fails_closed_and_blocks_publication():
    system = build_challenge_research_demo_system("dispute")
    _prime(system)
    _run_cycles(system)

    analyst = system.runtime.get_peer(ANALYST_AGENT_ID)
    skeptic = system.runtime.get_peer(SKEPTIC_AGENT_ID)
    synthesizer = system.runtime.get_peer(SYNTHESIZER_AGENT_ID)

    assert isinstance(analyst, ChallengeCapablePeerAgent)
    assert isinstance(skeptic, ChallengeCapablePeerAgent)
    assert CHALLENGE_ID in analyst.challenge_ledger.responded_challenge_ids
    assert CHALLENGE_ID in skeptic.challenge_ledger.disputed_challenge_ids
    assert skeptic.state.status is PeerStatus.ESCALATED
    assert FINAL_WORK_PRODUCT_ID not in synthesizer.work_products


def test_challenge_demo_helper_returns_only_after_verified_revision_path():
    final = run_challenge_research_demo()

    assert final.work_product_id == FINAL_WORK_PRODUCT_ID
    assert final.metadata.get("artifact_type") == "final_brief"
