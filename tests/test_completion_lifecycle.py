"""Regression tests for durable peer completion during multi-peer missions."""

from peer_coordination.models import PeerStatus
from peer_coordination.research import (
    FINAL_WORK_PRODUCT_ID,
    SOURCE_AGENT_ID,
    SYNTHESIZER_AGENT_ID,
    build_research_demo_system,
    make_initial_research_request,
)


def test_completed_peer_releases_role_without_reentering_mission() -> None:
    system = build_research_demo_system()
    initiator = system.runtime.get_peer(SYNTHESIZER_AGENT_ID)
    initiator.broadcast_mission()
    system.runtime.run_round()
    system.runtime.run_round()
    initiator.broadcast_work_request(make_initial_research_request())

    for _ in range(5):
        system.runtime.run_round()
        system.runtime.run_execution_round()

    source = system.runtime.get_peer(SOURCE_AGENT_ID)
    synthesizer = system.runtime.get_peer(SYNTHESIZER_AGENT_ID)

    assert FINAL_WORK_PRODUCT_ID in synthesizer.work_products
    assert source.state.status is PeerStatus.COMPLETED
    assert source.state.claimed_role is None
    assert source.state.completed_work_ids == ("work-source-001",)
