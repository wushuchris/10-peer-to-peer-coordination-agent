"""Tests for the presentation-safe Gradio demo layer."""

from __future__ import annotations

import importlib

from peer_coordination.demo import (
    DemoMode,
    architecture_comparison_rows,
    llm_governance_rows,
    llm_runtime_status,
    run_demo,
)
from peer_coordination.research import FINAL_WORK_PRODUCT_ID


def test_deterministic_demo_completes_verified_market_entry_mission() -> None:
    snapshot = run_demo(DemoMode.DETERMINISTIC)

    assert snapshot.completed is True
    assert snapshot.recommendation == "enter_with_conditions"
    assert snapshot.verification_status == "verified"
    assert snapshot.final_brief
    assert snapshot.final_evidence_ids == (
        "EV-001",
        "EV-002",
        "EV-003",
        "EV-004",
        "EV-005",
        "EV-006",
    )
    assert snapshot.error is None


def test_demo_exposes_peer_state_work_products_and_typed_protocol_trace() -> None:
    snapshot = run_demo()

    assert len(snapshot.peers) == 4
    assert all(row.known_peers == 3 for row in snapshot.peers)
    assert any(row.work_product_id == FINAL_WORK_PRODUCT_ID for row in snapshot.work_products)

    message_types = {row.message_type for row in snapshot.messages}
    assert "mission_announcement" in message_types
    assert "role_claim" in message_types
    assert "work_request" in message_types
    assert "work_result" in message_types
    assert "role_release" in message_types
    assert any(row.correlation_id != "—" for row in snapshot.messages)


def test_business_conversation_is_traceable_to_real_protocol_events() -> None:
    snapshot = run_demo()

    assert snapshot.conversation
    assert len(snapshot.conversation) == len(snapshot.messages)
    messages_by_id = {row.message_id: row for row in snapshot.messages}

    for conversation in snapshot.conversation:
        source = messages_by_id[conversation.message_id]
        assert conversation.protocol_event == source.message_type
        assert conversation.correlation_id == source.correlation_id
        assert conversation.statement

    assert snapshot.conversation[0].protocol_event == "mission_announcement"
    assert "Asteria Robotics" in snapshot.conversation[0].statement
    assert any(
        row.protocol_event == "work_request" and "centrally" in row.statement
        for row in snapshot.conversation
    )
    assert any(row.speaker == "Analyst" for row in snapshot.conversation)
    assert any(row.speaker == "Skeptic / Verifier" for row in snapshot.conversation)


def test_architecture_comparison_rows_include_both_architectures_per_scenario() -> None:
    rows, observations = architecture_comparison_rows()

    assert len(rows) == 6
    assert {row[1] for row in rows} == {"peer_to_peer", "centralized"}
    assert {row[0] for row in rows} == {
        "happy_path",
        "coordinator_outage",
        "unavailable_analyst",
    }
    assert observations


def test_llm_governance_rows_surface_all_adversarial_scenarios_and_limitations() -> None:
    rows, limitations = llm_governance_rows()

    assert len(rows) == 11
    assert all(row[5] is True for row in rows)
    assert any(row[0] == "analyst_contradictory_narrative" for row in rows)
    assert any("semantically entailed" in limitation for limitation in limitations)


def test_llm_runtime_status_never_requires_live_call(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("MODEL_ID", raising=False)
    status = llm_runtime_status()
    assert status.startswith("Not configured")

    monkeypatch.setenv("HF_TOKEN", "hf_test_placeholder")
    monkeypatch.setenv("MODEL_ID", "example/model")
    status = llm_runtime_status()
    assert status == "Configured for model: example/model"
    assert "hf_test_placeholder" not in status


def test_app_imports_and_builds_gradio_blocks() -> None:
    app = importlib.import_module("app")
    assert app.demo is not None


def test_business_first_gradio_outputs_include_executive_story_and_real_event_transcript() -> None:
    app = importlib.import_module("app")
    outputs = app._mission_outputs("Deterministic")

    assert len(outputs) == 9
    status, brief, business_summary, conversation = outputs[:4]
    assert "Mission completed" in status
    assert "Final market-entry brief" in brief
    assert "Executive summary" in business_summary
    assert "Enter With Conditions" in business_summary
    assert "No central AI manager" in business_summary
    assert "This is not a scripted chat" in conversation
    assert "mission_announcement" in conversation
    assert "role_claim" in conversation


def test_business_layout_uses_narrow_vertical_reading_path() -> None:
    app = importlib.import_module("app")

    assert "max-width: 980px" in app.APP_CSS
    assert ".agent-grid { display: grid; grid-template-columns: 1fr;" in app.APP_CSS
    assert ".process-flow { display: grid; grid-template-columns: 1fr;" in app.APP_CSS
    assert "font-size: 1.08rem" in app.APP_CSS


def test_streaming_mission_yields_visible_protocol_progress_before_completion(monkeypatch) -> None:
    app = importlib.import_module("app")
    monkeypatch.setattr(app, "_playback_delay_for_event", lambda _event: 0)

    frames = list(app._stream_mission("Deterministic"))

    assert len(frames) > 3
    assert all(len(frame) == 10 for frame in frames)
    assert "RUNNING" in frames[0][0]
    assert any("mission_announcement" in frame[0] for frame in frames[1:-1])
    assert any("role_claim" in frame[0] for frame in frames[1:-1])
    assert "COMPLETE" in frames[-1][0]
    assert "Mission completed" in frames[-1][1]
    assert "Executive summary" in frames[-1][3]


def test_completed_activity_retains_full_scrollable_protocol_transcript() -> None:
    app = importlib.import_module("app")
    snapshot = run_demo(DemoMode.DETERMINISTIC)
    completed = app._activity_html(snapshot, len(snapshot.conversation), complete=True)

    assert "overflow-y: auto" in app.APP_CSS
    assert "Scroll inside the transcript" in completed
    assert snapshot.conversation[0].message_id in completed
    assert snapshot.conversation[-1].message_id in completed
    assert completed.count('class="activity-line"') == len(snapshot.conversation)
