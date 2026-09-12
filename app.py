"""Gradio demo for Agent 10 — Peer-to-Peer Coordination Agent."""

from __future__ import annotations

import html
import sys
import time
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from peer_coordination.demo import (  # noqa: E402
    DemoMode,
    DemoSnapshot,
    architecture_comparison_rows,
    llm_governance_rows,
    llm_runtime_status,
    run_demo,
)
from peer_coordination.llm import ModelConfigurationError  # noqa: E402


PEER_HEADERS = [
    "Peer",
    "Display name",
    "Eligible role",
    "Capability",
    "Final status",
    "Accepted work",
    "Completed work",
    "Known peers",
]
EVIDENCE_HEADERS = ["Evidence", "Title", "Signal", "Weight", "Finding"]
WORK_HEADERS = ["Work product", "Work ID", "Artifact type", "Evidence", "Summary"]
MESSAGE_HEADERS = [
    "Message ID",
    "Type",
    "Correlation",
    "Sender",
    "Recipient",
    "Delivered to",
    "Status",
]
ARCHITECTURE_HEADERS = [
    "Scenario",
    "Architecture",
    "Completed",
    "Coordination steps",
    "Protocol messages",
    "Deliveries",
    "Work executions",
    "Central coordinator required",
    "Failure code",
]
LLM_EVAL_HEADERS = [
    "Scenario",
    "Expected completion",
    "Completed",
    "Escalated peer(s)",
    "Model calls",
    "Governance preserved",
    "Semantic contradiction blocked",
    "Finding",
]

# Human-readable playback delay only. The coordination protocol itself is not slowed.
PLAYBACK_DELAY_SECONDS = 0.24

APP_CSS = """
.gradio-container {
    max-width: 980px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
.hero-card {
    border: 1px solid rgba(148, 163, 184, .28);
    border-radius: 22px;
    padding: 30px 32px;
    margin-bottom: 18px;
    background: linear-gradient(135deg, rgba(79,70,229,.18), rgba(14,165,233,.08));
}
.hero-card h1 { margin: 6px 0 12px; font-size: 2.15rem; line-height: 1.14; }
.hero-card p { font-size: 1.08rem; line-height: 1.6; max-width: 820px; margin: 0 0 14px; }
.eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: .78rem; font-weight: 700; opacity: .8; }
.pill-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
.pill { border: 1px solid rgba(148,163,184,.35); border-radius: 999px; padding: 7px 11px; font-size: .9rem; }
.business-case {
    border: 1px solid rgba(148,163,184,.28);
    border-radius: 18px;
    padding: 20px 22px;
    margin: 12px 0 20px;
    background: rgba(148,163,184,.05);
}
.business-case h3 { margin: 0 0 8px; font-size: 1.18rem; }
.business-case p { margin: 0; font-size: 1.02rem; line-height: 1.58; }
.agent-grid { display: grid; grid-template-columns: 1fr; gap: 10px; margin: 12px 0 22px; }
.agent-card {
    display: grid;
    grid-template-columns: 54px 1fr;
    gap: 14px;
    align-items: start;
    border: 1px solid rgba(148,163,184,.28);
    border-radius: 16px;
    padding: 16px 18px;
    background: rgba(148,163,184,.05);
}
.agent-card .icon { font-size: 1.7rem; line-height: 1.2; }
.agent-card h3 { margin: 0 0 5px; font-size: 1.08rem; }
.agent-card p { margin: 0; font-size: .98rem; line-height: 1.5; opacity: .92; }
.process-flow { display: grid; grid-template-columns: 1fr; gap: 9px; margin: 10px 0 22px; }
.process-step {
    border-left: 4px solid rgba(79,70,229,.65);
    border-radius: 0 14px 14px 0;
    padding: 13px 16px;
    background: rgba(79,70,229,.055);
    font-size: .98rem;
    line-height: 1.5;
}
.process-step strong { display:block; margin-bottom:4px; font-size: 1.02rem; }
.result-card { border: 1px solid rgba(148,163,184,.25); border-radius: 16px; padding: 12px 16px; font-size: 1rem; line-height: 1.55; }
.transcript-note { border-left: 4px solid #f97316; padding: 11px 14px; margin-bottom: 12px; background: rgba(249,115,22,.07); border-radius: 8px; }
.activity-card {
    border: 1px solid rgba(59,130,246,.28);
    border-radius: 18px;
    padding: 18px 20px;
    margin: 14px 0 18px;
    background: rgba(59,130,246,.055);
}
.activity-head { display:flex; align-items:center; gap:11px; margin-bottom:8px; }
.activity-spinner {
    width: 22px;
    height: 22px;
    border: 3px solid rgba(59,130,246,.22);
    border-top-color: rgba(59,130,246,.95);
    border-radius: 50%;
    animation: peer-spin .85s linear infinite;
    flex: 0 0 auto;
}
.activity-badge {
    display:inline-block;
    border:1px solid rgba(59,130,246,.45);
    border-radius:999px;
    padding:3px 8px;
    font-size:.76rem;
    font-weight:800;
    letter-spacing:.08em;
}
.activity-title { font-size:1.08rem; font-weight:750; }
.activity-sub { font-size:.93rem; opacity:.82; margin-bottom:10px; }
.activity-track { height:8px; background:rgba(148,163,184,.2); border-radius:999px; overflow:hidden; margin:10px 0 14px; }
.activity-fill { height:100%; background:linear-gradient(90deg, rgba(59,130,246,.9), rgba(79,70,229,.9)); border-radius:999px; transition:width .18s ease; }
.activity-track.indeterminate .activity-fill { width:32%; animation: peer-slide 1.15s ease-in-out infinite; }
.activity-feed { max-height: 320px; overflow-y: auto; padding-right: 6px; }
.activity-line { border-top:1px solid rgba(148,163,184,.18); padding:9px 0; font-size:.94rem; line-height:1.45; }
.activity-line:first-child { border-top:0; }
.activity-line small { display:block; opacity:.7; margin-top:2px; }
.activity-complete { border-color:rgba(34,197,94,.32); background:rgba(34,197,94,.055); }
@keyframes peer-spin { to { transform: rotate(360deg); } }
@keyframes peer-slide { 0% { transform:translateX(-110%); } 50% { transform:translateX(215%); } 100% { transform:translateX(-110%); } }
@media (max-width: 620px) {
  .gradio-container { max-width: 100% !important; }
  .hero-card { padding: 24px 20px; }
  .agent-card { grid-template-columns: 42px 1fr; }
}
"""


PEER_ICONS = {
    "Source Finder": "🔎",
    "Analyst": "📊",
    "Skeptic / Verifier": "🛡️",
    "Synthesizer": "📝",
}


def _friendly_recommendation(value: str | None) -> str:
    if not value:
        return "No recommendation published"
    return value.replace("_", " ").title()


def _mode_from_label(mode_label: str) -> DemoMode:
    return (
        DemoMode.LLM_ASSISTED
        if mode_label == "LLM-assisted"
        else DemoMode.DETERMINISTIC
    )


def _business_summary(snapshot: DemoSnapshot) -> str:
    positives = [row.title for row in snapshot.evidence if row.signal == "positive"]
    risks = [row.title for row in snapshot.evidence if row.signal == "risk"]
    score = sum(row.weight for row in snapshot.evidence)

    if not snapshot.completed:
        return (
            "### Executive summary\n"
            "**No business recommendation was published.** The peer network stopped safely "
            "because one of its controls failed or a required peer escalated. This is a feature "
            "of the design: the system prefers no answer over an unverified answer.\n\n"
            f"**What stopped the mission:** {snapshot.error or 'bounded mission limit reached'}"
        )

    return (
        "### Executive summary\n"
        f"**Decision:** {_friendly_recommendation(snapshot.recommendation)}  \n"
        f"**Evidence score:** {score:+d}  \n"
        f"**Independent verification:** {snapshot.verification_status or 'not available'}\n\n"
        "**What supports the decision:** " + ", ".join(positives) + ".  \n"
        "**What management still has to address:** " + ", ".join(risks) + ".\n\n"
        "**Why the coordination design matters:** No central AI manager assigned the work. "
        "Specialized peers discovered the mission, claimed roles, exchanged typed work products, "
        "and the final brief was published only after the verification boundary passed."
    )


def _conversation_markdown(snapshot: DemoSnapshot) -> str:
    if not snapshot.conversation:
        return "Run the mission to see the peer conversation."

    lines = [
        '<div class="transcript-note"><strong>This is not a scripted chat.</strong> '
        "Every line below is a business-readable rendering of an actual typed protocol event. "
        "The underlying event name and message ID are shown for auditability.</div>"
    ]
    for row in snapshot.conversation:
        icon = PEER_ICONS.get(row.speaker, "🤖")
        lines.extend(
            [
                f"**{row.step}. {icon} {row.speaker} → {row.audience}**",
                f"> {row.statement}",
                (
                    f"`{row.protocol_event}` · message `{row.message_id}`"
                    + (
                        f" · correlation `{row.correlation_id}`"
                        if row.correlation_id != "—"
                        else ""
                    )
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _starting_activity_html(mode: DemoMode) -> str:
    mode_name = mode.value.replace("_", " ").title()
    return f"""
<div class="activity-card">
  <div class="activity-head">
    <span class="activity-spinner" aria-hidden="true"></span>
    <span class="activity-badge">RUNNING</span>
    <span class="activity-title">Peer network is starting</span>
  </div>
  <div class="activity-sub">{html.escape(mode_name)} mode · discovering the mission and preparing the typed coordination trace.</div>
  <div class="activity-track indeterminate"><div class="activity-fill"></div></div>
  <div class="activity-feed"><div class="activity-line">The network is working. Protocol events will appear here as soon as the mission snapshot is available.</div></div>
</div>
"""


def _activity_html(snapshot: DemoSnapshot, count: int, *, complete: bool = False) -> str:
    total = max(len(snapshot.conversation), 1)
    visible_count = min(max(count, 0), len(snapshot.conversation))
    percent = min(100.0, (visible_count / total) * 100)
    rows = snapshot.conversation if complete else snapshot.conversation[max(0, visible_count - 8):visible_count]

    if complete:
        title = "Peer mission complete"
        badge = "COMPLETE"
        sub = f"{len(snapshot.conversation)} protocol-derived coordination events retained. Scroll inside the transcript to review the full mission."
        spinner = ""
        card_class = "activity-card activity-complete"
    else:
        current = snapshot.conversation[visible_count - 1] if visible_count else None
        title = "Peers are coordinating"
        badge = "RUNNING"
        sub = (
            f"Event {visible_count}/{len(snapshot.conversation)} · {current.speaker} · {current.protocol_event.replace('_', ' ')}"
            if current
            else "Waiting for the first protocol event."
        )
        spinner = '<span class="activity-spinner" aria-hidden="true"></span>'
        card_class = "activity-card"

    feed_lines: list[str] = []
    for row in rows:
        icon = PEER_ICONS.get(row.speaker, "🤖")
        correlation = (
            f" · correlation {html.escape(row.correlation_id)}"
            if row.correlation_id != "—"
            else ""
        )
        feed_lines.append(
            '<div class="activity-line">'
            f"<strong>{row.step}. {icon} {html.escape(row.speaker)} → {html.escape(row.audience)}</strong><br>"
            f"{html.escape(row.statement)}"
            f"<small>{html.escape(row.protocol_event)} · message {html.escape(row.message_id)}{correlation}</small>"
            "</div>"
        )

    feed = "".join(feed_lines) or '<div class="activity-line">Waiting for the first protocol event.</div>'
    return f"""
<div class="{card_class}">
  <div class="activity-head">
    {spinner}
    <span class="activity-badge">{badge}</span>
    <span class="activity-title">{html.escape(title)}</span>
  </div>
  <div class="activity-sub">{html.escape(sub)}</div>
  <div class="activity-track"><div class="activity-fill" style="width:{percent:.1f}%"></div></div>
  <div class="activity-feed">{feed}</div>
</div>
"""


def _playback_delay_for_event(protocol_event: str, base: float = PLAYBACK_DELAY_SECONDS) -> float:
    if base <= 0:
        return 0
    if protocol_event == "mission_announcement":
        return base * 2.2
    if protocol_event in {"role_claim", "work_request"}:
        return base * 1.45
    if protocol_event in {"work_result", "challenge", "revision"}:
        return base * 1.8
    if protocol_event == "role_release":
        return base * 1.25
    return base


def _render_snapshot_outputs(snapshot: DemoSnapshot):
    if snapshot.completed:
        status = (
            "### Mission completed ✅\n"
            f"**Mode:** {snapshot.mode.value.replace('_', ' ').title()}  \n"
            f"**Recommendation:** {snapshot.recommendation}  \n"
            f"**Verification:** {snapshot.verification_status}"
        )
        brief = (
            "### Final market-entry brief\n"
            f"{snapshot.final_brief}\n\n"
            "**Published evidence:** " + ", ".join(snapshot.final_evidence_ids)
        )
    else:
        status = (
            "### Mission did not complete ⚠️\n"
            f"**Mode:** {snapshot.mode.value.replace('_', ' ').title()}"
        )
        brief = snapshot.error or "The system failed closed before publication."

    peer_rows = [
        [
            row.agent_id,
            row.display_name,
            row.role,
            row.capability,
            row.final_status,
            row.accepted_work,
            row.completed_work,
            row.known_peers,
        ]
        for row in snapshot.peers
    ]
    evidence_rows = [
        [row.evidence_id, row.title, row.signal, row.weight, row.finding]
        for row in snapshot.evidence
    ]
    work_rows = [
        [
            row.work_product_id,
            row.work_id,
            row.artifact_type,
            row.evidence_ids,
            row.summary,
        ]
        for row in snapshot.work_products
    ]
    message_rows = [
        [
            row.message_id,
            row.message_type,
            row.correlation_id,
            row.sender,
            row.recipient,
            row.delivered_to,
            row.status,
        ]
        for row in snapshot.messages
    ]

    return (
        status,
        brief,
        _business_summary(snapshot),
        _conversation_markdown(snapshot),
        peer_rows,
        evidence_rows,
        work_rows,
        message_rows,
        snapshot.model_dump(mode="json"),
    )


def _error_outputs(message: str, mode: DemoMode):
    return (
        "### Mission failed closed",
        message,
        "### Executive summary\nNo recommendation was published because the mission failed closed.",
        "No protocol-derived conversation is available for this failed startup path.",
        [],
        [],
        [],
        [],
        {"error": message, "mode": mode.value},
    )


def _mission_outputs(mode_label: str):
    """Static rendering path retained for tests and non-streaming callers."""
    mode = _mode_from_label(mode_label)
    try:
        snapshot = run_demo(mode)
    except ModelConfigurationError as exc:
        return _error_outputs(str(exc), mode)
    except Exception as exc:
        return _error_outputs(f"{type(exc).__name__}: {exc}", mode)
    return _render_snapshot_outputs(snapshot)


def _stream_mission(mode_label: str):
    """Yield one visible UI frame per real protocol-derived conversation event."""
    mode = _mode_from_label(mode_label)
    idle_outputs = (
        "### Mission running…",
        "The verified final brief will appear after the peer network completes.",
        "### Executive summary\nThe peer network is still coordinating.",
        "Live coordination is shown above. The full trace will remain available after completion.",
        [],
        [],
        [],
        [],
        {"mode": mode.value, "status": "running"},
    )
    yield (_starting_activity_html(mode), *idle_outputs)

    try:
        snapshot = run_demo(mode)
    except ModelConfigurationError as exc:
        error = _error_outputs(str(exc), mode)
        yield (
            '<div class="activity-card"><div class="activity-head"><span class="activity-badge">STOPPED</span><span class="activity-title">Mission not started</span></div>'
            f'<div class="activity-sub">{html.escape(str(exc))}</div></div>',
            *error,
        )
        return
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        error = _error_outputs(message, mode)
        yield (
            '<div class="activity-card"><div class="activity-head"><span class="activity-badge">STOPPED</span><span class="activity-title">Mission failed closed</span></div>'
            f'<div class="activity-sub">{html.escape(message)}</div></div>',
            *error,
        )
        return

    for index, row in enumerate(snapshot.conversation, start=1):
        yield (_activity_html(snapshot, index), *idle_outputs)
        time.sleep(_playback_delay_for_event(row.protocol_event))

    yield (
        _activity_html(snapshot, len(snapshot.conversation), complete=True),
        *_render_snapshot_outputs(snapshot),
    )


def _architecture_outputs():
    rows, observations = architecture_comparison_rows()
    notes = "### What the comparison shows\n" + "\n".join(
        f"- {observation}" for observation in observations
    )
    return rows, notes


def _llm_evaluation_outputs():
    rows, limitations = llm_governance_rows()
    notes = "### Known limitations\n" + "\n".join(
        f"- {limitation}" for limitation in limitations
    )
    return rows, notes


with gr.Blocks(
    title="Agent 10 — Peer-to-Peer Coordination",
    analytics_enabled=False,
) as demo:
    gr.HTML(
        """
<div class="hero-card">
  <div class="eyebrow">Agent 10 · Peer-to-Peer Coordination</div>
  <h1>Can an AI research team coordinate without a central boss?</h1>
  <p><strong>Business decision:</strong> Should fictional Asteria Robotics enter the Borealis market? Four specialized AI peers must gather evidence, analyze it, challenge weak reasoning, and publish a decision brief—without a manager agent assigning each step.</p>
  <div class="pill-row">
    <span class="pill">🤝 No central AI boss</span>
    <span class="pill">🔎 Evidence-first</span>
    <span class="pill">🛡️ Independent verification</span>
    <span class="pill">🧾 Auditable messages</span>
    <span class="pill">🔒 Fail closed</span>
  </div>
</div>
"""
    )

    gr.HTML(
        """
<div class="business-case">
  <h3>Why would a business care?</h3>
  <p>A single AI can research, analyze, verify, and summarize its own work—but that concentrates authority and can hide mistakes. This demo tests a different operating model: specialized peers self-select bounded responsibilities, exchange typed work products, independently challenge one another, and publish only after the required verification boundary passes.</p>
</div>
"""
    )

    mode = gr.Radio(
        choices=["Deterministic", "LLM-assisted"],
        value="Deterministic",
        label="Execution mode",
    )
    gr.Markdown(
        "**Live LLM runtime:** " + llm_runtime_status()
        + "\n\n**Engineering principle:** *The runtime turns the clock; the peers make the decisions.*"
    )

    run_button = gr.Button("Run the peer research mission", variant="primary")
    live_activity = gr.HTML(
        '<div class="activity-card"><div class="activity-title">Live peer activity</div><div class="activity-sub">Run the mission to watch real protocol-derived coordination events appear here.</div></div>'
    )

    mission_status = gr.Markdown(
        "Run the mission to see the peer network coordinate.",
        elem_classes=["result-card"],
    )
    final_brief = gr.Markdown(
        "The verified final brief will appear here.",
        elem_classes=["result-card"],
    )

    with gr.Tabs():
        with gr.Tab("Business Story"):
            gr.Markdown(
                "## Meet the peer team\n"
                "Each peer has a bounded business responsibility. No supervisor agent assigns the work; the peers coordinate through the engineered protocol."
            )
            gr.HTML(
                """
<div class="agent-grid">
  <div class="agent-card"><div class="icon">🔎</div><div><h3>Source Finder</h3><p><strong>Find the approved facts.</strong> Has exclusive access to the synthetic source corpus and packages evidence for the rest of the team.</p></div></div>
  <div class="agent-card"><div class="icon">📊</div><div><h3>Analyst</h3><p><strong>Turn evidence into business insight.</strong> Interprets the evidence and produces structured market-entry analysis without becoming the source of record.</p></div></div>
  <div class="agent-card"><div class="icon">🛡️</div><div><h3>Skeptic / Verifier</h3><p><strong>Challenge weak reasoning.</strong> Independently checks the analysis and can force revision instead of allowing unsupported work to flow downstream.</p></div></div>
  <div class="agent-card"><div class="icon">📝</div><div><h3>Synthesizer</h3><p><strong>Prepare the decision brief.</strong> Publishes only after the verification boundary passes; it cannot bypass failed verification.</p></div></div>
</div>
<h3>How the mission unfolds</h3>
<div class="process-flow">
  <div class="process-step"><strong>1 · Discover & claim</strong>The peers discover the shared mission and locally claim work they are authorized to perform.</div>
  <div class="process-step"><strong>2 · Exchange work</strong>Evidence and analysis move between peers as typed, validated work products rather than hidden shared context.</div>
  <div class="process-step"><strong>3 · Challenge & revise</strong>The Skeptic / Verifier independently checks the analysis and can trigger bounded revision.</div>
  <div class="process-step"><strong>4 · Publish</strong>The Synthesizer prepares the final market-entry brief only after the required verification boundary passes.</div>
</div>
"""
            )
            business_summary = gr.Markdown(
                "Run the mission to generate the executive summary.",
                elem_classes=["result-card"],
            )

        with gr.Tab("Coordination Transcript"):
            gr.Markdown(
                "## Review the complete peer coordination trace\n"
                "The readable transcript is derived from the real typed-message audit trail. It is not invented dialogue, and every line remains traceable to its protocol event and message ID."
            )
            conversation_view = gr.Markdown(
                "Run the mission to see the real protocol-derived conversation."
            )

        with gr.Tab("Decision & Evidence"):
            gr.Markdown(
                "The Source Finder has exclusive access to the synthetic corpus. Downstream peers receive evidence only through typed work products."
            )
            evidence_table = gr.Dataframe(
                headers=EVIDENCE_HEADERS,
                value=[],
                interactive=False,
                label="Synthetic evidence corpus",
            )
            work_table = gr.Dataframe(
                headers=WORK_HEADERS,
                value=[],
                interactive=False,
                label="Published work products",
            )

        with gr.Tab("Engineering Audit"):
            gr.Markdown(
                "## Engineering evidence\n"
                "Inspect independent peer state and the append-only transport audit behind the business story."
            )
            peer_table = gr.Dataframe(
                headers=PEER_HEADERS,
                value=[],
                interactive=False,
                label="Independent peer state",
            )
            message_table = gr.Dataframe(
                headers=MESSAGE_HEADERS,
                value=[],
                interactive=False,
                label="Append-only protocol transport audit",
            )

        with gr.Tab("Architecture Tradeoff"):
            gr.Markdown(
                "Compare the same fictional research work under decentralized peer coordination and an Agent-8-style centralized allocator."
            )
            compare_button = gr.Button("Run architecture comparison")
            architecture_table = gr.Dataframe(
                headers=ARCHITECTURE_HEADERS,
                value=[],
                interactive=False,
                label="P2P vs centralized metrics",
            )
            architecture_notes = gr.Markdown()

        with gr.Tab("Stress & Governance"):
            gr.Markdown(
                "These eleven offline scenarios simulate bad model behavior without making external API calls. They test whether a model can escape application-owned governance boundaries."
            )
            evaluate_button = gr.Button("Run adversarial LLM evaluation")
            llm_table = gr.Dataframe(
                headers=LLM_EVAL_HEADERS,
                value=[],
                interactive=False,
                label="Adversarial governance matrix",
            )
            llm_notes = gr.Markdown()

        with gr.Tab("Raw Snapshot"):
            gr.Markdown(
                "Machine-readable state from the most recent mission run. No secret material is included. The business conversation includes message IDs so it can be reconciled to this technical record."
            )
            raw_snapshot = gr.JSON(value={})

    run_button.click(
        fn=_stream_mission,
        inputs=[mode],
        outputs=[
            live_activity,
            mission_status,
            final_brief,
            business_summary,
            conversation_view,
            peer_table,
            evidence_table,
            work_table,
            message_table,
            raw_snapshot,
        ],
        show_progress="hidden",
    )
    compare_button.click(
        fn=_architecture_outputs,
        outputs=[architecture_table, architecture_notes],
    )
    evaluate_button.click(
        fn=_llm_evaluation_outputs,
        outputs=[llm_table, llm_notes],
    )


if __name__ == "__main__":
    demo.launch(css=APP_CSS)
