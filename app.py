"""Gradio demo for Agent 10 — Peer-to-Peer Coordination Agent."""

from __future__ import annotations

import sys
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

APP_CSS = """
.gradio-container { max-width: 1540px !important; }
.hero-card {
    border: 1px solid rgba(148, 163, 184, .28);
    border-radius: 22px;
    padding: 28px 30px;
    margin-bottom: 18px;
    background: linear-gradient(135deg, rgba(79,70,229,.18), rgba(14,165,233,.08));
}
.hero-card h1 { margin: 6px 0 10px; font-size: 2.1rem; line-height: 1.15; }
.hero-card p { font-size: 1.02rem; max-width: 1080px; margin: 0 0 14px; }
.eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: .74rem; font-weight: 700; opacity: .8; }
.pill-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
.pill { border: 1px solid rgba(148,163,184,.35); border-radius: 999px; padding: 6px 10px; font-size: .82rem; }
.agent-grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; margin: 10px 0 18px; }
.agent-card { border: 1px solid rgba(148,163,184,.28); border-radius: 16px; padding: 16px; min-height: 150px; background: rgba(148,163,184,.06); }
.agent-card .icon { font-size: 1.45rem; }
.agent-card h3 { margin: 8px 0 6px; font-size: 1rem; }
.agent-card p { margin: 0; font-size: .9rem; opacity: .9; }
.process-flow { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; margin: 8px 0 18px; }
.process-step { border: 1px solid rgba(148,163,184,.25); border-radius: 14px; padding: 14px; }
.process-step strong { display:block; margin-bottom:4px; }
.result-card { border: 1px solid rgba(148,163,184,.25); border-radius: 16px; padding: 8px 14px; }
.transcript-note { border-left: 4px solid #f97316; padding: 10px 14px; margin-bottom: 12px; background: rgba(249,115,22,.07); border-radius: 8px; }
@media (max-width: 900px) {
  .agent-grid, .process-flow { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 620px) {
  .agent-grid, .process-flow { grid-template-columns: 1fr; }
}
"""


def _friendly_recommendation(value: str | None) -> str:
    if not value:
        return "No recommendation published"
    return value.replace("_", " ").title()


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

    icons = {
        "Source Finder": "🔎",
        "Analyst": "📊",
        "Skeptic / Verifier": "🛡️",
        "Synthesizer": "📝",
    }
    lines = [
        '<div class="transcript-note"><strong>This is not a scripted chat.</strong> '
        "Every line below is a business-readable rendering of an actual typed protocol event. "
        "The underlying event name and message ID are shown for auditability.</div>"
    ]
    for row in snapshot.conversation:
        icon = icons.get(row.speaker, "🤖")
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


def _mission_outputs(mode_label: str):
    mode = (
        DemoMode.LLM_ASSISTED
        if mode_label == "LLM-assisted"
        else DemoMode.DETERMINISTIC
    )
    try:
        snapshot = run_demo(mode)
    except ModelConfigurationError as exc:
        return (
            "### Mission not started\n" + str(exc),
            "LLM-assisted mode needs the deployment runtime model settings. "
            "Choose **Deterministic** to run the complete agent system without a model provider.",
            "### Executive summary\nNo business recommendation was produced because the live model runtime is not configured.",
            "Run the mission after configuration to see the real protocol-derived conversation.",
            [],
            [],
            [],
            [],
            {"error": str(exc), "mode": mode.value},
        )
    except Exception as exc:
        return (
            "### Mission failed closed",
            f"The demo stopped safely: `{type(exc).__name__}: {exc}`",
            "### Executive summary\nNo recommendation was published because the mission failed closed.",
            "No conversation is available for this failed startup path.",
            [],
            [],
            [],
            [],
            {"error": f"{type(exc).__name__}: {exc}", "mode": mode.value},
        )

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
    css=APP_CSS,
) as demo:
    gr.HTML(
        """
<div class="hero-card">
  <div class="eyebrow">Agent 10 · Business + Engineering Demo</div>
  <h1>Should Asteria Robotics enter the Borealis market?</h1>
  <p>Watch four specialized AI peers investigate a fictional market-entry decision. Instead of one AI acting as the boss, the peers discover the mission, claim responsibilities, exchange work, challenge one another, and publish only after verification passes.</p>
  <div class="pill-row">
    <span class="pill">🤝 No central AI boss</span>
    <span class="pill">🔎 Evidence-first</span>
    <span class="pill">🛡️ Independent verification</span>
    <span class="pill">🧾 Auditable messages</span>
    <span class="pill">🔒 Fail-closed controls</span>
  </div>
</div>
"""
    )

    with gr.Row():
        mode = gr.Radio(
            choices=["Deterministic", "LLM-assisted"],
            value="Deterministic",
            label="Execution mode",
        )
        gr.Markdown(
            "**Live LLM runtime:** " + llm_runtime_status()
            + "\n\n**Engineering principle:** *The runtime turns the clock; the peers make the decisions.*"
        )

    run_button = gr.Button("Run coordinated mission", variant="primary")

    with gr.Row():
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
                "## What the business user should care about\n"
                "This view translates the engineering system into a management story: what decision was made, what evidence supports it, what risks remain, and why the control design matters."
            )
            gr.HTML(
                """
<div class="agent-grid">
  <div class="agent-card"><div class="icon">🔎</div><h3>Source Finder</h3><p><strong>Business role:</strong> Find the approved facts.<br><br>“I locate the evidence the team is allowed to use.”</p></div>
  <div class="agent-card"><div class="icon">📊</div><h3>Analyst</h3><p><strong>Business role:</strong> Turn evidence into insight.<br><br>“I explain what the evidence means for the decision.”</p></div>
  <div class="agent-card"><div class="icon">🛡️</div><h3>Skeptic / Verifier</h3><p><strong>Business role:</strong> Challenge weak reasoning.<br><br>“I check the analysis before anyone can rely on it.”</p></div>
  <div class="agent-card"><div class="icon">📝</div><h3>Synthesizer</h3><p><strong>Business role:</strong> Prepare the decision brief.<br><br>“I publish only what has passed verification.”</p></div>
</div>
<div class="process-flow">
  <div class="process-step"><strong>1 · Gather</strong>Approved evidence enters through the Source Finder.</div>
  <div class="process-step"><strong>2 · Analyze</strong>The Analyst turns evidence into structured claims.</div>
  <div class="process-step"><strong>3 · Verify</strong>The Skeptic independently checks the work.</div>
  <div class="process-step"><strong>4 · Publish</strong>The Synthesizer releases the brief only after verification.</div>
</div>
"""
            )
            business_summary = gr.Markdown(
                "Run the mission to generate the executive summary.",
                elem_classes=["result-card"],
            )

        with gr.Tab("Agent Conversation"):
            gr.Markdown(
                "## Watch the peers coordinate\n"
                "The dialogue below is generated from the real typed-message audit trail. It is a readable translation of what each protocol event means—not a fictional chat transcript."
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

        with gr.Tab("Technical View"):
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

        with gr.Tab("Architecture Comparison"):
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

        with gr.Tab("LLM Stress Tests"):
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
        fn=_mission_outputs,
        inputs=[mode],
        outputs=[
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
    demo.launch()
