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


with gr.Blocks(title="Agent 10 — Peer-to-Peer Coordination", analytics_enabled=False) as demo:
    gr.Markdown(
        """
# Agent 10 — Peer-to-Peer Coordination Agent

Four specialized peers independently coordinate a fictional market-entry investigation for **Asteria Robotics**. There is no semantic supervisor assigning the work: peers discover the mission, claim roles locally, broadcast capability requests, validate one another's work, and publish only after the verification boundary passes.

**Teaching principle:** *The runtime turns the clock; the peers make the decisions.*
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
            + "\n\nDeterministic mode is always available and uses the same coordination protocol."
        )

    run_button = gr.Button("Run coordinated mission", variant="primary")

    with gr.Row():
        mission_status = gr.Markdown("Run the mission to see the peer network coordinate.")
        final_brief = gr.Markdown("The verified final brief will appear here.")

    with gr.Tabs():
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

        with gr.Tab("Peer Network"):
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
                "Machine-readable state from the most recent mission run. No secret material is included."
            )
            raw_snapshot = gr.JSON(value={})

    run_button.click(
        fn=_mission_outputs,
        inputs=[mode],
        outputs=[
            mission_status,
            final_brief,
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
