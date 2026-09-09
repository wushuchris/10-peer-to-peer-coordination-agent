---
title: Agent 10 — Peer-to-Peer Coordination
emoji: 🤝
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
python_version: "3.11"
license: mit
pinned: false
short_description: Typed peer coordination without a central orchestrator
---

# Agent 10 — Peer-to-Peer Coordination Agent

A production-validated multi-agent coordination system where specialized peers communicate directly, self-select bounded work, challenge one another, and publish only after verification—without a central semantic orchestrator assigning the work.

## Status

**Complete and production validated — September 2026.**

- **Public repo:** https://github.com/wushuchris/10-peer-to-peer-coordination-agent
- **Live demo:** https://huggingface.co/spaces/FlyingNunchucks/10-peer-to-peer-coordination-agent
- **Automated tests:** **112 passed**
- **Live LLM path:** validated with `Qwen/Qwen3.8-27B:ovhcloud`
- **Deployment:** GitHub Actions tests gate automatic GitHub → Hugging Face deployment

This is Agent 10 in the **30 Agents for AI Engineers** portfolio. It moves beyond centralized multi-agent orchestration by allowing specialized peers to coordinate directly while keeping the protocol controlled, testable, observable, and auditable.

## Business Story

The public-safe demo asks a simple management question:

> **Should fictional Asteria Robotics enter the fictional Borealis industrial automation market?**

Instead of asking one AI to research, analyze, verify, and decide everything, four specialized peers collaborate:

1. **Source Finder** — finds the approved facts.
2. **Analyst** — turns evidence into structured insight.
3. **Skeptic / Verifier** — challenges weak reasoning and checks the evidence.
4. **Synthesizer** — prepares the final brief only after verification passes.

The business-facing workflow is:

```text
Gather evidence
      ↓
Analyze evidence
      ↓
Verify the analysis
      ↓
Publish the final brief
```

The system reaches:

> **Enter with conditions**

The result is deliberately easy to understand for a non-technical viewer while the underlying engineering evidence remains inspectable.

## Business-First Demo Design

The Gradio app now presents two layers.

### Business layer

The default experience explains:

- the business decision,
- the four-agent team,
- the Gather → Analyze → Verify → Publish workflow,
- the executive recommendation,
- opportunities and risks,
- why independent verification matters,
- and how the agents coordinate in plain English.

### Engineering layer

Technical reviewers can still inspect:

- independent peer state,
- typed protocol messages,
- message IDs and correlation IDs,
- work products,
- evidence references,
- centralized-vs-P2P evaluation,
- adversarial LLM tests,
- and the raw machine-readable mission snapshot.

The **Agent Conversation** view is not invented dialogue. It is a presentation layer derived from the real append-only typed-message audit trail. Each human-readable line remains traceable to the underlying protocol event.

## Problem

Centralized orchestrators are useful when one component should decide who works, what happens next, and how a process advances. They also create a semantic coordination dependency.

This project asks a different question:

> How can specialized agents coordinate directly with one another while the overall system remains typed, bounded, observable, testable, and aligned with a shared mission?

The goal is not to make several LLMs chat freely. The goal is to engineer the protocol that allows peers to coordinate safely.

## Reusable Primitive

> **A typed peer-to-peer coordination protocol that lets autonomous peers discover one another, propagate mission state, self-select work using local rules, exchange validated work products, resolve bounded disagreements, and maintain an auditable communication history without a central semantic orchestrator.**

## Teaching Principles

> **The runtime turns the clock; the peers make the decisions.**

and

> **The agents reason with LLMs. The agents coordinate through an engineered protocol.**

The runtime schedules opportunities to act. It does not assign semantic work, choose a worker, or decide the mission outcome.

## Coordination Architecture

```text
Shared Mission
      |
      v
Peer Registry / Discovery
      |
      v
+----------------+      typed messages      +----------------+
|  Source Finder | <----------------------> |     Analyst    |
|  local state   |                          |   local state   |
+----------------+                          +----------------+
         ^                                          ^
         |                                          |
         |          Passive Message Bus             |
         |     validation + dedupe + audit          |
         v                                          v
+----------------+      typed messages      +----------------+
| Skeptic/Verify | <----------------------> |  Synthesizer   |
|  local state   |                          |   local state   |
+----------------+                          +----------------+
```

### Infrastructure may

- validate message schemas,
- verify approved senders and recipients,
- expose peer discovery information,
- deliver messages,
- deduplicate deliveries,
- record append-only transport events,
- schedule processing turns,
- detect coordination failures,
- and stop the system when safety conditions fail.

### Infrastructure must not

- assign semantic work to agents,
- choose which peer should reason about a task,
- determine research conclusions,
- bypass verification,
- silently replace an unavailable peer,
- or act as a hidden supervisor agent.

## Typed Protocol

The protocol supports:

- `MISSION_ANNOUNCEMENT`
- `CAPABILITY_ADVERTISEMENT`
- `ROLE_CLAIM`
- `ROLE_RELEASE`
- `WORK_REQUEST`
- `WORK_RESULT`
- `CHALLENGE`
- `CHALLENGE_RESPONSE`
- `STATUS`
- `ESCALATION`

Every envelope carries application-generated identifiers, mission context, sender/recipient identity, protocol version, correlation state, timestamp, and a typed payload.

Structural schema validity is intentionally separate from contextual authorization. A message can be structurally valid and still be rejected if the sender is unknown, unavailable, ineligible, or attempting to expand its registered authority.

## Local Coordination

Each peer maintains its own mission state rather than sharing a globally synchronized hidden context.

Peers locally decide whether to:

- claim an eligible role,
- retain or release a role,
- accept a capability request,
- ignore irrelevant work,
- wait for dependencies,
- challenge peer work,
- respond to a challenge,
- or escalate.

Duplicate exclusive role claims converge through a transparent deterministic tie-break rule rather than a central allocator.

## Work Authorization Boundary

Receiving a work request does not execute work automatically.

```text
WORK_REQUEST
     ↓
local policy decision
     ↓
accepted_work_ids
     ↓
dependencies available?
     ↓
registered local handler?
     ↓
execute
```

Only work explicitly accepted by the peer and backed by a registered local handler can execute.

If execution fails and the peer escalates, later execution rounds cannot retry that accepted work automatically.

## Verification and Peer Disagreement

The deterministic research path verifies:

- cited evidence IDs exist,
- the complete retrieved evidence set is declared,
- the weighted evidence score is correct,
- and the recommendation matches the deterministic score rule.

The challenge protocol supports a real multi-turn disagreement loop:

```text
Analyst WORK_RESULT
        ↓
Skeptic CHALLENGE
        ↓
Analyst CHALLENGE_RESPONSE
        ↓
Skeptic independently evaluates response
        ↓
resolved / revised / disputed
```

An Analyst may label a response as revised, but that label is not trusted. The Skeptic independently validates the replacement work product before publication can continue.

## Failure Containment

The system detects and fails closed on conditions including:

- no available peer for a required role,
- an eligible peer that never participates,
- persistent duplicate role claims,
- missing work dependencies,
- unanswered challenges,
- and general no-progress conditions.

Failure containment does **not** perform advanced replanning or invent replacement agents. Those capabilities belong to later portfolio agents.

## Centralized vs Peer-to-Peer Evaluation

The evaluation harness runs the same fictional mission and deterministic work logic under two coordination architectures.

### Peer-to-peer

Peers discover the mission, self-select work, exchange protocol messages, and coordinate through local state.

### Centralized baseline

One Agent-8-style coordinator owns a capability-to-worker mapping and directly invokes the same handlers.

The comparison measures:

- mission completion,
- final work-product digest,
- coordination steps,
- protocol message count,
- message fan-out deliveries,
- work executions,
- trace evidence,
- central coordinator dependency,
- and explicit failure behavior.

Healthy runs must produce the same final work-product digest. This keeps the comparison focused on coordination architecture rather than different business logic.

The conclusion is intentionally nuanced:

> **Centralized coordination is mechanically simpler and lower-overhead. Peer-to-peer coordination distributes semantic authority and removes the central coordinator dependency, at the cost of additional coordination machinery.**

The failure comparison also avoids overclaiming: losing the central coordinator breaks the centralized baseline but not the P2P system, while losing the only available Analyst blocks both systems. Decentralization removes one dependency; it does not create magical redundancy.

## Bounded LLM Mode

LLMs are optional and live only inside peer work handlers.

### Analyst LLM may propose

- summary prose,
- claim text,
- evidence references,
- assumptions,
- and confidence.

It cannot control:

- recommendation,
- evidence score,
- claim IDs,
- role identity,
- routing,
- work IDs,
- or completion state.

### Skeptic LLM may propose

- review prose,
- concerns,
- and evidence-linked critique.

It cannot set the deterministic verification status.

### Synthesizer LLM may propose

- final narrative prose.

It cannot change:

- the verified recommendation,
- verification status,
- evidence set,
- work-product ID,
- or publication permission.

The Source Finder remains deterministic so the model cannot invent the evidence corpus.

Model output is constrained by strict Pydantic JSON Schemas with extra fields forbidden. Unknown evidence references, malformed structured output, provider errors, and truncation fail closed.

## Live LLM Configuration Validated

The production Space was validated using:

```text
MODEL_ID=Qwen/Qwen3.8-27B:ovhcloud
HF_BASE_URL=https://router.huggingface.co/v1
```

The runtime token is stored only as the Hugging Face Space secret `HF_TOKEN`.

For Qwen3.8, the provider adapter requests `reasoning_effort="low"` because the default reasoning depth can consume the bounded completion budget before the structured JSON object finishes. Task-specific token ceilings remain bounded, and provider output must still pass the exact application schema.

See [`PRODUCTION_VALIDATION.md`](PRODUCTION_VALIDATION.md) for the live validation record and the provider failures that were converted into regression-tested safeguards.

## Adversarial LLM Evaluation

The offline evaluation matrix covers 11 model-behavior scenarios:

- healthy bounded reasoning,
- hallucinated Analyst evidence,
- forbidden Analyst control fields,
- malformed Analyst JSON,
- Analyst model outage,
- contradictory Analyst narrative,
- hallucinated Skeptic evidence,
- contradictory Skeptic advisory prose,
- incomplete synthesis evidence,
- forbidden Synthesizer control fields,
- and Synthesizer model outage.

The tests ask:

> **Can bad or broken model output escape deterministic governance?**

For the implemented control-plane guarantees, model failures either preserve application-owned state or fail closed before publication.

## Known Semantic Limitation

The deterministic verifier checks evidence membership, completeness, score correctness, and recommendation rules.

It does **not** prove that free-form natural-language prose is semantically entailed by the evidence it cites.

That means structurally valid but semantically contradictory prose can currently survive structural verification while the application-owned recommendation, verification result, and evidence set remain correct.

This limitation is explicitly surfaced by the adversarial evaluation rather than hidden.

## Gradio Demo

`app.py` provides a public-facing Gradio interface with:

- a **Business Story** view with agent cards, process flow, and executive summary,
- a protocol-derived **Agent Conversation** view,
- deterministic and optional LLM-assisted mission modes,
- final recommendation and verified brief,
- synthetic evidence and published work products,
- a **Technical View** with independent peer state and transport audit,
- centralized-vs-P2P comparison,
- 11-scenario adversarial LLM evaluation,
- and a raw machine-readable mission snapshot.

Deterministic mode requires no API token and remains fully functional if the model provider is unavailable.

## Deployment

GitHub remains the source of truth. The GitHub Actions pipeline runs the complete automated test suite before syncing `main` to the public Hugging Face Space.

The workflow uses two intentionally separate credentials:

- GitHub secret `HF_DEPLOY_TOKEN` — write-scoped deployment credential used only by GitHub Actions.
- Hugging Face Space secret `HF_TOKEN` — runtime inference credential used only by optional LLM-assisted mode.

The final business-first UI and both deterministic and live LLM-assisted mission paths were manually validated in the public Space after deployment.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Run automated tests with:

```bash
python -m pytest -q
```

## Current Test Coverage

The repository contains **112 automated tests** covering:

- protocol schemas,
- discovery and transport,
- authorization and local policy,
- independent peer runtime,
- work execution,
- deterministic research,
- challenge/revision,
- failure containment,
- centralized-vs-P2P evaluation,
- bounded LLM behavior,
- strict provider structured output,
- live-discovered provider regressions,
- adversarial model scenarios,
- business-story presentation,
- protocol-to-conversation provenance,
- and Gradio service/UI wiring.

## Project Structure

```text
.
├── app.py
├── README.md
├── PRODUCTION_VALIDATION.md
├── requirements.txt
├── .env.example
├── pyproject.toml
├── src/
│   └── peer_coordination/
│       ├── models.py
│       ├── registry.py
│       ├── transport.py
│       ├── policy.py
│       ├── runtime.py
│       ├── research.py
│       ├── challenge.py
│       ├── challenge_demo.py
│       ├── resilience.py
│       ├── evaluation.py
│       ├── llm.py
│       ├── hf_runtime.py
│       ├── llm_evaluation.py
│       └── demo.py
├── tests/
└── .github/
    └── workflows/
        ├── tests.yml
        └── deploy-huggingface.yml
```

## Engineering Principles

- GitHub is the source of truth.
- Build in small, auditable commits.
- Models propose; application code enforces.
- Prefer typed schemas and deterministic controls over prompt-only rules.
- Test deterministic behavior before making live model calls.
- Validate every peer message before it becomes trusted state.
- Do not use LLM-generated prose as a system identifier.
- Keep peer local state explicit rather than hiding coordination in one global context.
- Turn meaningful live failures into regression tests.
- Fail closed after escalation rather than retrying blindly.
- Use only synthetic, public-safe demo data.
- Never commit secrets or private planning material.
- Present the business story first while preserving technical evidence underneath.

## Development Workflow

```text
Design in ChatGPT
      ↓
Build in GitHub
      ↓
Automated tests
      ↓
Tests pass
      ↓
GitHub Actions deployment
      ↓
Hugging Face Space
      ↓
Live deterministic validation
      ↓
Live LLM-assisted validation
      ↓
Documentation and portfolio closeout
```

Deployment is intentionally gated behind successful automated tests.

## Non-Goals for Agent 10

To keep the learning objective clear, this build does not implement:

- auction-based task allocation,
- sophisticated reputation or trust scoring,
- voting or consensus algorithms,
- full role-drift monitoring,
- compromised-agent isolation,
- or advanced fault-tolerant replanning.

Those capabilities belong to later agents in the portfolio.

## Production Validation

Agent 10 passed the final live checks in both execution modes after the business-first UI redesign:

- deterministic mission completed,
- LLM-assisted mission completed,
- recommendation remained `enter_with_conditions`,
- verification remained `verified`,
- Business Story rendered correctly,
- Agent Conversation rendered from real protocol events,
- Technical View preserved the raw audit evidence,
- and the public repository hygiene review found no real credentials or private planning material.

## License

MIT
