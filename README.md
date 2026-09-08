# Agent 10 — Peer-to-Peer Coordination Agent

Peer-to-peer multi-agent coordination with typed messaging, local state, bounded decisions, auditable communication, deterministic verification, and optional LLM-assisted work products.

## Status

**Core system and Gradio demo implemented. Hugging Face deployment and live production validation are next.**

This is Agent 10 in the **30 Agents for AI Engineers** portfolio. It moves beyond centralized multi-agent orchestration by allowing specialized peers to coordinate directly while keeping the protocol controlled, testable, and observable.

## Problem

Centralized orchestrators are useful when one component should decide who works, what happens next, and how a process advances. They also create a semantic coordination dependency.

This project asks a different question:

> How can specialized agents coordinate directly with one another while the overall system remains typed, bounded, observable, testable, and aligned with a shared mission?

The goal is not to make several LLMs chat freely. The goal is to engineer the protocol that allows peers to coordinate safely.

## Reusable Primitive

> **A typed peer-to-peer coordination protocol that lets autonomous peers discover one another, propagate mission state, self-select work using local rules, exchange validated work products, resolve bounded disagreements, and maintain an auditable communication history without a central semantic orchestrator.**

## Teaching Principle

> **The runtime turns the clock; the peers make the decisions.**

The runtime schedules opportunities to act. It does not assign semantic work, choose a worker, or decide the mission outcome.

## Public Demo Scenario

The public-safe demo uses a fictional company, **Asteria Robotics**, evaluating whether to enter the fictional **Borealis industrial automation market**.

Four specialized peers coordinate the mission:

1. **Source Finder** — retrieves approved evidence from a synthetic corpus.
2. **Analyst** — produces evidence-linked structured analysis.
3. **Skeptic / Verifier** — checks evidence use, score consistency, and recommendation rules.
4. **Synthesizer** — publishes the final brief only after verification passes.

The synthetic evidence covers demand, channel readiness, unit economics, certification burden, incumbent concentration, and service coverage.

The deterministic mission produces the recommendation:

> **Enter with conditions**

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

The protocol currently supports:

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

Each peer maintains its own mission state rather than sharing a magical globally synchronized context.

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

An Analyst may label a response as revised, but that label is not trusted. The Skeptic must independently validate the replacement work product before publication can continue.

## Failure Containment

The system detects and fails closed on conditions including:

- no available peer for a required role,
- an eligible peer that never participates,
- persistent duplicate role claims,
- missing work dependencies,
- unanswered challenges,
- and general no-progress conditions.

Failure containment does **not** perform advanced replanning or invent replacement agents. That capability belongs to later portfolio agents.

## Centralized vs Peer-to-Peer Evaluation

The evaluation harness runs the same fictional mission and deterministic work logic under two coordination architectures:

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

The current result is intentionally nuanced:

> **Centralized coordination is mechanically simpler and lower-overhead. Peer-to-peer coordination distributes semantic authority and removes the central coordinator dependency, at the cost of additional coordination machinery.**

## Bounded LLM Mode

LLMs are optional and live only inside peer work handlers.

**Agents reason with LLMs. Agents coordinate through an engineered protocol.**

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

Model output is parsed through strict Pydantic schemas with extra fields forbidden.

## Adversarial LLM Evaluation

The offline evaluation matrix currently covers 11 model-behavior scenarios:

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

- deterministic and optional LLM-assisted mission modes,
- final recommendation and verified brief,
- synthetic evidence table,
- published work-product table,
- independent peer-state view,
- typed protocol transport audit,
- centralized-vs-P2P comparison,
- 11-scenario adversarial LLM evaluation,
- and a raw machine-readable mission snapshot.

Deterministic mode requires no API token and remains fully functional if the model provider is unavailable.

## Optional Live LLM Configuration

The live adapter uses the Hugging Face OpenAI-compatible Inference Providers endpoint.

Environment variables:

```text
HF_TOKEN=<runtime inference token>
MODEL_ID=<supported Hugging Face model ID>
HF_BASE_URL=https://router.huggingface.co/v1
```

Never commit real token values. `.env.example` contains placeholders only.

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

The repository currently contains **102 automated tests** covering protocol schemas, discovery, transport, authorization, local policy, runtime behavior, work execution, deterministic research, challenge/revision, failure containment, architecture evaluation, bounded LLM behavior, adversarial model scenarios, and Gradio service/UI wiring.

## Project Structure

```text
.
├── app.py
├── README.md
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
│       ├── llm_evaluation.py
│       └── demo.py
├── tests/
└── .github/
    └── workflows/
        └── tests.yml
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
- Turn meaningful failures into regression tests.
- Fail closed after escalation rather than retrying blindly.
- Use only synthetic, public-safe demo data.
- Never commit secrets or private planning material.

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
Live production validation
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

## License

MIT
