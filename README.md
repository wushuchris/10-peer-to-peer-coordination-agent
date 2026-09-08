# Agent 10 — Peer-to-Peer Coordination Agent

Peer-to-peer multi-agent coordination with typed messaging, local state, bounded decisions, and auditable communication.

## Status

**In development — architecture and project foundation established.**

This is Agent 10 in the **30 Agents for AI Engineers** portfolio. It begins the portfolio's transition from centralized multi-agent control toward decentralized agent coordination.

## Problem

Centralized orchestrators are useful when one component should decide who works, what happens next, and how a process advances. They also create a central coordination dependency.

This project asks a different question:

> How can specialized agents coordinate directly with one another while the overall system remains typed, bounded, observable, testable, and aligned with a shared mission?

The goal is not to make several LLMs chat freely. The goal is to engineer the protocol that allows peers to coordinate safely.

## Learning Objective

Build a small group of agents that can:

- discover approved peers and their capabilities,
- receive and propagate a shared mission,
- maintain independent local state,
- self-select work using bounded local rules,
- exchange typed peer-to-peer messages,
- challenge and respond to peer work,
- detect invalid or stalled coordination,
- and preserve an auditable communication history.

## Reusable Primitive

The intended reusable primitive is:

> **A typed peer-to-peer coordination protocol that lets autonomous peers discover one another, propagate mission state, self-select work using local rules, exchange validated work products, and maintain an auditable communication history without a central semantic orchestrator.**

## Architecture Boundary

The system deliberately separates **coordination infrastructure** from **agent reasoning**.

```text
Shared Mission
      |
      v
Peer Registry / Discovery
      |
      v
+-------------+       typed messages       +-------------+
| Peer Agent  | <-------------------------> | Peer Agent  |
| local state |                             | local state |
+-------------+                             +-------------+
       ^                                           ^
       |                                           |
       +---------- Message Transport --------------+
                    + validation
                    + deduplication
                    + audit logging
```

Infrastructure may:

- validate schemas,
- verify known senders and recipients,
- deliver messages,
- deduplicate messages,
- record communication events,
- and detect protocol-level failures.

Infrastructure must **not**:

- assign semantic work to agents,
- choose which peer should reason about a task,
- determine research conclusions,
- bypass verification,
- or act as a hidden supervisor agent.

A runtime may schedule opportunities for peers to process messages, but peers own the coordination decisions.

## Planned MVP

The first public-safe scenario will be a synthetic research mission with four specialized peers:

1. **Source Finder** — retrieves approved evidence from a synthetic corpus.
2. **Analyst** — produces structured analysis from validated evidence.
3. **Skeptic / Verifier** — challenges unsupported reasoning and evidence use.
4. **Synthesizer** — produces the final brief from validated peer contributions.

The coordination layer will be deterministic first. Bounded LLM work handlers will be introduced only after the peer protocol is proven by automated tests.

## LLM Boundary

LLMs will contribute bounded work products such as:

- evidence interpretation,
- analysis,
- critique,
- challenge responses,
- and final synthesis.

Application code will retain authority over:

- agent identifiers,
- mission identifiers,
- message identifiers,
- protocol versions,
- message validation,
- peer discovery,
- role claims,
- permissions,
- delivery,
- deduplication,
- timeout/escalation rules,
- and publication boundaries.

**Agents reason with LLMs. Agents coordinate through an engineered protocol.**

## Non-Goals for Agent 10

To keep the learning objective clear, this build will not yet implement:

- auction-based task allocation,
- sophisticated reputation or trust scoring,
- voting or consensus algorithms,
- full role-drift monitoring,
- compromised-agent isolation,
- or advanced fault-tolerant replanning.

Those capabilities belong to later agents in the portfolio.

## Planned Project Structure

```text
.
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   └── peer_coordination/
│       └── __init__.py
├── tests/
│   └── __init__.py
└── .github/
    └── workflows/
```

Additional modules, evaluation files, demo data, UI, and deployment configuration will be added incrementally as their behavior becomes testable.

## Engineering Principles

- GitHub is the source of truth.
- Build in small, auditable commits.
- Prefer typed schemas and deterministic controls over prompt-only rules.
- Test deterministic behavior before making live model calls.
- Validate every peer message before it becomes trusted state.
- Do not use LLM-generated prose as a system identifier.
- Keep peer local state explicit rather than hiding coordination in one global context.
- Turn meaningful live failures into regression tests.
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

Deployment will be added only after the deterministic core and evaluation suite are working.

## License

MIT
