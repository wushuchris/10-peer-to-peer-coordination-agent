# Production Validation

## Status

**Passed — September 8, 2026**

Agent 10 was validated end-to-end on the public Hugging Face Space after deployment from GitHub through the gated CI workflow.

## Validated deployment

- GitHub repository: `wushuchris/10-peer-to-peer-coordination-agent`
- Hugging Face Space: `FlyingNunchucks/10-peer-to-peer-coordination-agent`
- Runtime mode: LLM-assisted
- Model/provider: `Qwen/Qwen3.8-27B:ovhcloud`
- Hugging Face router: `https://router.huggingface.co/v1`
- Structured output: strict Pydantic JSON Schema
- Qwen reasoning setting: `reasoning_effort="low"`

## Live mission result

The full four-peer market-entry mission completed successfully in LLM-assisted mode.

Application-owned outcome:

- Mission: completed
- Recommendation: `enter_with_conditions`
- Verification: `verified`
- Published evidence: `EV-001`, `EV-002`, `EV-003`, `EV-004`, `EV-005`, `EV-006`

The live path exercised:

1. deterministic Source Finder retrieval,
2. bounded Analyst LLM work-product generation,
3. bounded Skeptic LLM critique plus deterministic verification,
4. bounded Synthesizer LLM narrative generation,
5. deterministic publication gating.

## Production findings converted into safeguards

Live validation exposed several provider/model integration failures before the final successful run. Each was handled without weakening the governance boundary:

- unsupported provider `extra_body` arguments were removed,
- provider requests now use exact Pydantic `json_schema` output contracts,
- provider errors are sanitized before appearing in the UI,
- `finish_reason="length"` is detected explicitly as truncation,
- task-specific bounded token budgets are used,
- Qwen3.8 receives `reasoning_effort="low"` to avoid spending the completion budget on unnecessary reasoning,
- peers fail closed after model/provider errors or malformed outputs,
- escalated work is not retried automatically.

## Governance guarantees preserved in the live run

The model still does not control:

- peer identity,
- discovery,
- role claims,
- routing,
- message IDs or work-product IDs,
- evidence access,
- deterministic score,
- final recommendation,
- verification status,
- publication authority,
- retry/escalation state.

The production result therefore validates the project teaching principle:

> **The agents reason with LLMs. The agents coordinate through an engineered protocol.**

and the broader engineering rule:

> **Models propose; application code enforces.**
