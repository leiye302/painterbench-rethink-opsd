# Selected local-hint profile: MiMo-V2.6-Pro

Use `configs/local_hints_mimo_go.json` for the selected OpenCode Go profile.
This changes only the external hint writer and its prompt/transport parameters,
not GRPO, SDAR beta=5, rewards, task selection, evaluation, Actor routing, or
Student visibility. The generic `configs/local_hints.json` remains available.

## Configuration

```bash
export HINT_API_BASE="https://opencode.ai/inference/go/openai/v1"
export HINT_MODEL="mimo-v2.6-pro"
export HINT_API_SESSION="painterbench-method-mimo-v1"
read -rsp "Hint API key: " HINT_API_KEY; echo
export HINT_API_KEY
```

Use your own key. `HINT_API_SESSION` is an honest, non-secret identifier for
this run: keep it stable across requests and resume. Do not rotate sessions or
credentials to evade provider quotas. The adapter supplies its own honest
User-Agent and the required `x-opencode-session` header. It requests
temperature=0, max_tokens=2048, stream=false, thinking.type=disabled and
response_format.type=json_object. Provider support for request options does
not guarantee that backend hidden reasoning is disabled.

The system prompt is the selected constrained prompt, with normalized UTF-8/LF
SHA256 `d17fb02199b175a8eabae7be456ba8ebc9ed1d08a3a1ce9df8a94c90e3d1d6cf`.
In particular, it distinguishes white outline erasure from clearing the canvas,
requires integer indices and forbids aesthetic preference as an error diagnosis.

Keep the README's preparation, model, training and storage instructions. In
both its integration and formal method commands, replace only
`--method-config configs/local_hints.json` with
`--method-config configs/local_hints_mimo_go.json`. Use a fresh run/output and
checkpoint directory for a newly selected provider; do not silently change an
existing experiment or bypass its resume contract. Restore requires the same
model, prompt, provider and session identifier. There is no GPT model fallback.

The Go endpoint can charge an account balance when subscription allowance is
unavailable. The client does not guarantee subscription-only billing or impose
a dollar cap. Set account-side spending controls and obtain the run owner's
budget authorization before training. Configuration and CPU tests make no API
calls. A paid quality-test allowance does not authorize a formal training budget.

## Unchanged training boundary

Only the Actor's eligible generated marker activates hints. Post-rollout
requests contain at most four selected decisions from the same trajectory,
with at most four trajectories in flight. Each request has 2048 output tokens;
an eleven-trigger trajectory starts with batches of 4, 4 and 3, not one long
eleven-hint response. A hint is privileged Teacher-only information;
the Teacher scores the exact sampled response. Empty, invalid, truncated or
failed responses retain the ordinary static-skill Teacher and never cancel
the route or GRPO. The hint writer supplies no Good/Bad classifier. Student
inputs and evaluation never receive hints.

## MiMo v2 failure isolation and resume

- Keep every individually valid hint. A bad index, extra field, duplicate,
  oversized hint or forbidden action/control markup invalidates only the
  affected decision; foreign, string, boolean and float indices are never
  coerced or attached to another turn.
- Missing/invalid items and transient transport/response failures receive at
  most two retries per decision. Retry batches shrink from four to two to one;
  successful items are not requested again. This can make more API requests
  than a single whole-trajectory call, so account for cost and latency.
- An explicit empty hint is a completed, safe abstention. It is not retried
  and is counted in `empty_hint_abstentions`, separately from `hint_failures`
  and `teacher_context_fallbacks`. The legacy `fallbacks` total still counts
  all unapplied hints; do not interpret that aggregate as an API failure rate.
- Null or reasoning-only content, invalid JSON, wrong envelope and truncated
  responses have distinct failure codes. Text content blocks are supported;
  hidden reasoning is never substituted for the required hint answer. No
  truncated text is applied, even if its JSON happens to parse.
- Text-budget overflow is split before dispatch; a single oversized decision
  falls back with an explicit code. HTTP 401/403 stops further dispatch for
  that episode. Transient 408/429/5xx/network failures stay within the same
  retry budget, without key/model/session switching.
- A versioned cache records outcomes and counts before each dispatch and
  after each response. Partial resumes preserve successes and consume the
  remaining budget; completed failure outcomes are reused without starting
  another retry loop. After a crash, an in-flight request is counted as an
  attempted request because the provider may already have billed it. This
  is not an exactly-once delivery or billing guarantee.
- Per-decision audit files include attempt count, fixed error codes,
  requested indices, payload SHA, elapsed time, finish reason, content type
  and numeric token usage when available. They do not contain keys, headers,
  raw HTTP error bodies or hidden-reasoning text. Valid hints and the original
  trajectory remain private runtime artifacts, not public Git contents.

The generic provider's strict schema and transport path are unchanged. The
new MiMo profile adds `max_decisions_per_request=4` and records protocol
`mimo_partial_hints_v2`; its cache and training contract are not interchangeable
with v1. Do not install this as an unrecorded change mid-training or bypass
the existing checkpoint contract check.

Offline verification commands (no provider charge or GPU):

```bash
python -m unittest discover -s tests -p test_hint_protocol.py -v
python -m unittest discover -s tests -v
python scripts/check_release.py
```

Fixtures include an eleven-decision batch, ten valid hints alongside one bad
item, null content, truncation, text blocks, missing/duplicate indices,
authentication failures and interrupted-cache resume. These establish code
behavior, not a measured live-provider success rate. A fresh real multi-trigger
smoke is still needed before claiming that the production failure rate is fixed.

## Evidence and remaining checks

A small historical-decision quality test returned valid JSON for 12/12 MiMo
requests. Manual review found 7 directly usable, 4 partially useful and 1
unsuitable hint. Mean response latency was approximately 5.6 seconds; the 12
calls cost approximately USD 0.115. These are single-decision requests, not
fresh Stage-2 marker-triggered, multi-decision batches or training results.
They are not a claim that hint errors have been eliminated or that task
performance improves. A real multi-trigger runtime integration still needs
the documented GPU smoke and audit before a formal run.
