# Exact Stage-2 method contract

## Native information flow

The Qwen template opens `<think>\n`. The Actor freely generates reasoning and
closes it, then emits the ordinary native tool call or:

```text
reasoning body</think>

<RETHINK><tool_call><function=...>...native parameters...</function></tool_call><|im_end|>
```

The prefix is the actual pre-action native task/history/canvas/tool schema.
No supplied old reasoning, forced marker, Episode Analyzer, JSON-only target or
post-action protocol is added. Parser, dispatcher, canvas feedback and history
windows remain baseline. Actual tools execute before hint generation: Teacher
does not regenerate or replace the executed action.

For Student prefix `s` and sampled response IDs `y`, routing is Actor-only:

$$
z_i=\mathbf{1}\{\text{one valid Actor-generated pre-action }\texttt{<RETHINK>}\}.
$$

The hint writer receives the completed trajectory plus selected decisions' exact
pre-action histories, reasoning and canvases. It writes local hints, **not**
Good/Bad, confidence thresholds or permission to route. Teacher conditioning is:

$$
s_i^T=s_i\oplus k_{\mathrm{static}}\oplus z_i h_i.
$$

Implementation appends static skill, then local hint, to a **copied Teacher
system message**, before native processing. Student messages, image pixels and
response IDs are not edited. Missing `h_i` preserves the original static-skill
Teacher; it never cancels the marker, GRPO participation or an executed action.
Malformed/misplaced/repeated markers remain recorded but get no local hint;
their real final canvas and generated tokens still train normally as baseline.
Evaluation creates no hint client and makes no hint API requests.

## Teacher versus reference

- Teacher: current Actor weights, before optimization, detached scoring of
  the **same sampled response IDs** under privileged conditioning.
- Student: original task/history/images and sampled prefix, trainable weights.
- Reference: frozen **Stage-2 initial policy**, step520 for the method, base
  for the original base-start control. This baseline logic is unchanged.

Sample/full-forward log-probability discrepancy above 0.005 aborts as baseline.
Hint fallback never bypasses that numerical alignment check.

## Unchanged objective

Public VERL advantages group complete trajectories by initial stimulus. Each
episode gets one terminal ViDrA-adapted score, expanded to its generated tokens.
Only generated tokens have loss. Accumulation gives a single generated-token
mean across the whole update, not independent optimizer steps per decision.

Pinned public SDAR computes:

$$
\delta_t=\log p_T(y_t\mid s^T,y_{<t})-
\operatorname{sg}[\log p_S(y_t\mid s,y_{<t})],\quad
w_t=\operatorname{sg}[\sigma(5\delta_t)].
$$

$$
L_{\mathrm{SDAR}}=
\frac{\sum_t m_t w_t(\operatorname{sg}[\log p_T(y_t)]-\log p_S(y_t))}{\sum_t m_t}.
$$

This signed sampled-token scalar is not a nonnegative full-vocabulary KL.
Teacher scores and gate weights are detached. Original functions, not replacement
surrogates, are loaded from SHA-verified public sources.

$$
L=L_{\mathrm{GRPO}}+0.01L_{\mathrm{SDAR}}+
0.01L_{\mathrm{reference\ KL}}-0.001H.
$$

GRPO clipping, low-variance reference KL estimator, entropy, normalization,
coefficients and **beta=5** remain baseline. Generated reasoning, markers and
tool calls all retain their baseline RL participation. The Stage-1 reasoning
mask and marker BCE are not silently reused in Stage 2.

## Hints and failure handling

The complete prompt is `configs/local_hint_prompt.txt`. It requests one to three
local, actionable sentences about state, geometry, tool premises or correction,
preserving the intended composition and unaffected marks. It forbids inventing
a hidden reference drawing, clearing the whole canvas as a default, rewarding
texture/ink quantity, or generating replacement full tool calls.

Each completed-trajectory request batches its triggered decisions and actual
Actor-visible PNGs; four trajectory requests can run concurrently. The API must
be vision-capable. Two bounded retries cover transport/429/5xx/JSON failures.
Structural checks reject classifier fields, duplicate/foreign indices, overlong
hints and full-action markup. They do **not** prove semantic hint accuracy.

Successful exact payload/model/prompt/settings are content-addressed and cached.
Student input tensors are hashed before/after attachment. Selected decisions
store hint/fallback audits and actual Teacher token/pixel hashes. Hints never
leak into a neighboring unmarked decision. Empty/missing hints, API failure or
additional-Teacher-context overflow independently fall back to static skill;
ordinary context, environment, sampler and numerical checks remain intact.
No raw API error text, authentication header or key is logged.

## Initialization and reproduction limits

The supplied epoch2 step520 model used 1,560 natural Good + 520 natural Bad,
6:2 batches, two epochs, no synthetic augmentation. Stage 1 used suffix CE +
0.1 independent full-vocabulary marker BCE, no KL. Reasoning and its closing
separator had zero CE. Bad targets were local next native operations, not
full-canvas replacement. The original native tokenizer/processor/template and
all FP32 tensor bytes are preserved.

This repository independently prepares **Stage 2 from the supplied model**.
It does not distribute the original annotation corpus or Stage-1 optimizer,
promise byte-for-byte Stage-1 rebuilding, or guarantee identical API text after
provider backend changes. Pin your API model and retain the request cache.

The default 50 updates and six-task validation are the public Painter baseline's
protocol, not ALFWorld's 150 updates or an official PainterBench RL recipe.
Disclose initialization and hint factors separately in comparisons.
