# Baseline contract — integration verified, not a reproduced learning result

## Source boundary

This repository combines an existing group-relative policy gradient objective
with a new drawing-environment adapter. It does not reproduce a PainterBench RL
experiment: PainterBench releases a drawing benchmark and creativity scorer,
not an Actor post-training recipe.

Source revisions are in `sources.lock.json`. The loss loader compiles the exact
selected upstream function bodies, validates file SHA-256, and records AST hashes.

- VAGEN's VERL fork supplies `compute_grpo_outcome_advantage`,
  `compute_policy_loss`, and `kl_penalty`.
- SDAR supplies `agg_loss` and `compute_sdar_loss`. Gate beta is **5**, not 0.
- PainterBench supplies the native drawing environment and candidate
  **ViDrA-adapted** score.
- The Actor starts from **Qwen/Qwen3.5-4B**, before any project Route-SFT.

Official sources:
[VAGEN](https://github.com/mll-lab-nu/VAGEN),
[VERL fork](https://github.com/JamesKrW/verl),
[SDAR](https://github.com/ZJU-REAL/SDAR),
[PainterBench](https://huggingface.co/datasets/painterbench/painterbench).

## Trajectory grouping and gradient path

For each initial stimulus and task prompt, sample \(G=8\) independent trajectories.
Do not group different initial stimuli together. Do not count successive turns
of a single trajectory as independent samples.

Given one final outcome score \(r_i\) per trajectory:

$$
A_i=\frac{r_i-\operatorname{mean}_{j\in g(i)} r_j}
{\operatorname{std}_{j\in g(i)}r_j+10^{-6}}.
$$

The pinned upstream implementation uses the sample standard deviation. An
all-equal group receives zero GRPO advantage. Do not delete unsuccessful or
non-finished trajectories from a group.

Each trajectory's advantage is broadcast over its actual sampled tokens.
The generated-token mask includes sampled reasoning and native function-call
tokens. System/user text, images, previous tool feedback, padding and
template-provided assistant prefixes have mask zero. Training on a per-turn
record must use that turn's **exact generation-time context**; do not replace
a truncated online context with an untruncated full history.

The GRPO term is the upstream clipped policy loss:

$$
\mathcal L_{\rm GRPO} =
-\operatorname{mean}_{M_{it}=1}
\min\left(\rho_{it}A_i,
\operatorname{clip}(\rho_{it},1-\epsilon,1+\epsilon)A_i\right).
$$

Microbatch losses must be weighted by their number of generated tokens to
reproduce the global token mean. Equal averaging of differently sized turns
would implement a different objective.

The old policy is the rollout checkpoint. It does not change during the update.
Before enabling training, score replay must verify raw sampled token IDs and
causal label shift. The training sampler must match the probabilities used for
the importance ratio: use temperature 1 and no top-k/top-p truncation or
repetition/presence penalty, unless an explicitly tested adapter also replays
those transformations when computing both old and current log probabilities.

## OPSD

The Teacher is **the current Actor under an enriched prompt, evaluated without
gradients**. It is not a permanently frozen Teacher model. The separately frozen
base model is only the reference for reference KL.

Teacher scores the Student's exact sampled continuation. It does not generate a
replacement action, rewrite a trajectory, or change the environment reward.
The baseline adds a fixed drawing-tool skill to the Teacher prompt only.
There is no Analyzer, local hint, Route-SFT checkpoint, or RETHINK token in this
baseline.

For sampled-token log probabilities \(\ell_T,\ell_S\), public SDAR uses:

$$
g=\operatorname{stopgrad}
\sigma\left(5(\ell_T-\operatorname{stopgrad}\ell_S)\right),
\qquad
\mathcal L_{\rm SDAR}=
\operatorname{mean}_{M=1}\left[g(\ell_T-\ell_S)\right].
$$

This sampled, detached-gate auxiliary objective is not interchangeable with a
full-vocabulary KL. The combined candidate defaults retain the prior public
baseline's coefficients:

$$
\mathcal L =
\mathcal L_{\rm GRPO}
+0.01\mathcal L_{\rm SDAR}
+0.01\mathcal L_{\rm ref}
-0.001\mathcal H.
$$

These coefficients are a proposed matched experimental contract, not official
PainterBench settings or evidence of reward validity.

## Drawing environment

Preserve native function schemas, the 400-by-400 canvas, fixed 5-pixel strokes,
indelible starting stimulus, last 10 tool-call/result history and 3-canvas visual
history. Do not replace the environment by full SVG generation.

Keep real tool errors and their feedback. Record termination separately:
`drawing_finished`, no tool call, token limit, turn limit, or infrastructure
failure. A capped rollout still has a last canvas and must not disappear from
the denominator. Retry infrastructure failures only; do not resample bad model
outputs until they become valid.

The sampler stops at the tokenizer's native chat EOS (`<|im_end|>`) as well as
any model EOS. A missing optional `generation_config.json` must not allow it to
invent subsequent user/tool turns. Do not stop at the first tool-call closing
tag: the native executor handles all complete calls in a response and executes
finish last, matching the upstream dispatch semantics.

## Reward: passive validation passed; optimization robustness is not proven

The candidate terminal reward is the released **ViDrA-adapted** prediction on
the final canvas, including the last canvas of a capped rollout. Do not sum
the absolute score over turns: that would directly pay for prolonging episodes.
Do not add positive reward for token count, tool count, ink coverage or RETHINK.

There is no unique reference image for the native creative task. DINO/DreamSim
similarity to a chosen image would change the objective to reconstruction.
ViDrA's DINO features are inputs to a creativity head, not similarity to a target.

The candidate head is a proxy for crowd-rated creativity, not objective
correctness. Run `scripts/reward_probe.py` on actual rollouts and inspect the
derived images before accepting it. Synthetic added-line controls test
shortcut sensitivity; they are not human ratings. Do not change the reward
or task definition based solely on a score increase without inspecting images.

The complete 480-image released human-rated test split has been re-scored with
the actual head and DINO weights. Correlations agree with the official report.
See [the numerical results and source comparison](RELATED_GRPO.md).

Report out-of-range predictions and failure cases. Do not silently clip them or
drop samples. Lock all scorer and feature-extractor hashes between experimental
arms. Do not select a checkpoint solely by the same reward that it optimizes:
use independent blinded image-quality checks too.

An Actor train/held-out stimulus split is **not** proof that the public reward
model never saw those stimuli. The public adapted head's training corpus and
split must be disclosed separately.

## End-to-end validation boundary

Storage policy: retain the two latest complete, matching-contract checkpoints.
Write the new checkpoint atomically and verify both retained file sets and
SHA-256 hashes before deleting an older same-run save. Preserve all rollout
images, text/tool traces, token IDs, sampled probabilities, metrics and
evaluations. Do not permanently serialize the expanded image-input tensors for
every turn; the exact tensors remain in memory through that update. Store
lossless observed PNGs and processor input hashes instead. A small scoring
fixture is kept inside each checkpoint and covered by its manifest.
This changes storage only, not the data seen by the loss or the algorithm.

The pure-loss tests establish grouping, mask and gradient contracts only.
They do not validate Qwen3.5 multimodal training compatibility, causal token
alignment, model precision, GPU memory, optimizer state, rollout weight sync,
restart equivalence or learning improvement.

Before a long run, require a small real test covering:

1. Native rollout with same-initial-state groups and all terminal canvases.
2. Reward computation and manual inspection of high/low/control images.
3. Exact Student / old-policy / Teacher / reference token-score alignment.
4. Finite backward and optimizer update on the intended trainable parameters.
5. New weights used by rollout, plus a complete checkpoint and resume test.

All five have been exercised in the short GPU integration check described in
[RELATED_GRPO.md](RELATED_GRPO.md#real-model-integration-check). This does not
establish memory at the maximum context, long-run reward robustness or a
50-update improvement over the base model.
