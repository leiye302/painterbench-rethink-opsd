# PainterBench RETHINK + GRPO + OPSD

Stage-2 implementation of **Actor-routed, Teacher-only local hints**, based on
[PainterBench GRPO + OPSD](https://github.com/leiye302/painterbench-grpo-opsd-baseline)
at **392a5d99beb5edada502ca8806190e9ecc49a703**.

Start the Actor from the supplied natural-only **Route-SFT epoch-2 step520**
Qwen3.5-4B checkpoint. Whole trajectories receive the original GRPO loss.
The Teacher gets the original static drawing skill; **only decisions where the
Actor itself emits `<RETHINK>` get an additional local hint**. The hint writer
does not classify or veto routing. The Student and evaluation policy never see
hints or retrospective information.

## Method versus baseline

| Component | Method |
| --- | --- |
| Initialization | Exact released Route-SFT step520, not base or epoch1 |
| Routing | Actor-generated special-token ID at the pre-tool boundary |
| Unmarked decision | Original static-skill Teacher |
| Marked decision | Static skill + local hint, Teacher only |
| Failed/missing/unusable hint | Original static-skill Teacher; keep GRPO and route |
| Teacher | Current Actor, detached scoring of exact sampled response IDs |
| Reference | Frozen Stage-2 initial policy: step520 here, base for base-start control |
| Loss | Original public GRPO + SDAR OPSD + reference KL − entropy; beta=5 |
| Reward/tasks/evaluation/sampling/optimizer | Identical baseline defaults |

No post-action backtracking, replacement Teacher action, Episode Analyzer,
separate frozen OPSD Teacher, or extra Stage-2 marker BCE is introduced.
See [METHOD.md](docs/METHOD.md), [baseline diff](docs/BASELINE_DIFF.md),
and [verification boundaries](docs/VERIFICATION.md).

## Setup and resources

Linux, Python 3.12, Git, and a suitable CUDA GPU are required for training.
The baseline reference stack is Torch 2.13.0+cu130 / torchvision 0.28.0 and
`requirements-runtime.txt`. Full FP32 training is preserved: no silent LoRA,
quantization, precision change or smaller context. Plan **250–300 GiB** durable
storage including model/assets, latest two full saves, atomic third-save
headroom and rollout evidence. A baseline short GPU test used approximately
84 GiB addressable VRAM; full-budget method memory/throughput remains untested.

Training requires your own **vision-capable, OpenAI-compatible HTTPS hint API**.
No key or API charge is needed for preparation, CPU tests or evaluation. First
downloads use GitHub Releases, Hugging Face and Meta's public servers. No private
host, local experiment folder, Stage-1 annotation corpus or private key is needed.

```bash
git clone https://github.com/leiye302/painterbench-rethink-opsd.git
cd painterbench-rethink-opsd
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-runtime.txt

# Native environment, unchanged public GRPO/SDAR functions, and reward.
python scripts/prepare.py --work work
python scripts/reward_probe.py --work work/reward --prepare-only

# Public FP32 tensor files, original tokenizer/processor/template; SHA checked.
python scripts/prepare_method.py --model-dir models/route-sft-step520
python scripts/prepare_method.py --model-dir models/route-sft-step520 --verify-only
python -m unittest discover -s tests -v
```

The [step520 Release](https://github.com/leiye302/painterbench-rethink-opsd/releases/tag/route-sft-step520-v1)
contains losslessly split assets (at most 512 MiB each). The downloader verifies
each part and complete file against `artifacts/route_sft_step520.lock.json`.
It preserves tensor bytes and marker ID **248077**. Do not substitute the base
tokenizer or resize embeddings. This is an inference-weight export sufficient
to initialize Stage 2, **not** a Stage-1 optimizer/RNG resume package. Sanitized
provenance is provided; private annotations and operational logs are not.

## Hint API (training only)

```bash
export HINT_API_BASE="https://your-provider.example/v1"
export HINT_MODEL="your-vision-capable-model-id"
read -rsp "Hint API key: " HINT_API_KEY; echo
export HINT_API_KEY
```

Use your own values; never commit keys. `env.example` lists the variables.
The full prompt and structural validation are shipped in
`configs/local_hint_prompt.txt` and `painter_rl/local_hints.py`.
`configs/local_hints.json` fixes bounded retries, timeouts, concurrency and
budgets. One post-rollout request batches a trajectory's triggered decisions;
up to four trajectory requests run concurrently. Unmarked trajectories make
no call. Exact requests are content-addressed and cached for resume.
Malformed/empty/oversized hints or failures fall back to static-skill Teacher;
no raw error response, request header or key is logged.
Structural checks do **not** establish a provider's semantic hint quality.

## Integration test, then 50 matched updates

```bash
python scripts/train.py \
  --model models/route-sft-step520 --work work --reward-work work/reward \
  --config configs/smoke.json --method-config configs/local_hints.json \
  --hint-cache work/hint_cache --output runs/method-smoke \
  --checkpoints checkpoints/method-smoke \
  --smoke --skip-initial-eval --skip-final-eval

python scripts/verify_restore.py \
  --checkpoint checkpoints/method-smoke/update_0001 \
  --run runs/method-smoke --work work

python scripts/train.py \
  --model models/route-sft-step520 --work work --reward-work work/reward \
  --config configs/baseline.json --method-config configs/local_hints.json \
  --hint-cache work/hint_cache --output runs/method \
  --checkpoints checkpoints/method
```

Defaults remain **50 updates**, 1 group/update, 8 complete trajectories/group,
64 turns, 16,384 new tokens/turn, 32,768 context, temperature 1. Evaluation and
full atomic saves occur every 10 updates; keep the latest two verified saves.
See the [upstream storage/restore details](docs/UPSTREAM_BASELINE.md).

For a same-step520 initialization control, omit `--method-config`. For the
original base-start control, download the pinned base with
`scripts/prepare.py --work work --model-dir models/Qwen3.5-4B` and run without
`--method-config`. Initialization and local hints are distinct factors: report
which comparison you make. The public loss, optimizer and evaluation config
are unchanged in either case.

## Evaluation: no hint API

```bash
python scripts/evaluate.py \
  --model models/route-sft-step520 --rethink \
  --work work --reward-work work/reward --config configs/baseline.json \
  --output runs/eval-step520

python scripts/evaluate.py \
  --model checkpoints/method/update_0050/model --rethink \
  --work work --reward-work work/reward --config configs/baseline.json \
  --output runs/eval-method-final
```

The exact baseline held-out IDs are `is03,is12,is13,is16,ot02,ot03`, two fixed-seed
trajectories each. This **24/6 internal RL split is not an official PainterBench
train/test split** or its paper's all-30-stimuli × 5 full evaluation. Score every
actual final canvas, including capped/failed episodes. `mean_reward` is predicted
creativity, not success rate or human quality ground truth.
`rethink_turns / turns` is decision-level trigger rate, not token fraction;
`valid_pre_action_rethink_turns` reports properly placed/parsed triggers separately.
Keep tasks, seeds, budgets and score implementation identical across arms.

## Resume and validation

Resume a verified retained checkpoint into a fresh run directory, with the same
step520 initialization, code, config, prompt, API model and hint-cache path. Never
resume a partial save or rename an update. The current restored Actor is Teacher;
the reference stays step520. API failure is a per-decision fallback, not a reason
to relabel, restart or tune a trigger threshold. See [METHOD.md](docs/METHOD.md).

For CPU checks, install a compatible CPU Torch and `requirements-audit.txt`, then:

```bash
python -m unittest discover -s tests -v
python scripts/check_release.py
git diff --check
```

CI uses a clean Linux checkout. Tests do not demonstrate a completed Stage-2
GPU experiment or improved drawing quality. Provenance and third-party notices:
[NOTICE.md](NOTICE.md), [sources.lock.json](sources.lock.json),
[original baseline documentation](docs/UPSTREAM_BASELINE.md).
