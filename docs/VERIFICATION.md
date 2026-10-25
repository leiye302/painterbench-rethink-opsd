# Verification boundaries

The baseline's earlier real-Qwen GPU integration is not proof of this new
hint extension. This release contains implementation and CPU/native-processor
checks, not a completed Stage-2 run or a demonstrated creativity improvement.

Release checks cover original loss/grouping/token/gradient/checkpoint tests;
Actor ID and native pre-action routing; zero requests on unmarked trajectories;
per-trajectory batching; unchanged Student inputs/responses; no neighboring hint
leak; per-decision fallback; hint schema; native checkpoint processor; model
asset hashes; public allowlist/privacy scan; clean-checkout CPU CI.

The opt-in MiMo v2 protocol additionally has offline fixtures for 11-decision
chunking, valid-neighbor retention, bounded per-item retries, null/block/
truncated responses, explicit-empty abstention, diagnostic redaction and
partial-cache resume. These do not establish a live MiMo acceptance rate.

Still requiring a real runtime-specific method test:

- Live provider semantic hint accuracy, latency, cost and backend reproducibility.
- Full FP32 4B rollout/backward/optimizer/restore integration of this extension.
- Long-context GPU memory and 50-update performance.

No live/paid API request is made by unit tests. A short run can naturally have
zero markers, so untriggered GRPO success alone does not test the live local-hint
branch. Inspect actual per-decision hint/audit files. Internal 24/6 validation is
not the paper's official 30×5 evaluation. Low SFT loss or high trigger rate alone
is not evidence of model quality. Run the documented integration/restore/eval
commands and record source/model/API versions before reporting experiments.
