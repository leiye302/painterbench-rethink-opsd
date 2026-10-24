# Changes from the pinned baseline

Base: https://github.com/leiye302/painterbench-grpo-opsd-baseline

Commit: `392a5d99beb5edada502ca8806190e9ecc49a703`.

Unchanged: public loss functions, environment/parser/dispatch, reward, upstream
source locks, static drawing skill, baseline/smoke configs, optimizer, grouping,
sampling, score alignment, checkpoint atomicity/retention. File SHA values are
locked and tested in `artifacts/baseline.lock.json`.

Runtime integration points:

1. `actor.py`: optional Teacher hint after static skill, baseline no-op route
   hook. Native sampler and `skip_special_tokens=False` remain unchanged.
2. `rethink.py`: validates the checkpoint tokenizer and records native Actor-only
   routing; does not resize, force generation or classify reasoning.
3. `local_hints.py`: one post-rollout trajectory-batched hint request; only
   selected Teacher inputs change; independent fallback, cache and audits.
4. `training.py`: optional initialization/client contract and one post-rollout,
   pre-scoring hook. Public loss and backward path unchanged. With no
   `--method-config`, hints are entirely off.
5. `evaluate.py`: optional marker reporting; no hint client or API.
6. Public SHA-verified model downloader, prompt/config, tests and docs.

No new reward, route classifier, Analyzer veto, separate Teacher checkpoint,
Oracle action replacement, target-image loss, trajectory filter, or Stage-2
marker BCE is introduced.
