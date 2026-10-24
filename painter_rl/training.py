"""Native drawing rollouts around unchanged, pinned public loss functions."""
import gc
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from painter_rl.actor import QwenActor, score_tokens
from painter_rl.checkpoints import save_checkpoint, verify_checkpoint, restore_optimizer, prune_checkpoints
from painter_rl.environment import split_stimuli
from painter_rl.public_losses import PublicLosses
from painter_rl.rewards import ViDrAReward
from painter_rl.upstream import activate_environment
from scripts.reward_probe import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def load_actor(path):
    from transformers import AutoModelForImageTextToText
    # FP32 parameters, compute and Adam states in the validated configuration.
    # Neither LoRA nor quantization is silently substituted.
    return AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.float32, attn_implementation="sdpa",
        local_files_only=True)


def validate_config(config):
    required = set(json.loads((ROOT / "configs/baseline.json").read_text()))
    if set(config) != required:
        raise ValueError(f"Unknown/missing config keys: {set(config) ^ required}")
    for name in ("updates", "groups_per_update", "max_turns", "max_new_tokens",
                 "max_context", "ppo_epochs", "save_every", "eval_every",
                 "eval_replicas", "logit_chunk_size", "keep_last_checkpoints"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"Invalid {name}")
    if config["group_size"] < 2 or config["max_new_tokens"] >= config["max_context"]:
        raise ValueError("Invalid group/context budget")
    if config["gate_beta"] != 5:
        raise ValueError("This baseline's fixed SDAR gate_beta is 5")
    if config["compute_dtype"] != "float32":
        raise ValueError("The validated sampler/score path uses FP32; BF16 requires a separate parity audit")
    if config["learning_rate"] <= 0 or config["grad_clip"] <= 0:
        raise ValueError("Invalid optimizer configuration")
    if config["keep_last_checkpoints"] < 2:
        raise ValueError("Keep at least two complete checkpoints")


def base_manifest(path):
    names = ("*.safetensors", "*.json", "*.jinja", "tokenizer.model",
             "vocab.txt", "merges.txt")
    files = {p for pattern in names for p in Path(path).glob(pattern) if p.is_file()}
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Expected a complete local, pinned Hugging Face base model")
    return {p.name: sha256(p) for p in sorted(files)}


def code_manifest():
    paths = [*ROOT.glob("painter_rl/*.py"), *ROOT.glob("scripts/*.py"),
             ROOT / "sources.lock.json"]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def parameter_probe(model):
    # Verify an actual FP32 parameter changed; not a rounded BF16 export only.
    p = model.get_output_embeddings().weight
    return p.detach().reshape(-1)[::max(1, p.numel() // 100_000)].float().cpu().clone()


def evaluate(actor, reward, output, tasks, replicas, seed):
    paths, records = [], []
    for task_index, stimulus in enumerate(tasks):
        for replica in range(replicas):
            directory = output / f"{stimulus}_{replica:02d}"
            _, canvas, episode = actor.rollout(
                directory, stimulus, seed + task_index * 100 + replica, greedy=False)
            paths.append(canvas)
            records.append({"stimulus": stimulus, "replica": replica,
                            "turns": len(episode.records), "finished": episode.finished,
                            "tool_errors": sum(r["tool_error_count"] for r in episode.records),
                            "reasoning_turns": sum(r["reasoning_present"] for r in episode.records),
                            "rethink_turns": sum(bool(r.get("route", {}).get("actor_emitted_marker"))
                                                 for r in episode.records),
                            "valid_pre_action_rethink_turns": sum(bool(r.get("route", {}).get("hint_eligible"))
                                                                  for r in episode.records)})
    reward.to("cuda")
    values = reward.score(paths)
    reward.to("cpu")
    torch.cuda.empty_cache()
    for r, value in zip(records, values):
        r["reward"] = value
    result = {"mean_reward": float(np.mean(values)), "n": len(values),
              "seed": seed, "greedy": False, "records": records}
    atomic_json(output / "evaluation.json", result)
    return result


def train(args):
    from transformers import AutoProcessor
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; this is not a CPU simulation")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    hint_client = None
    method_config = getattr(args, "method_config", None)
    if method_config:
        from painter_rl.local_hints import HintClient, read_config
        from painter_rl.model_assets import verify_initialization
        initialization = verify_initialization(args.model)
        hint_client = HintClient(read_config(method_config), args.hint_cache)
    if args.output.exists():
        raise FileExistsError("Choose a fresh output; --resume reads an existing checkpoint separately")
    args.output.mkdir(parents=True)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("highest")
    environment = activate_environment(args.work)
    train_tasks, val_tasks = split_stimuli()
    skill_path = ROOT / "configs/static_drawing_skill.txt"
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    contract = {"config": config, "base_files": base_manifest(args.model),
                "source_lock": json.loads((ROOT / "sources.lock.json").read_text()),
                "environment_receipt": environment, "code": code_manifest(),
                "teacher_skill_sha256": sha256(skill_path),
                "train_tasks": train_tasks, "eval_tasks": val_tasks,
                "generated_only_mask": True, "teacher_current_actor_detached": True,
                "group_unit": "complete_trajectory_same_stimulus",
                "sampler": {"temperature": 1., "top_p": 1., "top_k": 0,
                            "repetition_penalty": 1.}}
    if hint_client is not None:
        contract["local_hint_method"] = hint_client.contract()
        contract["route_sft_initialization"] = initialization
        from painter_rl.rethink import marker_id
        contract["local_hint_method"]["marker_token_id"] = marker_id(processor.tokenizer)
    atomic_json(args.output / "contract.json", contract)
    start = 0
    if args.resume:
        start = verify_checkpoint(args.resume, contract)["update"]
    model = load_actor(args.resume / "model" if args.resume else args.model).to("cuda")
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    estimated_checkpoint_bytes = int(sum(p.numel()*p.element_size() for p in model.parameters()) * 3.1)
    if shutil.disk_usage(args.checkpoints).free < estimated_checkpoint_bytes:
        raise RuntimeError("Insufficient space for one complete model + Adam checkpoint; "
                           f"need approximately {estimated_checkpoint_bytes / 1024**3:.1f} GiB")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 betas=tuple(config["adam_betas"]),
                                 weight_decay=config["weight_decay"], foreach=False)
    if args.resume:
        if restore_optimizer(args.resume, optimizer) != start:
            raise AssertionError("Optimizer update and checkpoint manifest disagree")
    if hint_client is None:
        actor = QwenActor(model, processor, config, skill_path.read_text())
    else:
        from painter_rl.rethink import RethinkActor
        actor = RethinkActor(model, processor, config, skill_path.read_text())
        assert actor.marker_token_id == contract["local_hint_method"]["marker_token_id"]
    reward = ViDrAReward(args.reward_work, device="cpu")
    losses = PublicLosses(args.work / "public_sources")
    reference = None
    print(json.dumps({"stage": "loaded", "parameters": sum(p.numel() for p in model.parameters()),
                      "start_update": start, "train_stimuli": train_tasks,
                      "eval_stimuli": val_tasks}), flush=True)
    if not args.skip_initial_eval:
        evaluate(actor, reward, args.output / f"eval_{start:04d}", val_tasks,
                 config["eval_replicas"], config["seed"] + 9_000_000)
    for update in range(start, config["updates"]):
        began = time.time()
        trajectories, canvases, group_ids = [], [], []
        for group in range(config["groups_per_update"]):
            index = (update * config["groups_per_update"] + group) % len(train_tasks)
            stimulus = train_tasks[index]
            group_id = f"{update}:{group}:{stimulus}"
            for replica in range(config["group_size"]):
                directory = args.output / f"rollout_{update+1:04d}" / f"{group:03d}_{replica:02d}_{stimulus}"
                turns, canvas, episode = actor.rollout(
                    directory, stimulus, config["seed"] + update * 100_000 + group * 1_000 + replica)
                trajectories.append(turns)
                canvases.append(canvas)
                group_ids.append(group_id)
                print(json.dumps({"stage": "rollout", "update": update+1, "group": group,
                                  "replica": replica, "turns": len(turns),
                                  "tokens": sum(t.response.numel() for t in turns),
                                  "finished": episode.finished}), flush=True)
        if hint_client is not None:
            from painter_rl.local_hints import attach_hints
            hint_summary = attach_hints(trajectories, actor, hint_client)
            atomic_json(args.output / f"hints_{update+1:04d}.json", hint_summary)
            print(json.dumps({"stage": "local_hints", "update": update+1, **hint_summary}), flush=True)
        reward.to("cuda")
        rewards = reward.score(canvases)
        reward.to("cpu")
        torch.cuda.empty_cache()
        advantages = losses.trajectory_advantages(rewards, group_ids, config["group_size"])
        turns = [t for ts in trajectories for t in ts]
        for ts, adv in zip(trajectories, advantages):
            for t in ts:
                t.advantage = float(adv)
        # Score Student and Teacher before any update, using the same weights.
        # Do not regenerate responses under the privileged Teacher prompt.
        model.eval()
        sampling_errors = []
        with torch.no_grad():
            for i, turn in enumerate(turns):
                turn.old = score_tokens(model, turn.inputs, turn.response,
                                        config["logit_chunk_size"],
                                        compute_dtype=config["compute_dtype"])[0].cpu()
                error = (turn.old.flatten() - turn.sampled_log_probs).abs()
                sampling_errors.append(float(error.max()))
                if error.max() > .005:
                    raise RuntimeError(f"Sampler/full-forward logprob mismatch: {float(error.max()):.6g}")
                turn.teacher = score_tokens(model, turn.teacher_inputs, turn.response,
                                            config["logit_chunk_size"],
                                            compute_dtype=config["compute_dtype"])[0].cpu()
                print(json.dumps({"stage": "teacher_scores", "turn": i+1,
                                  "sampling_max_error": sampling_errors[-1]}), flush=True)
        if update == 0:
            # At initialization current Actor == frozen base exactly.
            for turn in turns:
                turn.reference = turn.old.clone()
        else:
            # Optimizer state + Actor remain on GPU. Reference is only resident
            # during no-grad scoring, then goes back to CPU before backward.
            if reference is None:
                reference = load_actor(args.model).eval().requires_grad_(False)
            reference.to("cuda")
            with torch.no_grad():
                for turn in turns:
                    turn.reference = score_tokens(reference, turn.inputs, turn.response,
                                                  config["logit_chunk_size"],
                                                  compute_dtype=config["compute_dtype"])[0].cpu()
            reference.to("cpu")
            torch.cuda.empty_cache()
        token_count = sum(t.response.numel() for t in turns)
        before = parameter_probe(model)
        stats = {}
        for epoch in range(config["ppo_epochs"]):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for i, turn in enumerate(turns):
                student, entropy = score_tokens(model, turn.inputs, turn.response,
                                                config["logit_chunk_size"], with_entropy=True,
                                                compute_dtype=config["compute_dtype"])
                mask = torch.ones_like(student)
                loss, metrics = losses.combined(
                    student, turn.old.to("cuda"), turn.teacher.to("cuda"),
                    turn.reference.to("cuda"), torch.full_like(student, turn.advantage), mask,
                    entropy, **{k: config[k] for k in (
                        "cliprange", "sdar_coef", "ref_kl_coef", "entropy_coef", "gate_beta")})
                scale = turn.response.numel() / token_count
                (loss * scale).backward()
                for k, v in metrics.items():
                    stats[k] = stats.get(k, 0.) + float(v) * scale / config["ppo_epochs"]
                del student, entropy, loss
                print(json.dumps({"stage": "backward", "update": update+1,
                                  "turn": i+1, "total_turns": len(turns)}), flush=True)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"],
                                                 error_if_nonfinite=True)
            if grad == 0:
                raise RuntimeError("All gradients zero")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        changed = float((parameter_probe(model) - before).abs().max())
        if not 0 < changed < float("inf"):
            raise RuntimeError("No finite measured FP32 parameter change")
        metrics = {"update": update+1, "rewards": rewards,
                   "mean_reward": float(np.mean(rewards)), "advantages": advantages.tolist(),
                   "generated_tokens": token_count, "turns": len(turns),
                   "gradient_norm": float(grad), "parameter_probe_max_change": changed,
                   "sampling_max_abs_error": max(sampling_errors), **stats,
                   "seconds_before_checkpoint_eval": time.time()-began}
        atomic_json(args.output / f"metrics_{update+1:04d}.json", metrics)
        print(json.dumps(metrics), flush=True)
        model.eval()
        if (update+1) % config["save_every"] == 0 or update+1 == config["updates"]:
            probe = turns[0]
            with torch.no_grad():
                expected = score_tokens(model, probe.inputs, probe.response,
                                        config["logit_chunk_size"],
                                        compute_dtype=config["compute_dtype"])[0].cpu()
            destination = args.checkpoints / f"update_{update+1:04d}"
            save_checkpoint(destination, model, processor, optimizer, update+1, contract,
                            restore_probe={"inputs": probe.inputs, "response": probe.response,
                                           "expected": expected})
            atomic_json(args.output / f"checkpoint_{update+1:04d}.json",
                        {"path": str(destination), "complete": True})
            retention = prune_checkpoints(args.checkpoints, contract,
                                          config["keep_last_checkpoints"])
            atomic_json(args.output / f"retention_{update+1:04d}.json", retention)
            print(json.dumps({"stage": "checkpoint_complete", "update": update+1}), flush=True)
            print(json.dumps({"stage": "checkpoint_retention", **retention}), flush=True)
        if (update+1) % config["eval_every"] == 0 or (update+1 == config["updates"] and not args.skip_final_eval):
            evaluate(actor, reward, args.output / f"eval_{update+1:04d}", val_tasks,
                     config["eval_replicas"], config["seed"]+9_000_000)
        del trajectories, turns
        gc.collect()
    atomic_json(args.output / "complete.json", {"updates": config["updates"],
                                               "formal_training_result": not args.smoke})
