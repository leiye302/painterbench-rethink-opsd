"""In a fresh process, validate model, optimizer, RNG and scoring restoration."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from painter_rl.actor import score_tokens
    from painter_rl.checkpoints import verify_checkpoint, restore_optimizer
    from painter_rl.training import load_actor
    from scripts.reward_probe import atomic_json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--work", type=Path, help="Also check a fresh native rollout using the restored weights")
    args = ap.parse_args()
    contract = json.loads((args.run / "contract.json").read_text())
    manifest = verify_checkpoint(args.checkpoint, contract)
    model = load_actor(args.checkpoint / "model").to("cuda").eval()
    config = contract["config"]
    torch.set_float32_matmul_precision("highest")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 betas=tuple(config["adam_betas"]),
                                 weight_decay=config["weight_decay"], foreach=False)
    update = restore_optimizer(args.checkpoint, optimizer)
    if update != manifest["update"] or not optimizer.state:
        raise RuntimeError("Missing or inconsistent optimizer state")
    probe_path = args.checkpoint / "restore_probe.pt"
    if not probe_path.exists():
        probe_path = args.run / "restore_probe.pt"  # legacy smoke evidence
    probe = torch.load(probe_path, map_location="cpu", weights_only=True)
    with torch.no_grad():
        score = score_tokens(model, probe["inputs"], probe["response"],
                             config["logit_chunk_size"], compute_dtype=config["compute_dtype"])[0].cpu()
    torch.testing.assert_close(score, probe["expected"], atol=1e-6, rtol=1e-6)
    state_steps = sorted({int(s["step"]) for s in optimizer.state.values()})
    if state_steps != [update * config["ppo_epochs"]]:
        raise RuntimeError(f"Wrong optimizer update count: {state_steps}")
    report = {"verified": True, "update": update, "optimizer_steps": state_steps,
              "scoring_max_abs_difference": float((score-probe["expected"]).abs().max()),
              "checkpoint_sha_verified": True, "fresh_process": True}
    if args.work:
        from transformers import AutoProcessor
        from painter_rl.actor import QwenActor
        from painter_rl.upstream import activate_environment
        activate_environment(args.work)
        processor = AutoProcessor.from_pretrained(args.checkpoint / "model", local_files_only=True)
        if "local_hint_method" in contract:
            from painter_rl.rethink import RethinkActor
            actor_type = RethinkActor
        else:
            actor_type = QwenActor
        actor = actor_type(model, processor, config,
                         (ROOT / "configs/static_drawing_skill.txt").read_text())
        turns, canvas, episode = actor.rollout(
            args.run / "post_restore_rollout", contract["train_tasks"][0], config["seed"])
        calls = sum(len(r["tool_calls"]) for r in episode.records)
        changed = sum(r["canvas_before_sha256"] != r["canvas_after_sha256"] for r in episode.records)
        if not calls or not changed:
            raise RuntimeError("Restored model did not execute a drawing operation; inspect the rollout")
        report["restored_native_rollout"] = {"turns": len(turns), "tool_calls": calls,
                                            "canvas_changing_turns": changed,
                                            "generated_tokens": sum(t.response.numel() for t in turns)}
    atomic_json(args.run / "restore_verification.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
