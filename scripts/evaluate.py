"""Evaluate a base model or exported checkpoint on the fixed held-out stimuli."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from transformers import AutoProcessor
    from painter_rl.actor import QwenActor
    from painter_rl.environment import split_stimuli
    from painter_rl.rewards import ViDrAReward
    from painter_rl.training import load_actor, evaluate, validate_config
    from painter_rl.upstream import activate_environment
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--reward-work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    ap.add_argument("--rethink", action="store_true", help="Record native Actor marker IDs; no hint API")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an evaluation")
    config = json.loads(args.config.read_text())
    validate_config(config)
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("highest")
    activate_environment(args.work)
    model = load_actor(args.model).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    if args.rethink:
        from painter_rl.rethink import RethinkActor
        actor_type = RethinkActor
    else:
        actor_type = QwenActor
    actor = actor_type(model, processor, config,
                      (ROOT / "configs/static_drawing_skill.txt").read_text())
    reward = ViDrAReward(args.reward_work, device="cpu")
    result = evaluate(actor, reward, args.output, split_stimuli()[1],
                      config["eval_replicas"], config["seed"]+9_000_000)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
