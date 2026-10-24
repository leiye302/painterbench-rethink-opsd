"""Train unchanged baseline, or Actor-routed Teacher-only hints when opted in."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Pinned local Qwen3.5-4B snapshot")
    parser.add_argument("--work", type=Path, required=True, help="Prepared public environment/loss sources")
    parser.add_argument("--reward-work", type=Path, required=True, help="Prepared official reward assets")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument("--output", type=Path, required=True, help="Fresh run directory")
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--smoke", action="store_true", help="Mark outputs integration-only, not a learning result")
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--method-config", type=Path,
                        help="Enable Actor-routed local hints, e.g. configs/local_hints.json")
    parser.add_argument("--hint-cache", type=Path, default=Path("work/hint_cache"),
                        help="Durable content-addressed hint cache, reuse when resuming")
    args = parser.parse_args()
    from painter_rl.training import train
    train(args)


if __name__ == "__main__":
    main()
