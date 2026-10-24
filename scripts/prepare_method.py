"""Fetch and SHA-verify public exact Route-SFT step520 release assets."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from painter_rl.model_assets import prepare_model, verify_initialization
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/route-sft-step520"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = (verify_initialization if args.verify_only else prepare_model)(args.model_dir)
    print(json.dumps(result), flush=True)
