"""CPU-only native tool dispatch/feedback check, NOT a model-quality test."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from painter_rl.environment import DrawingEpisode
    from painter_rl.upstream import activate_environment
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    activate_environment(args.work)
    episode = DrawingEpisode(args.output, "is01")
    image = episode.canvas.load().convert("RGB")
    point = next((x, y) for y in range(16, 385, 16) for x in range(16, 385, 16)
                 if image.getpixel((x, y)) == (255, 255, 255))
    first = ('Draw one visible dot.</think>\n\n<RETHINK><tool_call><function=draw_dots>'
             '<parameter=dots>' + json.dumps([{"x": point[0], "y": point[1]}]) +
             '</parameter></function></tool_call><|im_end|>')
    episode.step(first)
    record = episode.records[-1]
    if (record["parse_error"] or record["tool_error_count"] or
            record["canvas_before_sha256"] == record["canvas_after_sha256"]):
        raise AssertionError("Native RETHINK-prefixed call did not execute")
    feedback = episode.messages()
    if not any(m.get("role") == "tool" for m in feedback):
        raise AssertionError("Executed tool feedback missing from next context")
    episode.step('The test operation is complete.</think>\n\n<tool_call>'
                 '<function=drawing_finished></function></tool_call><|im_end|>')
    if not episode.finished or episode.records[-1]["tool_error_count"]:
        raise AssertionError("Native finish tool did not execute")
    final = episode.finish("cpu_protocol_test")
    report = {"passed": True, "synthetic_protocol_inputs": True,
              "model_loaded": False, "optimizer_updates": 0, "API_calls": 0,
              "native_RETHINK_prefixed_tool_executed": True,
              "actual_canvas_changed": True, "feedback_in_next_context": True,
              "native_finish_executed": True,
              "final_png_sha256": hashlib.sha256(final.read_bytes()).hexdigest()}
    (args.output / "native_protocol_check.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
