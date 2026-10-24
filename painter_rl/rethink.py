"""Actor-token routing. No Analyzer label, threshold, or forced marker."""
import copy
import hashlib
import json

from painter_rl.actor import QwenActor
from painter_rl.environment import parse_qwen_tools

MARKER = "<RETHINK>"


def marker_id(tokenizer, model=None):
    ids = tokenizer.encode(MARKER, add_special_tokens=False)
    if len(ids) != 1 or tokenizer.decode(ids, skip_special_tokens=False) != MARKER:
        raise ValueError("Use the released Route-SFT tokenizer: marker must round-trip as one token")
    if ids[0] not in tokenizer.all_special_ids:
        raise ValueError("RETHINK must be a registered special token")
    if model is not None:
        for embedding in (model.get_input_embeddings(), model.get_output_embeddings()):
            if embedding.weight.shape[0] <= ids[0]:
                raise ValueError("Marker ID is outside model embeddings; do not resize from a base tokenizer")
    return ids[0]


def route_decision(tokenizer, response, tools, mid):
    ids = response.tolist() if hasattr(response, "tolist") else list(response)
    text = tokenizer.decode(ids, skip_special_tokens=False)
    count = ids.count(mid)
    result = {"marker_token_count": count, "actor_emitted_marker": bool(count),
              "hint_eligible": False, "reason": "no_marker"}
    if count == 0:
        return result
    if count != 1 or text.count(MARKER) != 1:
        return dict(result, reason="multiple_or_decode_mismatched_markers")
    if text.count("</think>") != 1:
        return dict(result, reason="reasoning_boundary_invalid")
    reasoning, body = text.split("</think>", 1)
    if not reasoning.replace("<think>", "").strip():
        return dict(result, reason="empty_reasoning")
    if not body.lstrip().startswith(MARKER):
        return dict(result, reason="marker_not_at_pre_action_boundary")
    action = body.lstrip()[len(MARKER):].lstrip()
    if not action.startswith("<tool_call>"):
        return dict(result, reason="no_native_action_after_marker")
    calls, _, error = parse_qwen_tools(text, tools)
    if error or not calls:
        return dict(result, reason="invalid_native_tool_call")
    # Do not reinterpret a marker later inside a parameter as a route decision.
    return dict(result, hint_eligible=True, reason="actor_pre_action_marker")


def tensor_digest(values):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class RethinkActor(QwenActor):
    def __init__(self, model, processor, config, skill):
        super().__init__(model, processor, config, skill)
        self.marker_token_id = marker_id(processor.tokenizer, model)

    def capture_route(self, turn, messages, tools, text, directory, index):
        route = route_decision(self.processor.tokenizer, turn.response, tools,
                               self.marker_token_id)
        turn.route_audit = route
        if route["hint_eligible"]:
            turn.local_context = {"messages": copy.deepcopy(messages),
                                  "tools": copy.deepcopy(tools), "text": text,
                                  "directory": str(directory), "decision_index": index,
                                  "student_sha256": tensor_digest(turn.inputs)}
        return route
