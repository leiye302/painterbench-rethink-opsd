"""Dependency-free MiMo response handling and bounded, resumable partial retries."""
import hashlib
import json
import time


SCHEMA = "mimo_partial_hints_v2"
FORBIDDEN = ("<RETHINK>", "<tool_call>", "</tool_call>", "<function=", "<think>")


class HintProtocolError(RuntimeError):
    """Only fixed codes and allowlisted metadata; never a raw provider exception."""

    def __init__(self, code, *, retryable=True, details=None):
        super().__init__(code)
        self.code, self.retryable, self.details = code, retryable, details or {}


def decode_envelope(raw):
    """Do not mistake reasoning-only/null/block responses for a JSON hint string."""
    details = {}
    if not isinstance(raw, dict):
        raise HintProtocolError("response_envelope", details=details)
    usage = raw.get("usage")
    if isinstance(usage, dict):
        details["usage"] = {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                            if type(usage.get(k)) is int and usage[k] >= 0}
        tokens = usage.get("completion_tokens_details")
        if isinstance(tokens, dict) and type(tokens.get("reasoning_tokens")) is int:
            details["reasoning_tokens"] = max(0, tokens["reasoning_tokens"])
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise HintProtocolError("response_choices", details=details)
    choice = choices[0]
    finish = choice.get("finish_reason")
    details["finish_reason"] = finish if finish in ("stop", "length", "content_filter", "tool_calls", None) else "other"
    message = choice.get("message")
    if not isinstance(message, dict):
        raise HintProtocolError("response_message", details=details)
    content = message.get("content")
    details["content_type"] = type(content).__name__
    details["reasoning_content_present"] = bool(message.get("reasoning_content"))
    if finish == "length":
        raise HintProtocolError("response_truncated", details=details)
    if finish in ("content_filter", "tool_calls"):
        raise HintProtocolError("response_not_text", details=details)
    if isinstance(content, list):
        if not all(isinstance(v, dict) and v.get("type") == "text" and isinstance(v.get("text"), str)
                   for v in content):
            raise HintProtocolError("response_content_type", details=details)
        content = "".join(v["text"] for v in content)
    if content is None or (isinstance(content, str) and not content.strip()):
        raise HintProtocolError("response_empty_content", details=details)
    if not isinstance(content, str):
        raise HintProtocolError("response_content_type", details=details)
    details["content_chars"] = len(content)
    try:
        return json.loads(content), details
    except (ValueError, TypeError):
        raise HintProtocolError("response_json", details=details) from None


def partial_hints(raw, indices, maximum):
    """Reject only affected items, never assign an ambiguous/foreign hint to a turn."""
    if not isinstance(raw, dict) or set(raw) != {"hints"} or not isinstance(raw["hints"], list):
        raise HintProtocolError("hint_root_schema")
    accepted, errors, seen = {}, {}, set()
    for item in raw["hints"]:
        if not isinstance(item, dict):
            continue
        index = item.get("decision_index")
        # Do not coerce bool, float, string or an unrequested index.
        if type(index) is not int or index not in indices:
            continue
        if index in seen:
            accepted.pop(index, None)
            errors[index] = "hint_duplicate_index"
            continue
        seen.add(index)
        if set(item) != {"decision_index", "hint"}:
            errors[index] = "hint_item_fields"
            continue
        hint = item["hint"]
        if not isinstance(hint, str):
            errors[index] = "hint_content_type"
        elif len(hint.strip()) > maximum:
            errors[index] = "hint_too_long"
        elif any(tag in hint for tag in FORBIDDEN):
            errors[index] = "hint_contains_control_or_action"
        else:
            accepted[index] = hint.strip()  # Empty is a resolved abstention, not a failed request.
    for index in indices - accepted.keys():
        errors.setdefault(index, "hint_missing_or_invalid_index")
    return accepted, errors


def collect_hints(indices, build_payload, send_once, *, maximum, batch_limit, max_retries,
                  state=None, checkpoint=lambda state: None):
    """At most 1+max_retries dispatches per index, including across a partial resume.

    Successful/explicitly empty items are never re-requested. Retry batches shrink
    (4 -> 2 -> 1 by default). Input-budget overflow splits before making a request.
    Persist the attempt before dispatch: an interrupted request can still be billed.
    """
    indices = sorted(indices)
    if state is None:
        state = {"schema": SCHEMA, "indices": indices, "resolved": {}, "errors": {},
                 "tries": {str(i): 0 for i in indices}, "terminal": [], "attempts": [], "done": False}
    validate_state(state, indices, maximum, max_retries)
    if state["done"]:
        return state
    limit = max_retries + 1
    while True:
        pending = [i for i in indices if str(i) not in state["resolved"] and i not in state["terminal"]
                   and state["tries"][str(i)] < limit]
        if not pending:
            break
        retried = min(state["tries"][str(i)] for i in pending)
        width = max(1, batch_limit // (2 ** retried))
        queue = [pending[i:i+width] for i in range(0, len(pending), width)]
        if retried:
            time.sleep(min(2 ** (retried-1), 8))
        while queue:
            selected = queue.pop(0)
            if state.get("abort_code"):
                break
            try:
                payload = build_payload(selected)
            except HintProtocolError as error:
                if error.code == "request_text_budget" and len(selected) > 1:
                    mid = len(selected) // 2
                    queue[0:0] = [selected[:mid], selected[mid:]]
                    continue
                for i in selected:
                    state["errors"][str(i)] = error.code
                    state["terminal"].append(i)
                checkpoint(state)
                continue
            for i in selected:
                state["tries"][str(i)] += 1
            attempt = {"indices": selected, "code": "interrupted_or_in_flight", "time": time.time(),
                       "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}
            state["attempts"].append(attempt)
            checkpoint(state)
            began = time.monotonic()
            details = {}
            try:
                raw, details = send_once(payload)
                accepted, errors = partial_hints(raw, set(selected), maximum)
                attempt.update(code="ok" if not errors else "partial_hint_validation", details=details,
                               item_errors={str(i): code for i, code in errors.items()})
                for i, hint in accepted.items():
                    state["resolved"][str(i)] = hint
                    state["errors"].pop(str(i), None)
                state["errors"].update({str(i): code for i, code in errors.items()})
            except HintProtocolError as error:
                attempt.update(code=error.code, details={**details, **error.details})
                for i in selected:
                    state["errors"][str(i)] = error.code
                    if not error.retryable:
                        state["terminal"].append(i)
                if error.code in ("http_401", "http_403"):
                    state["abort_code"] = error.code
                    for i in pending:
                        if str(i) not in state["resolved"]:
                            state["errors"][str(i)] = error.code
                            if i not in state["terminal"]:
                                state["terminal"].append(i)
            attempt["elapsed_s"] = round(time.monotonic() - began, 3)
            checkpoint(state)
    for i in indices:
        if str(i) not in state["resolved"]:
            state["errors"].setdefault(str(i), "interrupted_request_or_retry_exhausted")
    state["done"] = True
    checkpoint(state)
    return state


def validate_state(state, indices, maximum, max_retries):
    """Validate durable outcomes before reuse; malformed caches must not supply hints."""
    expected = sorted(indices)
    if (not isinstance(state, dict) or state.get("schema") != SCHEMA or state.get("indices") != expected
            or type(state.get("done")) is not bool or not isinstance(state.get("attempts"), list)
            or not isinstance(state.get("terminal"), list)
            or not isinstance(state.get("resolved"), dict) or not isinstance(state.get("errors"), dict)
            or not isinstance(state.get("tries"), dict)):
        raise ValueError("Invalid MiMo outcome cache")
    allowed = {str(i) for i in expected}
    if (set(state["tries"]) != allowed or not set(state["errors"]) <= allowed
            or not set(state["resolved"]) <= allowed
            or any(type(i) is not int or i not in expected for i in state["terminal"])
            or any(type(n) is not int or not 0 <= n <= max_retries+1 for n in state["tries"].values())):
        raise ValueError("Invalid MiMo outcome indices or retry counts")
    raw = {"hints": [{"decision_index": int(i), "hint": hint} for i, hint in state["resolved"].items()]}
    valid, _ = partial_hints(raw, set(expected), maximum)
    if len(valid) != len(state["resolved"]):
        raise ValueError("Invalid cached MiMo hints")
    if state["done"] and (set(state["resolved"]) | set(state["errors"])) != allowed:
        raise ValueError("Incomplete final MiMo outcome cache")
