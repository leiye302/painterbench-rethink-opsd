"""Post-rollout, trajectory-batched hints; only Teacher inputs may change."""
import base64
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from painter_rl.actor import save_turn_audit
from painter_rl.rethink import tensor_digest
from painter_rl.hint_protocol import HintProtocolError, collect_hints, decode_envelope, validate_state
from scripts.reward_probe import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = json.loads((ROOT / "configs/local_hints.json").read_text())
    keys = set(expected)
    if config.get("provider") == "opencode_go":
        keys.add("max_decisions_per_request")
    if set(config) != keys or config["schema"] != expected["schema"]:
        raise ValueError("Invalid local-hint configuration schema")
    if config["provider"] not in ("openai_compatible", "opencode_go") or config["temperature"] != 0:
        raise ValueError("Use the documented deterministic OpenAI-compatible hint writer")
    for key in ("timeout_seconds", "max_inflight_episodes", "max_output_tokens",
                "max_request_text_chars", "max_hint_chars"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("Invalid hint resource limit: " + key)
    if type(config["max_retries"]) is not int or not 0 <= config["max_retries"] <= 5:
        raise ValueError("Invalid hint retry limit")
    if config["provider"] == "opencode_go" and (type(config["max_decisions_per_request"]) is not int
                                               or not 1 <= config["max_decisions_per_request"] <= 4):
        raise ValueError("MiMo requests contain at most four decisions")
    prompt = (ROOT / config["prompt_file"]).resolve()
    if not prompt.is_relative_to(ROOT) or not prompt.is_file():
        raise ValueError("Hint prompt must be shipped within this repository")
    return config


def validated_hints(raw, indices, maximum):
    if not isinstance(raw, dict) or set(raw) != {"hints"} or not isinstance(raw["hints"], list):
        raise ValueError("Invalid hint JSON")
    result = {}
    for item in raw["hints"]:
        if not isinstance(item, dict) or set(item) != {"decision_index", "hint"}:
            raise ValueError("Invalid hint item")
        index, hint = item["decision_index"], item["hint"]
        if type(index) is not int or index not in indices or index in result or not isinstance(hint, str):
            raise ValueError("Unexpected or duplicate hint index")
        hint = hint.strip()
        if len(hint) > maximum or any(tag in hint for tag in (
                "<RETHINK>", "<tool_call>", "</tool_call>", "<function=", "<think>")):
            raise ValueError("Hint is not a short local hint")
        result[index] = hint
    # Missing or empty entries fall back independently; they never cancel routing.
    return result


def build_request(turns, prompt, model, max_tokens, max_chars):
    contexts = [t.local_context for t in turns]
    directory = Path(contexts[0]["directory"])
    episode = json.loads((directory / "trajectory.json").read_text(encoding="utf-8"))
    content = [{"type": "text", "text": "Completed trajectory (privileged, Teacher-only):\n" +
                json.dumps(episode, ensure_ascii=False)}]
    final = directory / "final_painting.png"
    if final.is_file():
        content.append({"type": "text", "text": "Actual completed-rollout canvas (privileged hindsight):"})
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                        base64.b64encode(final.read_bytes()).decode()}})
    for context in contexts:
        content.append({"type": "text", "text":
                        f"Selected decision_index={context['decision_index']}; exact Actor history follows."})
        for message in context["messages"]:
            body = message.get("content", "")
            if isinstance(body, list):
                for item in body:
                    if item.get("type") == "image_url":
                        # Images already present in this exact pre-action Actor input.
                        content.append(copy.deepcopy(item))
                    elif item.get("type") == "text":
                        content.append({"type": "text", "text": item["text"]})
            else:
                content.append({"type": "text", "text": json.dumps(message, ensure_ascii=False)})
        content.append({"type": "text", "text": "Actor's selected response:\n" + context["text"]})
        content.append({"type": "text", "text": "Native tool schemas:\n" +
                        json.dumps(context["tools"], ensure_ascii=False)})
    if sum(len(x["text"]) for x in content if x["type"] == "text") > max_chars:
        raise ValueError("hint_request_text_budget")
    return {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": content}]}


class HintClient:
    def __init__(self, config, cache):
        self.config, self.cache = config, Path(cache)
        self.prompt_path = ROOT / config["prompt_file"]
        self.prompt = self.prompt_path.read_text(encoding="utf-8")
        self.model = os.environ.get(config["model_env"], "").strip()
        self.api_base = os.environ.get(config["api_base_env"], "").rstrip("/")
        self._key = os.environ.get(config["api_key_env"], "")
        url = urllib.parse.urlsplit(self.api_base)
        if not self.model or not self._key or url.scheme != "https" or not url.netloc:
            raise ValueError("Set HINT_MODEL, HINT_API_BASE (HTTPS), and HINT_API_KEY before training")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("API base must not embed credentials, query strings, or fragments")
        self._request_options = {}
        self._provider_headers = {}
        if config["provider"] == "opencode_go":
            if self.api_base != "https://opencode.ai/inference/go/openai/v1":
                raise ValueError("The OpenCode Go profile requires its documented HTTPS endpoint")
            session = os.environ.get("HINT_API_SESSION", "").strip()
            if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", session):
                raise ValueError("Set HINT_API_SESSION to a stable, non-secret run identifier")
            self._provider_headers = {"User-Agent": "painterbench-rethink-opsd/1.0",
                                      "x-opencode-session": session}
            self._request_options = {"stream": False, "thinking": {"type": "disabled"},
                                     "response_format": {"type": "json_object"}}
        self.cache.mkdir(parents=True, exist_ok=True)
        self._cache_guard = threading.Lock()
        self._request_locks = {}

    def contract(self):
        result = {"config": self.config, "prompt_sha256": sha256(self.prompt_path),
                "model": self.model, "api_base_sha256": hashlib.sha256(self.api_base.encode()).hexdigest(),
                "route_source": "Actor generated marker IDs only; no Analyzer classification",
                "Student_hint_visible": False, "evaluation_hint_calls": False,
                "request_unit": "one completed trajectory's triggered decisions",
                "failure_policy": "unchanged static-skill Teacher for that decision"}
        if self.config["provider"] == "opencode_go":
            result.update(request_options=self._request_options,
                          prompt_sha256=hashlib.sha256(self.prompt.encode()).hexdigest(),
                          protocol="mimo_partial_hints_v2",
                          request_unit="up to four triggered decisions from one completed trajectory",
                          failure_policy="per-item bounded retries; keep valid hints; static-skill fallback",
                          user_agent=self._provider_headers["User-Agent"],
                          session_sha256=hashlib.sha256(
                              self._provider_headers["x-opencode-session"].encode()).hexdigest())
        return result

    def _request(self, payload):
        """Original generic-provider transport; Go uses per-item retries below."""
        request = urllib.request.Request(
            self.api_base + "/chat/completions", data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json",
                     **self._provider_headers})
        for attempt in range(self.config["max_retries"] + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config["timeout_seconds"]) as response:
                    raw = json.load(response)
                choice = raw["choices"][0]
                if self.config["provider"] == "opencode_go" and choice.get("finish_reason") == "length":
                    raise ValueError("Truncated hint response")
                return json.loads(choice["message"]["content"])
            except urllib.error.HTTPError as error:
                # Never log the raw error, request headers, URL, or response body.
                if error.code not in (408, 429, 500, 502, 503, 504):
                    raise RuntimeError("hint_http_" + str(error.code)) from None
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                pass
            if attempt < self.config["max_retries"]:
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("hint_request_exhausted")

    def _mimo_request_once(self, payload):
        request = urllib.request.Request(
            self.api_base + "/chat/completions", data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json",
                     **self._provider_headers})
        try:
            with urllib.request.urlopen(request, timeout=self.config["timeout_seconds"]) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as error:
            raise HintProtocolError("http_" + str(error.code),
                                    retryable=error.code in (408, 429, 500, 502, 503, 504),
                                    details={"http_status": error.code}) from None
        except (TimeoutError, OSError, ValueError) as error:
            code = "transport_timeout" if isinstance(error, TimeoutError) else (
                "response_envelope_json" if isinstance(error, ValueError) else "transport_error")
            raise HintProtocolError(code) from None
        value, details = decode_envelope(raw)
        details["http_status"] = 200
        return value, details

    def _mimo_hints(self, turns):
        cfg = self.config
        lookup = {t.local_context["decision_index"]: t for t in turns}
        if len(lookup) != len(turns):
            raise ValueError("Duplicate Actor decision indices")
        contexts = [lookup[i].local_context for i in sorted(lookup)]
        directory = Path(contexts[0]["directory"])
        final = directory / "final_painting.png"
        identity = {"protocol": "mimo_partial_hints_v2", "contract": self.contract(),
                    "contexts": contexts, "trajectory_sha256": sha256(directory/"trajectory.json"),
                    "final_canvas_sha256": sha256(final) if final.is_file() else None}
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        path = self.cache / (digest + ".json")
        with self._cache_guard:
            request_lock = self._request_locks.setdefault(digest, threading.Lock())
        def build(indices):
            try:
                payload = build_request([lookup[i] for i in indices], self.prompt, self.model,
                                        cfg["max_output_tokens"], cfg["max_request_text_chars"])
            except ValueError as error:
                code = "request_text_budget" if error.args == ("hint_request_text_budget",) else "request_build_schema"
                raise HintProtocolError(code, retryable=False) from None
            payload.update(copy.deepcopy(self._request_options))
            return payload
        with request_lock:
            state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
            cached = state is not None and state.get("done") is True
            state = collect_hints(set(lookup), build, self._mimo_request_once,
                                  maximum=cfg["max_hint_chars"], batch_limit=cfg["max_decisions_per_request"],
                                  max_retries=cfg["max_retries"], state=state,
                                  checkpoint=lambda outcome: atomic_json(path, outcome))
        hints = {int(i): hint for i, hint in state["resolved"].items()}
        return hints, "cache_hit" if cached else "api", digest

    def diagnostics(self, digest):
        if self.config["provider"] != "opencode_go":
            return {}
        state = json.loads((self.cache/(digest+".json")).read_text(encoding="utf-8"))
        validate_state(state, state.get("indices", []), self.config["max_hint_chars"], self.config["max_retries"])
        return {int(i): {"hint_attempts": state["tries"][i],
                         "hint_failure_code": state["errors"].get(i),
                         "hint_request_audit": [a for a in state["attempts"] if int(i) in a["indices"]]}
                for i in state["tries"]}

    def hints(self, turns):
        if self.config["provider"] == "opencode_go":
            return self._mimo_hints(turns)
        cfg = self.config
        indices = {t.local_context["decision_index"] for t in turns}
        payload = build_request(turns, self.prompt, self.model, cfg["max_output_tokens"],
                                cfg["max_request_text_chars"])
        payload.update(copy.deepcopy(self._request_options))
        digest = hashlib.sha256(json.dumps({"payload": payload, "contract": self.contract()},
                                          ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        path = self.cache / (digest + ".json")
        with self._cache_guard:
            request_lock = self._request_locks.setdefault(digest, threading.Lock())
        # Serialize duplicate payloads only. Different trajectories still run
        # concurrently; one cache key must not race atomic_json's temporary file.
        with request_lock:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                status = "cache_hit"
            else:
                raw = self._request(payload)
                validated_hints(raw, indices, cfg["max_hint_chars"])
                atomic_json(path, raw)
                status = "api"
        return validated_hints(raw, indices, cfg["max_hint_chars"]), status, digest


def attach_hints(trajectories, actor, client):
    """Run only AFTER all trajectory tool execution; never during evaluation."""
    groups = [[t for t in ts if getattr(t, "route_audit", {}).get("hint_eligible")]
              for ts in trajectories]
    groups = [ts for ts in groups if ts]
    summaries = {"triggered_episodes": len(groups), "hints_applied": 0,
                 "empty_hint_abstentions": 0, "hint_failures": 0, "teacher_context_fallbacks": 0,
                 "eligible_marker_turns": sum(len(ts) for ts in groups),
                 "fallbacks": 0, "actor_marker_turns": sum(
                     bool(getattr(t, "route_audit", {}).get("actor_emitted_marker"))
                     for ts in trajectories for t in ts)}
    summaries["malformed_marker_turns"] = summaries["actor_marker_turns"] - summaries["eligible_marker_turns"]

    def one(ts):
        try:
            return client.hints(ts), None
        except Exception as error:
            # Do not serialize exception strings containing URLs or API data.
            return None, type(error).__name__

    with ThreadPoolExecutor(max_workers=client.config["max_inflight_episodes"]) as pool:
        answers = list(pool.map(one, groups))
    for turns, (answer, error) in zip(groups, answers):
        hints, source, request_sha = answer if answer is not None else ({}, "failed", None)
        details = client.diagnostics(request_sha) if answer is not None and hasattr(client, "diagnostics") else {}
        for turn in turns:
            context = turn.local_context
            audit = dict(turn.route_audit, source=source, request_sha256=request_sha,
                         hint_applied=False, fallback_reason=error)
            audit.update(details.get(context["decision_index"], {}))
            hint = hints.get(context["decision_index"], "")
            if hint:
                try:
                    teacher = actor.encode(context["messages"], context["tools"],
                                           teacher=True, local_hint=hint)
                    turn.teacher_inputs = teacher
                    audit.update(hint_applied=True, hint=hint, fallback_reason=None)
                except RuntimeError:
                    audit["fallback_reason"] = "teacher_context_budget"
                    summaries["teacher_context_fallbacks"] += 1
            elif not error:
                if context["decision_index"] in hints:
                    audit["fallback_reason"] = "no_supported_local_hint"
                    summaries["empty_hint_abstentions"] += 1
                else:
                    audit["fallback_reason"] = audit.get("hint_failure_code") or "empty_or_missing_local_hint"
            if not audit["hint_applied"] and audit["fallback_reason"] not in (
                    "no_supported_local_hint", "teacher_context_budget"):
                summaries["hint_failures"] += 1
            if tensor_digest(turn.inputs) != context["student_sha256"]:
                raise AssertionError("Teacher hint changed Student inputs")
            summaries["hints_applied" if audit["hint_applied"] else "fallbacks"] += 1
            directory = Path(context["directory"])
            save_turn_audit(directory, context["decision_index"], context["messages"],
                            turn.inputs, turn.teacher_inputs, turn.response, turn.sampled_log_probs)
            atomic_json(directory / f"local_hint_{context['decision_index']:03d}.json", audit)
            turn.local_context = None  # Release copied image transports after Teacher encoding.
    return summaries
