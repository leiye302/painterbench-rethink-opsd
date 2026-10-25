import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.error

import torch

from painter_rl.actor import QwenActor, Turn
from painter_rl.local_hints import HintClient, attach_hints, build_request, read_config, validated_hints
from painter_rl.rethink import marker_id, route_decision, tensor_digest


class Tokenizer:
    all_special_ids = [99]

    def encode(self, text, add_special_tokens=False):
        pieces = text.split("<RETHINK>")
        ids = []
        for i, piece in enumerate(pieces):
            if i:
                ids.append(99)
            ids.extend(ord(char)+1000 for char in piece)
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(("" if skip_special_tokens else "<RETHINK>") if i == 99 else chr(i-1000)
                       for i in ids)


TOOLS = [{"function": {"name": "draw", "parameters": {"properties": {"x": {"type": "integer"}}}}}]
CALL = "<tool_call><function=draw><parameter=x>12</parameter></function></tool_call>"


def turn_at(directory, index, marked=True):
    tokenizer = Tokenizer()
    text = "a concrete plan</think>\n\n" + ("<RETHINK>" if marked else "") + CALL
    response = torch.tensor(tokenizer.encode(text))
    inputs = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}
    turn = Turn(copy.deepcopy(inputs), copy.deepcopy(inputs), response, torch.zeros_like(response).float())
    turn.route_audit = route_decision(tokenizer, response, TOOLS, 99)
    if marked:
        turn.local_context = {"messages": [{"role": "system", "content": "native instruction"}],
                              "tools": copy.deepcopy(TOOLS), "text": text, "directory": str(directory),
                              "decision_index": index, "student_sha256": tensor_digest(inputs)}
    return turn


class FakeActor:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def encode(self, messages, tools, teacher=False, local_hint=None):
        assert teacher and local_hint
        self.calls.append((copy.deepcopy(messages), local_hint))
        if self.fail:
            raise RuntimeError("budget")
        return {"input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long)}


class FakeClient:
    config = {"max_inflight_episodes": 2}

    def __init__(self, hints=None, failure=False):
        self.result, self.failure, self.calls = hints or {}, failure, 0

    def hints(self, turns):
        self.calls += 1
        if self.failure:
            raise RuntimeError("never write the raw API exception")
        return self.result, "fake", "a"*64


class RouteContracts(unittest.TestCase):
    def decision(self, text):
        tokenizer = Tokenizer()
        return route_decision(tokenizer, tokenizer.encode(text), TOOLS, 99)

    def test_one_special_marker_roundtrip(self):
        self.assertEqual(marker_id(Tokenizer()), 99)
        bad = Tokenizer()
        bad.all_special_ids = []
        with self.assertRaises(ValueError):
            marker_id(bad)

    def test_base_tokenizer_is_rejected_not_silently_extended(self):
        tokenizer = Tokenizer()
        tokenizer.encode = lambda *a, **k: [1, 2, 3]
        with self.assertRaises(ValueError):
            marker_id(tokenizer)

    def test_native_pre_action_marker_eligible(self):
        self.assertTrue(self.decision("plan</think>\n\n<RETHINK>"+CALL)["hint_eligible"])

    def test_no_marker_means_zero_hint(self):
        self.assertFalse(self.decision("plan</think>"+CALL)["actor_emitted_marker"])

    def test_marker_in_reasoning_not_route(self):
        r = self.decision("plan<RETHINK>more plan</think>"+CALL)
        self.assertTrue(r["actor_emitted_marker"])
        self.assertFalse(r["hint_eligible"])

    def test_marker_after_action_not_route(self):
        self.assertFalse(self.decision("plan</think>"+CALL+"<RETHINK>")["hint_eligible"])

    def test_multiple_markers_not_route(self):
        self.assertFalse(self.decision("plan</think><RETHINK><RETHINK>"+CALL)["hint_eligible"])

    def test_unclosed_or_empty_reasoning_not_route(self):
        for text in ("plan<RETHINK>"+CALL, "</think><RETHINK>"+CALL):
            self.assertFalse(self.decision(text)["hint_eligible"])

    def test_incomplete_tool_call_not_hint_eligible(self):
        self.assertFalse(self.decision("plan</think><RETHINK>"+CALL[:-3])["hint_eligible"])

    def test_hint_schema_never_accepts_classifier(self):
        for result in ({"good": True}, {"hints": [{"decision_index": 1, "hint": "x", "bad": True}]}):
            with self.assertRaises(ValueError):
                validated_hints(result, {1}, 100)

    def test_duplicate_and_foreign_indices_fail(self):
        for indices in ([1, 1], [2]):
            with self.assertRaises(ValueError):
                validated_hints({"hints": [{"decision_index": i, "hint": "x"} for i in indices]}, {1}, 100)

    def test_hint_not_whole_action_and_empty_is_safe(self):
        with self.assertRaises(ValueError):
            validated_hints({"hints": [{"decision_index": 1, "hint": CALL}]}, {1}, 1000)
        self.assertEqual(validated_hints({"hints": [{"decision_index": 1, "hint": ""}]}, {1}, 100), {1: ""})

    def test_no_trigger_never_calls_api_or_changes_teacher(self):
        with tempfile.TemporaryDirectory() as folder:
            turn = turn_at(Path(folder), 0, False)
            before = tensor_digest(turn.teacher_inputs)
            client = FakeClient()
            result = attach_hints([[turn]], FakeActor(), client)
            self.assertEqual(client.calls, 0)
            self.assertEqual(result["hints_applied"], 0)
            self.assertEqual(tensor_digest(turn.teacher_inputs), before)

    def test_one_request_per_trajectory_and_student_ids_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            turns = [turn_at(Path(folder), i) for i in (0, 1)]
            student = [tensor_digest(t.inputs) for t in turns]
            responses = [t.response.clone() for t in turns]
            client = FakeClient({0: "Check the left line's coordinate.", 1: "Keep the existing circle intact."})
            actor = FakeActor()
            result = attach_hints([turns], actor, client)
            self.assertEqual(client.calls, 1)
            self.assertEqual(result["hints_applied"], 2)
            for i, turn in enumerate(turns):
                self.assertEqual(tensor_digest(turn.inputs), student[i])
                torch.testing.assert_close(turn.response, responses[i], rtol=0, atol=0)
                self.assertEqual(turn.teacher_inputs["input_ids"].shape[1], 4)
                self.assertIsNone(turn.local_context)

    def test_api_failure_empty_hint_and_overflow_fallback_to_original_teacher(self):
        for client, actor in ((FakeClient(failure=True), FakeActor()),
                              (FakeClient(), FakeActor()),
                              (FakeClient({0: "Check coordinate."}), FakeActor(fail=True))):
            with tempfile.TemporaryDirectory() as folder:
                turn = turn_at(Path(folder), 0)
                before = tensor_digest(turn.teacher_inputs)
                result = attach_hints([[turn]], actor, client)
                self.assertEqual(result["fallbacks"], 1)
                self.assertEqual(tensor_digest(turn.teacher_inputs), before)
                self.assertTrue(turn.route_audit["actor_emitted_marker"])

    def test_hints_cannot_leak_to_neighbor_unmarked_turn(self):
        with tempfile.TemporaryDirectory() as folder:
            turns = [turn_at(Path(folder), 0), turn_at(Path(folder), 1, False)]
            before = tensor_digest(turns[1].teacher_inputs)
            result = attach_hints([turns], FakeActor(), FakeClient({0: "Check x."}))
            self.assertEqual(result["hints_applied"], 1)
            self.assertEqual(tensor_digest(turns[1].teacher_inputs), before)

    def test_hint_request_is_post_rollout_and_missing_trajectory_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            turn = turn_at(directory, 0)
            with self.assertRaises(FileNotFoundError):
                build_request([turn], "prompt", "model", 100, 10000)
            (directory / "trajectory.json").write_text(json.dumps({"turns": [], "finished": True}))
            request = build_request([turn], "prompt", "model", 100, 10000)
            self.assertEqual(request["model"], "model")
            self.assertIn("Completed trajectory", request["messages"][1]["content"][0]["text"])

    def test_student_encode_rejects_hint_and_teacher_does_not_mutate_messages(self):
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return {"input_ids": torch.tensor([[1, 2]])}
        processor = Processor()
        actor = QwenActor(None, processor, {"max_new_tokens": 10, "max_context": 100}, "STATIC")
        messages = [{"role": "system", "content": "ORIGINAL"}]
        with self.assertRaises(ValueError):
            actor.encode(messages, TOOLS, local_hint="LOCAL")
        actor.encode(messages, TOOLS, teacher=True, local_hint="LOCAL")
        self.assertIn("STATIC", processor.messages[0]["content"])
        self.assertIn("LOCAL", processor.messages[0]["content"])
        self.assertEqual(messages, [{"role": "system", "content": "ORIGINAL"}])

    def test_local_config_is_separate_from_unchanged_optimizer_contract(self):
        root = Path(__file__).resolve().parents[1]
        config = read_config(root / "configs/local_hints.json")
        self.assertEqual(config["schema"], "actor_routed_teacher_local_hint_v1")
        baseline = json.loads((root / "configs/baseline.json").read_text())
        self.assertEqual(baseline["updates"], 50)
        self.assertEqual(baseline["gate_beta"], 5)
        self.assertEqual(baseline["sdar_coef"], .01)

    def test_concurrent_identical_requests_use_one_API_call_and_no_cache_race(self):
        root = Path(__file__).resolve().parents[1]
        config = read_config(root/"configs/local_hints.json")
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory/"trajectory.json").write_text(json.dumps({"finished": True, "turns": []}))
            turn = turn_at(directory, 0)
            with patch.dict("os.environ", {"HINT_API_BASE": "https://provider.example/v1",
                                           "HINT_MODEL": "mock-vision", "HINT_API_KEY": "test-only"}):
                client = HintClient(config, directory/"cache")
            def response(payload):
                time.sleep(.02)
                return {"hints": [{"decision_index": 0, "hint": "Check the coordinate."}]}
            with patch.object(client, "_request", side_effect=response) as request:
                with ThreadPoolExecutor(4) as pool:
                    answers = list(pool.map(lambda _: client.hints([turn]), range(4)))
                self.assertEqual(request.call_count, 1)
            self.assertEqual(sum(a[1] == "api" for a in answers), 1)
            self.assertEqual(sum(a[1] == "cache_hit" for a in answers), 3)
            self.assertEqual(len(list((directory/"cache").glob("*.json"))), 1)

    def test_mimo_go_profile_matches_selected_prompt_and_resource_limits(self):
        root = Path(__file__).resolve().parents[1]
        cfg = read_config(root/"configs/local_hints_mimo_go.json")
        generic = read_config(root/"configs/local_hints.json")
        for key in set(generic) - {"provider", "prompt_file"}:
            self.assertEqual(cfg[key], generic[key])
        self.assertEqual(cfg["max_decisions_per_request"], 4)
        prompt = (root/cfg["prompt_file"]).read_text(encoding="utf-8")
        self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(),
                         "d17fb02199b175a8eabae7be456ba8ebc9ed1d08a3a1ce9df8a94c90e3d1d6cf")

    def test_mimo_go_sends_selected_options_and_own_session_without_logging_key(self):
        root = Path(__file__).resolve().parents[1]
        cfg = read_config(root/"configs/local_hints_mimo_go.json")
        env = {"HINT_API_BASE": "https://opencode.ai/inference/go/openai/v1",
               "HINT_MODEL": "mimo-v2.6-pro", "HINT_API_KEY": "test-only",
               "HINT_API_SESSION": "painterbench-method-test"}
        raw = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
            {"hints": [{"decision_index": 0, "hint": "Keep the existing marks intact."}]})}}]}
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory/"trajectory.json").write_text(json.dumps({"finished": True, "turns": []}))
            with patch.dict("os.environ", env):
                client = HintClient(cfg, directory/"cache")
            with patch("painter_rl.local_hints.urllib.request.urlopen",
                       return_value=io.BytesIO(json.dumps(raw).encode())) as send:
                hints, source, digest = client.hints([turn_at(directory, 0)])
            request = send.call_args.args[0]
            payload = json.loads(request.data)
            self.assertEqual(request.get_header("X-opencode-session"), env["HINT_API_SESSION"])
            self.assertEqual(request.get_header("User-agent"), "painterbench-rethink-opsd/1.0")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-only")
            self.assertEqual(payload["model"], "mimo-v2.6-pro")
            self.assertEqual(payload["max_tokens"], 2048)
            self.assertEqual(payload["temperature"], 0)
            self.assertIs(payload["stream"], False)
            self.assertEqual(payload["thinking"], {"type": "disabled"})
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            self.assertEqual(hints[0], "Keep the existing marks intact.")
            self.assertEqual(source, "api")
            self.assertNotIn(env["HINT_API_KEY"], json.dumps(client.contract()))
            self.assertNotIn(env["HINT_API_KEY"], (directory/"cache"/(digest+".json")).read_text())

    def test_mimo_go_rejects_missing_or_unsafe_session_and_wrong_endpoint(self):
        root = Path(__file__).resolve().parents[1]
        cfg = read_config(root/"configs/local_hints_mimo_go.json")
        env = {"HINT_API_BASE": "https://opencode.ai/inference/go/openai/v1",
               "HINT_MODEL": "mimo-v2.6-pro", "HINT_API_KEY": "test-only"}
        with tempfile.TemporaryDirectory() as folder:
            for session in ("", "unsafe\r\nheader", "x"*129):
                with patch.dict("os.environ", dict(env, HINT_API_SESSION=session)):
                    with self.assertRaises(ValueError):
                        HintClient(cfg, Path(folder)/"cache")
            with patch.dict("os.environ", dict(env, HINT_API_SESSION="valid-run",
                                              HINT_API_BASE="https://provider.example/v1")):
                with self.assertRaises(ValueError):
                    HintClient(cfg, Path(folder)/"cache")

    def test_mimo_go_truncated_response_falls_back_and_preserves_route_and_student(self):
        root = Path(__file__).resolve().parents[1]
        cfg = dict(read_config(root/"configs/local_hints_mimo_go.json"), max_retries=0)
        raw = {"choices": [{"finish_reason": "length", "message": {
            "content": '{"hints":[{"decision_index":0,"hint":"Do not use this truncated answer."}]}'}}]}
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory/"trajectory.json").write_text(json.dumps({"finished": True, "turns": []}))
            turn = turn_at(directory, 0)
            before_student, before_teacher = tensor_digest(turn.inputs), tensor_digest(turn.teacher_inputs)
            with patch.dict("os.environ", {"HINT_API_BASE": "https://opencode.ai/inference/go/openai/v1",
                                           "HINT_MODEL": "mimo-v2.6-pro", "HINT_API_KEY": "test-only",
                                           "HINT_API_SESSION": "test-run"}):
                client = HintClient(cfg, directory/"cache")
            with patch("painter_rl.local_hints.urllib.request.urlopen",
                       return_value=io.BytesIO(json.dumps(raw).encode())):
                summary = attach_hints([[turn]], FakeActor(), client)
            self.assertEqual(summary["hints_applied"], 0)
            self.assertEqual(summary["fallbacks"], 1)
            self.assertEqual(tensor_digest(turn.inputs), before_student)
            self.assertEqual(tensor_digest(turn.teacher_inputs), before_teacher)
            self.assertTrue(turn.route_audit["actor_emitted_marker"])
            files = list((directory/"cache").glob("*.json"))
            self.assertEqual(len(files), 1)
            outcome = json.loads(files[0].read_text())
            self.assertEqual(outcome["errors"], {"0": "response_truncated"})
            self.assertEqual(outcome["attempts"][0]["details"]["finish_reason"], "length")

    def test_mimo_partial_validation_retries_only_failed_turn_and_separates_empty(self):
        root = Path(__file__).resolve().parents[1]
        cfg = read_config(root/"configs/local_hints_mimo_go.json")
        seen = []
        def send(payload):
            indices = [int(v["text"].split("=", 1)[1].split(";", 1)[0])
                       for v in payload["messages"][1]["content"]
                       if v.get("type") == "text" and v["text"].startswith("Selected decision_index=")]
            seen.append(indices)
            if len(seen) == 1:
                return {"hints": [{"decision_index": 0, "hint": "Check the coordinate."},
                                  {"decision_index": 1, "hint": ""},
                                  {"decision_index": "2", "hint": "Invalid index."}]}, {}
            return {"hints": [{"decision_index": 2, "hint": "Keep the existing circle intact."}]}, {}
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory/"trajectory.json").write_text(json.dumps({"turns": [], "finished": True}))
            turns = [turn_at(directory, i) for i in range(3)]
            before = [tensor_digest(t.inputs) for t in turns]
            env = {"HINT_API_BASE": "https://opencode.ai/inference/go/openai/v1",
                   "HINT_MODEL": "mimo-v2.6-pro", "HINT_API_KEY": "test-only", "HINT_API_SESSION": "test-run"}
            with patch.dict("os.environ", env):
                client = HintClient(cfg, directory/"cache")
            with patch.object(client, "_mimo_request_once", side_effect=send), patch("painter_rl.hint_protocol.time.sleep"):
                result = attach_hints([turns], FakeActor(), client)
            self.assertEqual(seen, [[0, 1, 2], [2]])
            self.assertEqual(result["hints_applied"], 2)
            self.assertEqual(result["empty_hint_abstentions"], 1)
            self.assertEqual(result["hint_failures"], 0)
            for i, t in enumerate(turns):
                self.assertEqual(tensor_digest(t.inputs), before[i])
            audit = json.loads((directory/"local_hint_001.json").read_text())
            self.assertEqual(audit["fallback_reason"], "no_supported_local_hint")

    def test_mimo_concurrent_cache_and_completed_failure_do_not_reissue(self):
        root = Path(__file__).resolve().parents[1]
        cfg = dict(read_config(root/"configs/local_hints_mimo_go.json"), max_retries=0)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory/"trajectory.json").write_text(json.dumps({"turns": [], "finished": True}))
            turn = turn_at(directory, 0)
            env = {"HINT_API_BASE": "https://opencode.ai/inference/go/openai/v1",
                   "HINT_MODEL": "mimo-v2.6-pro", "HINT_API_KEY": "test-only", "HINT_API_SESSION": "test-run"}
            with patch.dict("os.environ", env):
                client = HintClient(cfg, directory/"cache")
            raw = {"choices": [{"finish_reason": "stop", "message": {"content": None}}]}
            with patch("painter_rl.local_hints.urllib.request.urlopen",
                       side_effect=lambda *a, **k: io.BytesIO(json.dumps(raw).encode())) as send:
                with ThreadPoolExecutor(4) as pool:
                    results = list(pool.map(lambda _: client.hints([turn]), range(4)))
                self.assertEqual(send.call_count, 1)
            self.assertTrue(all(result[0] == {} for result in results))
            self.assertEqual(sum(result[1] == "api" for result in results), 1)
            self.assertEqual(sum(result[1] == "cache_hit" for result in results), 3)
            details = client.diagnostics(results[0][2])[0]
            self.assertEqual(details["hint_failure_code"], "response_empty_content")
            self.assertEqual(details["hint_attempts"], 1)

    def test_mimo_transport_logs_only_fixed_codes_not_raw_http_or_network_errors(self):
        root = Path(__file__).resolve().parents[1]
        cfg = read_config(root/"configs/local_hints_mimo_go.json")
        env = {"HINT_API_BASE": "https://opencode.ai/inference/go/openai/v1",
               "HINT_MODEL": "mimo-v2.6-pro", "HINT_API_KEY": "test-only", "HINT_API_SESSION": "test-run"}
        from painter_rl.hint_protocol import HintProtocolError
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict("os.environ", env):
                client = HintClient(cfg, Path(folder)/"cache")
            errors = [(urllib.error.HTTPError("https://private.invalid", 401, "private-error-body", {}, None), "http_401"),
                      (TimeoutError("private-network-details"), "transport_timeout"),
                      (urllib.error.URLError("private-network-details"), "transport_error")]
            for error, expected in errors:
                with patch("painter_rl.local_hints.urllib.request.urlopen", side_effect=error):
                    with self.assertRaises(HintProtocolError) as caught:
                        client._mimo_request_once({"model": "test"})
                self.assertEqual(caught.exception.code, expected)
                self.assertNotIn("private", str(caught.exception.details))
                self.assertNotIn("private", str(caught.exception))

    def test_generic_provider_contract_and_payload_options_remain_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        cfg = read_config(root/"configs/local_hints.json")
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict("os.environ", {"HINT_API_BASE": "https://provider.example/v1",
                                           "HINT_MODEL": "mock-vision", "HINT_API_KEY": "test-only"}):
                client = HintClient(cfg, Path(folder)/"cache")
            self.assertEqual(client._request_options, {})
            self.assertEqual(client._provider_headers, {})
            self.assertNotIn("request_options", client.contract())
            self.assertNotIn("session_sha256", client.contract())


if __name__ == "__main__":
    unittest.main()
