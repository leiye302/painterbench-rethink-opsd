"""MiMo failure fixtures: no network, provider account, model weights, or GPU."""
import copy
import unittest
from unittest.mock import patch

from painter_rl.hint_protocol import (
    HintProtocolError, collect_hints, decode_envelope, partial_hints, validate_state)


def response(indices):
    return {"hints": [{"decision_index": i, "hint": "Check the visible coordinate."} for i in indices]}


def envelope(content, finish="stop", **extra):
    return {"choices": [{"finish_reason": finish, "message": {"content": content, **extra}}]}


class EnvelopeContracts(unittest.TestCase):
    def test_null_content_has_actionable_code_not_typeerror(self):
        with self.assertRaises(HintProtocolError) as caught:
            decode_envelope(envelope(None, reasoning_content="not a hint"))
        self.assertEqual(caught.exception.code, "response_empty_content")
        self.assertTrue(caught.exception.details["reasoning_content_present"])
        self.assertNotIn("not a hint", str(caught.exception.details))

    def test_truncation_is_distinct_even_if_content_looks_parseable(self):
        with self.assertRaises(HintProtocolError) as caught:
            decode_envelope(envelope('{"hints":[]}', finish="length"))
        self.assertEqual(caught.exception.code, "response_truncated")
        self.assertEqual(caught.exception.details["finish_reason"], "length")

    def test_text_content_blocks_supported_without_using_reasoning(self):
        value, details = decode_envelope(envelope(
            [{"type": "text", "text": '{"hints":'}, {"type": "text", "text": "[]}"}],
            reasoning_content="private hidden text"))
        self.assertEqual(value, {"hints": []})
        self.assertEqual(details["content_type"], "list")
        self.assertNotIn("private hidden text", str(details))

    def test_empty_invalid_and_nontext_envelopes_are_classified(self):
        for raw, code in ((None, "response_envelope"), ({"choices": None}, "response_choices"),
                          ({"choices": [None]}, "response_choices"),
                          ({"choices": [{"message": None}]}, "response_message"),
                          (envelope(7), "response_content_type"),
                          (envelope([{ "type": "image_url"}]), "response_content_type"),
                          (envelope("not JSON"), "response_json"),
                          (envelope("", finish="content_filter"), "response_not_text")):
            with self.subTest(code=code):
                with self.assertRaises(HintProtocolError) as caught:
                    decode_envelope(raw)
                self.assertEqual(caught.exception.code, code)

    def test_usage_whitelisted_not_raw_provider_metadata(self):
        raw = envelope('{"hints":[]}')
        raw["usage"] = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                        "private_account": "never-log-this",
                        "completion_tokens_details": {"reasoning_tokens": 2}}
        _, details = decode_envelope(raw)
        self.assertEqual(details["usage"]["completion_tokens"], 5)
        self.assertEqual(details["reasoning_tokens"], 2)
        self.assertNotIn("never-log-this", str(details))


class PartialValidationContracts(unittest.TestCase):
    def test_one_invalid_item_does_not_discard_ten_valid_neighbors(self):
        raw = response(range(11))
        raw["hints"][10]["hint"] = "x" * 1201
        valid, errors = partial_hints(raw, set(range(11)), 1200)
        self.assertEqual(set(valid), set(range(10)))
        self.assertEqual(errors, {10: "hint_too_long"})

    def test_string_bool_float_foreign_indices_never_assigned(self):
        for index in ("1", True, 1.0, 99):
            with self.subTest(index=index):
                valid, errors = partial_hints(
                    {"hints": [{"decision_index": index, "hint": "Check x."}]}, {1}, 1200)
                self.assertEqual(valid, {})
                self.assertIn(1, errors)

    def test_duplicate_invalidates_only_that_index(self):
        raw = response([0, 1, 0])
        valid, errors = partial_hints(raw, {0, 1}, 1200)
        self.assertEqual(set(valid), {1})
        self.assertEqual(errors, {0: "hint_duplicate_index"})

    def test_extra_item_field_or_action_rejected_without_neighbor_loss(self):
        for change in ({"label": "bad"}, {"hint": "<tool_call>not a hint</tool_call>"}, {"hint": None}):
            raw = response([0, 1])
            raw["hints"][0].update(change)
            valid, errors = partial_hints(raw, {0, 1}, 1200)
            self.assertEqual(set(valid), {1})
            self.assertIn(0, errors)

    def test_empty_hint_is_resolved_abstention(self):
        valid, errors = partial_hints({"hints": [{"decision_index": 0, "hint": "  "}]}, {0}, 1200)
        self.assertEqual(valid, {0: ""})
        self.assertEqual(errors, {})

    def test_root_classifier_not_a_hint_document(self):
        for raw in ({"good": True}, [], {"hints": [], "bad": True}):
            with self.assertRaises(HintProtocolError):
                partial_hints(raw, {0}, 1200)


class BoundedCollectionContracts(unittest.TestCase):
    def collect(self, indices, send, **kwargs):
        with patch("painter_rl.hint_protocol.time.sleep"):
            return collect_hints(set(indices), kwargs.pop("build", lambda ids: {"indices": ids}), send,
                                 maximum=1200, batch_limit=4, max_retries=2, **kwargs)

    def test_eleven_decisions_split_four_four_three(self):
        calls = []
        def send(payload):
            calls.append(payload["indices"])
            return response(payload["indices"]), {}
        state = self.collect(range(11), send)
        self.assertEqual([len(c) for c in calls], [4, 4, 3])
        self.assertEqual(len(state["resolved"]), 11)
        self.assertEqual(state["errors"], {})

    def test_only_failed_index_retried_valid_neighbors_retained(self):
        calls = []
        def send(payload):
            calls.append(payload["indices"])
            raw = response(payload["indices"])
            if len(calls) == 1:
                raw["hints"][-1]["decision_index"] = "3"
            return raw, {"finish_reason": "stop"}
        state = self.collect(range(4), send)
        self.assertEqual(calls, [[0, 1, 2, 3], [3]])
        self.assertEqual(state["tries"], {"0": 1, "1": 1, "2": 1, "3": 2})
        self.assertEqual(state["errors"], {})
        self.assertEqual(len(state["attempts"][0]["request_sha256"]), 64)

    def test_explicit_empty_is_not_retried(self):
        calls = []
        def send(payload):
            calls.append(payload)
            return {"hints": [{"decision_index": 0, "hint": ""}]}, {}
        state = self.collect([0], send)
        self.assertEqual(len(calls), 1)
        self.assertEqual(state["resolved"], {"0": ""})
        self.assertEqual(state["errors"], {})

    def test_malformed_root_retries_shrink_and_are_bounded_per_index(self):
        calls = []
        def send(payload):
            calls.append(payload["indices"])
            return {"wrong_root": True}, {"finish_reason": "stop"}
        state = self.collect(range(4), send)
        self.assertEqual([len(c) for c in calls], [4, 2, 2, 1, 1, 1, 1])
        self.assertTrue(all(n == 3 for n in state["tries"].values()))
        self.assertTrue(all(a["details"]["finish_reason"] == "stop" for a in state["attempts"]))
        self.assertTrue(all(code == "hint_root_schema" for code in state["errors"].values()))

    def test_null_and_transport_errors_leave_typed_final_evidence(self):
        calls = []
        def send(payload):
            calls.append(payload)
            if len(calls) == 1:
                raise HintProtocolError("transport_error")
            raise HintProtocolError("response_empty_content", details={"content_type": "NoneType"})
        state = self.collect([0], send)
        self.assertEqual(len(calls), 3)
        self.assertEqual(state["errors"], {"0": "response_empty_content"})
        self.assertEqual(state["attempts"][-1]["details"]["content_type"], "NoneType")

    def test_auth_failure_stops_remaining_episode_dispatch(self):
        calls = []
        def send(payload):
            calls.append(payload)
            raise HintProtocolError("http_401", retryable=False)
        state = self.collect(range(11), send)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(state["errors"]), 11)
        self.assertTrue(state["done"])

    def test_input_budget_splits_before_dispatch(self):
        calls = []
        def build(ids):
            if len(ids) > 1:
                raise HintProtocolError("request_text_budget", retryable=False)
            return {"indices": ids}
        def send(payload):
            calls.append(payload["indices"])
            return response(payload["indices"]), {}
        state = self.collect(range(4), send, build=build)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(n == 1 for n in state["tries"].values()))

    def test_single_oversized_input_makes_no_api_call(self):
        def build(ids):
            raise HintProtocolError("request_text_budget", retryable=False)
        def send(payload):
            self.fail("No oversized request may be dispatched")
        state = self.collect([0], send, build=build)
        self.assertEqual(state["errors"], {"0": "request_text_budget"})
        self.assertEqual(state["tries"], {"0": 0})

    def test_interrupted_request_is_counted_and_resume_does_not_repeat_success(self):
        saved = []
        def checkpoint(state):
            saved.append(copy.deepcopy(state))
        def interrupted(payload):
            self.assertEqual(saved[-1]["tries"]["0"], 1)
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.collect([0], interrupted, checkpoint=checkpoint)
        calls = []
        def send(payload):
            calls.append(payload)
            return response(payload["indices"]), {}
        state = self.collect([0], send, state=saved[-1], checkpoint=checkpoint)
        self.assertEqual(state["tries"]["0"], 2)
        self.assertEqual(len(calls), 1)
        self.collect([0], lambda _: self.fail("Final cache must make no request"), state=state)

    def test_corrupt_or_incomplete_final_cache_rejected(self):
        state = self.collect([0], lambda p: (response(p["indices"]), {}))
        for mutate in (lambda s: s["resolved"].update({"0": "<RETHINK>"}),
                       lambda s: s["tries"].update({"0": 99}),
                       lambda s: s["resolved"].clear()):
            corrupt = copy.deepcopy(state)
            mutate(corrupt)
            with self.assertRaises(ValueError):
                validate_state(corrupt, [0], 1200, 2)


if __name__ == "__main__":
    unittest.main()
