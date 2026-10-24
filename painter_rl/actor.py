"""Single-process correctness-first Qwen3.5 adapter, without a stale rollout server.

Transformers performs native multimodal sampling and forward/backward. Public
GRPO/SDAR objectives remain in public_losses.py. Nothing trains image pixels.
"""
import base64
import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint


@dataclass
class Turn:
    inputs: dict
    teacher_inputs: dict
    response: torch.Tensor
    sampled_log_probs: torch.Tensor
    old: torch.Tensor | None = None
    teacher: torch.Tensor | None = None
    reference: torch.Tensor | None = None
    advantage: float = 0.


def save_turn_audit(directory, index, messages, inputs, teacher_inputs,
                    response, sampled_log_probs):
    """Keep compact, replayable evidence, not expanded image-input tensors.

    The exact training tensors remain in the in-memory Turn until the update
    finishes. PNGs are lossless and content-addressed within the trajectory.
    Tensor hashes let a later reconstruction check processor equivalence.
    """
    from scripts.reward_probe import atomic_json
    directory = Path(directory)
    recorded = copy.deepcopy(messages)
    for message in recorded:
        if not isinstance(message.get("content"), list):
            continue
        for i, item in enumerate(message["content"]):
            if item.get("type") != "image_url":
                continue
            url = item["image_url"]["url"]
            prefix = "data:image/png;base64,"
            if not url.startswith(prefix):
                raise ValueError("Audit images must be native inline PNGs")
            raw = base64.b64decode(url[len(prefix):], validate=True)
            digest = hashlib.sha256(raw).hexdigest()
            name = f"observed_images/{digest}.png"
            path = directory / name
            path.parent.mkdir(exist_ok=True)
            if path.exists():
                if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError("Observed image cache changed")
            else:
                with path.open("xb") as handle:
                    handle.write(raw)
            message["content"][i] = {"type": "image_file", "path": name, "sha256": digest}

    def describe(values):
        records = {}
        for name, value in values.items():
            value = value.detach().cpu().contiguous()
            raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
            item = {"dtype": str(value.dtype), "shape": list(value.shape),
                    "sha256": hashlib.sha256(raw).hexdigest()}
            if not name.startswith("pixel_values"):
                item["values"] = value.tolist()
            records[name] = item
        return records

    atomic_json(directory / f"turn_{index:03d}.json", {
        "format_version": 1, "expanded_image_tensors_saved": False,
        "messages": recorded, "inputs": describe(inputs),
        "teacher_inputs": describe(teacher_inputs),
        "response_token_ids": response.tolist(),
        "sampled_log_probs": sampled_log_probs.tolist(),
    })


class CaptureSampleLogProbs:
    """Observe the sampler distribution without changing any logits."""
    def __init__(self):
        self.previous = None
        self.values = []

    def __call__(self, input_ids, scores):
        if self.previous is not None:
            self.values.append(self.previous.gather(-1, input_ids[:, -1:]).squeeze(-1).cpu())
        self.previous = scores.float().log_softmax(-1).detach()
        return scores

    def finish(self, last_token):
        self.values.append(self.previous.gather(-1, last_token.reshape(1, 1)).squeeze(-1).cpu())
        return torch.cat(self.values)


def native_eos_ids(tokenizer, model_eos):
    """Respect the chat turn boundary even without generation_config.json."""
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no native chat EOS")
    ids = set(model_eos if isinstance(model_eos, (tuple, list)) else
              ([] if model_eos is None else [model_eos]))
    ids.add(tokenizer.eos_token_id)
    chat_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if tokenizer.convert_ids_to_tokens(chat_end) != "<|im_end|>":
        raise ValueError("Expected the pinned Qwen native chat-end token")
    ids.add(chat_end)
    return sorted(ids)


def extend_inputs(inputs, response, device):
    values = {k: v.to(device) for k, v in inputs.items()}
    p = values["input_ids"].shape[1]
    response = response.to(device).view(1, -1)
    values["input_ids"] = torch.cat([values["input_ids"], response], dim=1)
    values["attention_mask"] = torch.cat([values["attention_mask"], torch.ones_like(response)], dim=1)
    if "mm_token_type_ids" in values:
        values["mm_token_type_ids"] = torch.cat(
            [values["mm_token_type_ids"], torch.zeros_like(response)], dim=1)
    # Never reuse generation's mutable cached positions in a fresh forward.
    if "position_ids" in values:
        raise ValueError("Unexpected precomputed positions; audit the processor version")
    return values, p


def score_tokens(model, inputs, response, chunk_size=64, with_entropy=False, compute_dtype="bfloat16"):
    """Score exact sampled IDs at causal positions p-1 ... p+n-2.

    Chunk only the LM head to avoid a sequence-by-full-vocabulary allocation.
    Checkpointing preserves the exact function; it changes memory, not the loss.
    """
    device = next(model.parameters()).device
    values, p = extend_inputs(inputs, response, device)
    with torch.autocast(device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda" and compute_dtype == "bfloat16"):
        hidden = model.model(**values, use_cache=False, return_dict=True).last_hidden_state
        hidden = hidden[:, p-1:p+response.numel()-1, :]
        tokens = response.to(device).view(1, -1)
        head = model.get_output_embeddings()
        def part(h, y):
            logits = head(h).float()
            normalizer = torch.logsumexp(logits, dim=-1)
            logp = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1) - normalizer
            if with_entropy:
                entropy = normalizer - (torch.softmax(logits, dim=-1) * logits).sum(-1)
            else:
                entropy = torch.zeros_like(logp)
            return logp, entropy
        pieces = []
        for start in range(0, response.numel(), chunk_size):
            h, y = hidden[:, start:start+chunk_size], tokens[:, start:start+chunk_size]
            if torch.is_grad_enabled():
                pieces.append(checkpoint(part, h, y, use_reentrant=False))
            else:
                pieces.append(part(h, y))
    return torch.cat([v[0] for v in pieces], 1), torch.cat([v[1] for v in pieces], 1)


class QwenActor:
    def __init__(self, model, processor, config, skill):
        self.model, self.processor, self.config, self.skill = model, processor, config, skill

    def encode(self, messages, tools, teacher=False, local_hint=None):
        from painter_rl.environment import processor_messages
        messages = copy.deepcopy(messages)
        if teacher:
            messages[0]["content"] += "\n\nTraining-only drawing skill:\n" + self.skill
        if local_hint is not None:
            if not teacher:
                raise ValueError("Local hints are Teacher-only")
            messages[0]["content"] += "\n\nTraining-only local correction hint:\n" + local_hint
        inputs = self.processor.apply_chat_template(
            processor_messages(messages), tools=tools, tokenize=True,
            add_generation_prompt=True, enable_thinking=True,
            return_dict=True, return_tensors="pt")
        inputs = {k: v.cpu() for k, v in inputs.items()}
        if inputs["input_ids"].shape[1] + self.config["max_new_tokens"] > self.config["max_context"]:
            raise RuntimeError("Context budget exceeded; do not silently truncate or drop a group member")
        return inputs

    def capture_route(self, turn, messages, tools, text, directory, index):
        """Baseline no-op; the method subclass records Actor routing only."""
        return None

    @torch.no_grad()
    def generate(self, inputs, greedy=False):
        from transformers import GenerationConfig, LogitsProcessorList
        device = next(self.model.parameters()).device
        values = {k: v.to(device) for k, v in inputs.items()}
        tracker = CaptureSampleLogProbs()
        generation = GenerationConfig(
            do_sample=not greedy, temperature=1., top_p=1., top_k=0,
            repetition_penalty=1., max_new_tokens=self.config["max_new_tokens"],
            eos_token_id=native_eos_ids(self.processor.tokenizer, self.model.generation_config.eos_token_id),
            pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            use_cache=True, return_dict_in_generate=True)
        self.model.eval()
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda" and self.config.get("compute_dtype", "bfloat16") == "bfloat16"):
            generated = self.model.generate(**values, generation_config=generation,
                                            logits_processor=LogitsProcessorList([tracker]))
        response = generated.sequences[0, inputs["input_ids"].shape[1]:]
        sampled = tracker.finish(response[-1])
        if response.numel() != sampled.numel():
            raise AssertionError("Sampling log-prob alignment failed")
        text = self.processor.tokenizer.decode(response, skip_special_tokens=False)
        return response.cpu(), sampled, text

    def rollout(self, directory, stimulus, seed, greedy=False):
        from painter_rl.environment import DrawingEpisode
        from scripts.reward_probe import atomic_json
        device = next(self.model.parameters()).device
        episode = DrawingEpisode(directory, stimulus, self.config["max_new_tokens"])
        atomic_json(Path(directory) / "rollout_config.json", {
            "stimulus": stimulus, "seed": seed, "greedy": greedy,
            "config": self.config, "tools": episode.tools,
            "teacher_skill": self.skill,
        })
        turns, reason = [], "turn_limit"
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for i in range(self.config["max_turns"]):
                messages = episode.messages()
                inputs = self.encode(messages, episode.tools)
                teacher_inputs = self.encode(messages, episode.tools, teacher=True)
                response, sampled, text = self.generate(inputs, greedy)
                turn = Turn(inputs, teacher_inputs, response, sampled)
                turns.append(turn)
                route = self.capture_route(turn, messages, episode.tools, text, directory, i)
                # Training still uses the exact in-memory tensors, not a reconstruction.
                save_turn_audit(directory, i, messages, inputs, teacher_inputs, response, sampled)
                terminal = episode.step(text)
                if route is not None:
                    episode.records[-1]["route"] = route
                atomic_json(Path(directory) / "progress.json",
                            {"turn": i + 1, "response_tokens": response.numel(),
                             "finished": episode.finished})
                if terminal:
                    reason = ("finish_tool" if episode.finished else
                              "token_limit" if response.numel() >= self.config["max_new_tokens"] else
                              "no_tool_call")
                    break
        return turns, episode.finish(reason), episode
