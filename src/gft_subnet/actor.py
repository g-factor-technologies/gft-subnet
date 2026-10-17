"""HF miner: one shared base model, per-gym LoRA slots, no gym code."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from gft_subnet.policy import PolicyError, adapter_checksum
from gft_subnet.protocol import PolicyManifest, TurnRequest, TurnResponse


def policy_torch_dtype(device: str):
    import torch

    if str(device).startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def chat_template_kwargs(model_id: str) -> dict[str, Any]:
    return {"enable_thinking": False} if "Qwen3" in model_id else {}


def turn_prompt_ids(tokenizer: Any, request: TurnRequest, *, model_id: str) -> list[int]:
    """Exact turn context. Miner and validator must build it with this same function."""
    messages = list(request.messages)
    if request.prefix_token_ids:
        messages = [messages[-1]] if messages else [{"role": "user", "content": ""}]
    suffix = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False,
        **chat_template_kwargs(model_id),
    )
    return list(request.prefix_token_ids) + list(suffix)


class TokenTraceVerifier:
    """Validator-side check: the trained tokens are the tokens that earned the reward."""

    def __init__(self, tokenizer: Any, *, model_id: str) -> None:
        self.tokenizer = tokenizer
        self.model_id = model_id

    @classmethod
    def from_pretrained(cls, model_id: str) -> "TokenTraceVerifier":
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(model_id), model_id=model_id)

    def verify(self, request: TurnRequest, response: TurnResponse) -> str:
        """Return the validator-decoded text; never trust the miner's own text field."""
        if list(response.prompt_ids) != turn_prompt_ids(self.tokenizer, request, model_id=self.model_id):
            raise ValueError("miner prompt_ids differ from the validator-built context")
        vocab = len(self.tokenizer)
        if any(token < 0 or token >= vocab for token in response.token_ids):
            raise ValueError("miner returned tokens outside the tokenizer vocabulary")
        return self.tokenizer.decode(response.token_ids, skip_special_tokens=True)


class HuggingFacePolicyActor:
    scripted = False

    def __init__(self, *, hotkey: str, model_id: str, adapter_dir: str | None = None,
                 device: str = "cpu", temperature: float = 0.7,
                 max_new_tokens: int = 512, max_batch: int = 8) -> None:
        self.hotkey = hotkey
        self.model_id = model_id
        self.adapter_dir = adapter_dir
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_batch = max(1, int(max_batch))
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded_checksum: str | None = None
        self._loaded_slot: str | None = None
        self._lock = Lock()

    def load_policy(self, manifest: PolicyManifest) -> None:
        with self._lock:
            if manifest.base_model != self.model_id:
                raise PolicyError("actor base model mismatch")
            if manifest.revision is not None:
                raise PolicyError("revision-specific policies are not supported by this MVP")
            temperature = float(manifest.sampling["temperature"])
            max_tokens = int(manifest.sampling["max_tokens"])
            if temperature <= 0 or max_tokens <= 0:
                raise PolicyError("training requires stochastic sampling and a positive token budget")
            self.load_adapter(manifest.adapter_dir, manifest.checksum, slot=manifest.slot)
            self.temperature, self.max_new_tokens = temperature, max_tokens

    def load_adapter(self, adapter_dir: str, checksum: str | None = None, *, slot: str = "") -> None:
        actual = adapter_checksum(adapter_dir)
        if checksum is not None and actual != checksum:
            raise PolicyError("actor adapter checksum mismatch")
        if actual == self._loaded_checksum:
            return
        root = Path(adapter_dir)
        is_base = (root / "base_policy.json").is_file()
        if is_base:
            marker = json.loads((root / "base_policy.json").read_text())
            if marker.get("base_model") != self.model_id or (root / "VERSION").read_text() != "1":
                raise PolicyError("invalid base policy")
        elif not ((root / "adapter_config.json").is_file()
                  and (root / "adapter_model.safetensors").is_file()):
            raise PolicyError("policy must contain a real PEFT adapter")
        self._ensure_model()
        base = self._model.unload() if hasattr(self._model, "peft_config") else self._model
        if is_base:
            self._model = base
        else:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(base, adapter_dir)
        self._model.eval()
        self.adapter_dir = adapter_dir
        self._loaded_checksum = actual
        self._loaded_slot = slot

    def generate(self, request: TurnRequest) -> TurnResponse:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: list[TurnRequest]) -> list[TurnResponse]:
        """Forced-sampling turns for several episodes, ``max_batch`` rows per GPU call."""
        if any(not request.sampling_key for request in requests):
            raise PolicyError("turn has no forced-sampling key")
        with self._lock:
            if self._loaded_checksum is None:
                raise PolicyError("actor has not loaded a policy")
            responses: list[TurnResponse] = []
            for start in range(0, len(requests), self.max_batch):
                responses.extend(self._generate_rows(requests[start:start + self.max_batch]))
            return responses

    def _generate_rows(self, requests: list[TurnRequest]) -> list[TurnResponse]:
        import torch
        from transformers import GenerationConfig, LogitsProcessorList

        prompts = [turn_prompt_ids(self._tokenizer, request, model_id=self.model_id)
                   for request in requests]
        width = max(map(len, prompts))
        pad = self._tokenizer.pad_token_id
        # Left padding keeps every row's next token at the same column.
        input_ids = torch.tensor([[pad] * (width - len(ids)) + ids for ids in prompts],
                                 device=self._model.device)
        attention_mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in prompts],
                                      device=self._model.device)
        forced = ForcedSamplingProcessor(
            [request.sampling_key for request in requests],
            turns=[request.turn for request in requests],
            prompt_length=width, temperature=self.temperature,
        )
        with torch.inference_mode():
            # Greedy over a one-hot mask: the processor already made the forced pick.
            output = self._model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                logits_processor=LogitsProcessorList([forced]),
                generation_config=GenerationConfig(
                    do_sample=False, max_new_tokens=self.max_new_tokens,
                    eos_token_id=self._tokenizer.eos_token_id, pad_token_id=pad,
                    return_dict_in_generate=True,
                ),
            )
        responses = []
        for row, (request, prompt_ids) in enumerate(zip(requests, prompts, strict=True)):
            completion_ids = output.sequences[row, width:].tolist()
            # Finished rows are padded; keep tokens through the row's first EOS.
            if self._tokenizer.eos_token_id in completion_ids:
                completion_ids = completion_ids[:completion_ids.index(self._tokenizer.eos_token_id) + 1]
            logprobs = forced.logprobs[row][:len(completion_ids)]
            if not completion_ids or len(logprobs) != len(completion_ids):
                raise ValueError("generation did not return a complete token trace")
            responses.append(TurnResponse(
                policy_version=request.policy_version, episode_id=request.episode_id,
                hotkey=self.hotkey, action={},
                text=self._tokenizer.decode(completion_ids, skip_special_tokens=True),
                token_ids=completion_ids, logprobs=logprobs, prompt_ids=prompt_ids,
                scripted=False,
            ))
        return responses

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=policy_torch_dtype(self.device),
        )
        self._model.to(self.device)
        self._model.eval()


class ForcedSamplingProcessor:
    """HF logits processor that replaces sampling with the public forced pick, per batch row."""

    def __init__(self, keys: list[str], *, turns: list[int], prompt_length: int,
                 temperature: float) -> None:
        self.keys = keys
        self.turns = turns
        self.prompt_length = prompt_length
        self.temperature = temperature
        self.logprobs: list[list[float]] = [[] for _ in keys]

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        from gft_subnet.sampling import forced_pick, forced_uniform

        if scores.shape[0] != len(self.keys):
            raise ValueError("forced sampling needs one key per batch row")
        position = input_ids.shape[1] - self.prompt_length
        masked = torch.full_like(scores, float("-inf"))
        for row, (key, turn) in enumerate(zip(self.keys, self.turns, strict=True)):
            token, logprob = forced_pick(
                scores[row], forced_uniform(key, turn, position), self.temperature)
            self.logprobs[row].append(logprob)
            masked[row, token] = 0.0
        return masked
