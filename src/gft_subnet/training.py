"""One frozen-policy GRPO step on verified miner traces (TRL GRPOTrainer, one process)."""
from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from gft_subnet.policy import PolicyError, adapter_checksum
from gft_subnet.protocol import Episode
from gft_subnet.verify import _LOGIT_CHUNK, LOGPROB_TOLERANCE, _causal_lm, logprob_drift


def group_relative_advantages(rewards: list[float], *, group_size: int | None = None) -> list[float]:
    if not rewards:
        return []
    size = group_size if group_size is not None else len(rewards)
    if size < 2 or len(rewards) % size:
        raise ValueError("rewards must contain complete groups of at least two")
    advantages = []
    for start in range(0, len(rewards), size):
        group = rewards[start:start + size]
        mean = sum(group) / size
        advantages.extend(value - mean for value in group)
    return advantages


def _group_key(episode: Episode) -> str:
    return repr((episode.policy_version, episode.family, episode.task_seed))


def pack_episode(episode: Episode) -> dict[str, Any]:
    if not episode.turns or not episode.turns[0].prompt_ids:
        raise ValueError("episode has no exact prompt token trace")
    prompt_ids = list(episode.turns[0].prompt_ids)
    prefix = list(prompt_ids)
    completion_ids, logprobs, env_mask = [], [], []
    for turn in episode.turns:
        if turn.prompt_ids[:len(prefix)] != prefix:
            raise ValueError("miner changed the previously generated token context")
        gap = turn.prompt_ids[len(prefix):]
        if not turn.token_ids or len(turn.token_ids) != len(turn.logprobs):
            raise ValueError("episode has an incomplete generation trace")
        if any(not math.isfinite(v) for v in turn.logprobs):
            raise ValueError("non-finite behavior logprobs")
        completion_ids.extend(gap + turn.token_ids)
        logprobs.extend([0.0] * len(gap) + turn.logprobs)
        env_mask.extend([0] * len(gap) + [1] * len(turn.token_ids))
        prefix = turn.prompt_ids + turn.token_ids
    if any(token < 0 for token in prompt_ids + completion_ids):
        raise ValueError("negative token ID")
    if not math.isfinite(episode.reward):
        raise ValueError("non-finite verified reward")
    return dict(prompt_ids=prompt_ids, completion_ids=completion_ids, logprobs=logprobs,
                env_mask=env_mask, tool_mask=list(env_mask), reward=float(episode.reward),
                hotkey=episode.hotkey, episode_id=episode.episode_id, group=_group_key(episode))


def pack_episodes(episodes: list[Episode], *, allow_scripted: bool = False,
                  group_size: int | None = None) -> dict[str, Any]:
    groups = defaultdict(list)
    for episode in episodes:
        groups[_group_key(episode)].append(episode)
    rows = []
    for group in groups.values():
        # Remove entire infrastructure-failed groups, never individual completions.
        eligible = [ep for ep in group if ep.infra_error is None
                    and (allow_scripted or not ep.scripted)]
        if group_size is not None and (len(group) != group_size or len(eligible) != len(group)):
            continue
        rows.extend(pack_episode(ep) for ep in eligible)
    if len({ep.policy_version for ep in episodes if not ep.scripted}) > 1:
        raise ValueError("cannot train mixed policy versions")
    return {
        "n": len(rows),
        **{key: [row[key] for row in rows]
           for key in ("prompt_ids", "completion_ids", "logprobs", "env_mask", "tool_mask")},
        "rewards": [row["reward"] for row in rows],
        "hotkeys": [row["hotkey"] for row in rows],
        "episode_ids": [row["episode_id"] for row in rows],
        "groups": [row["group"] for row in rows],
    }


def informative_episodes(episodes: list[Episode], *, group_size: int) -> list[Episode]:
    """Complete, trainable groups whose rewards are not all equal (others carry no gradient)."""
    groups = defaultdict(list)
    for episode in episodes:
        groups[_group_key(episode)].append(episode)
    kept = []
    for group in groups.values():
        rewards = [ep.reward for ep in group]
        if (len(group) == group_size and max(rewards) != min(rewards)
                and all(ep.infra_error is None and not ep.scripted for ep in group)):
            kept.extend(group)
    return kept


def rollout_func_from_episodes(episodes: list[Episode], *, allow_scripted: bool = False,
                               group_size: int | None = None):
    packed = pack_episodes(episodes, allow_scripted=allow_scripted, group_size=group_size)
    indices = defaultdict(list)
    for index, key in enumerate(packed["groups"]):
        indices[key].append(index)

    def rollout_func(prompts: list[str], trainer: Any) -> dict[str, Any]:
        del trainer
        used = defaultdict(int)
        selected = []
        for key in prompts:
            if key not in indices or used[key] >= len(indices[key]):
                raise ValueError("rollout batch does not match the verified task groups")
            selected.append(indices[key][used[key]])
            used[key] += 1
        for key, count in used.items():
            if count != len(indices[key]):
                raise ValueError("partial GRPO group requested")
        return {
            **{key: [packed[key][i] for i in selected]
               for key in ("prompt_ids", "completion_ids", "logprobs", "env_mask")},
            "verified_reward": [packed["rewards"][i] for i in selected],
        }
    return rollout_func


def install_bittensor_rollout(trainer: Any, episodes: list[Episode], *,
                              allow_scripted: bool = False, group_size: int | None = None) -> Any:
    if getattr(trainer, "tools", None) or getattr(trainer, "environments", None):
        raise ValueError("remote episodes must not be executed again by trainer tools")
    trainer.rollout_func = rollout_func_from_episodes(
        episodes, allow_scripted=allow_scripted, group_size=group_size)
    return trainer


def _adapter_fingerprint(model: Any) -> str:
    import torch
    from peft import get_peft_model_state_dict

    digest = hashlib.sha256()
    for name, value in sorted(get_peft_model_state_dict(model).items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def chunked_completion_logps(
    model: Any, input_ids: Any, attention_mask: Any, logits_to_keep: int, *,
    temperature: float = 1.0, chunk_size: int = _LOGIT_CHUNK,
    compute_entropy: bool = False,
):
    """Score completion tokens without a dense sequence x vocab tensor."""
    import torch

    from gft_subnet._backbone import Backbone

    if logits_to_keep <= 0 or input_ids.size(1) <= logits_to_keep:
        raise ValueError("completion logps need a prompt prefix")
    backbone = Backbone(_causal_lm(model))
    rows = []
    entropies = []
    for index in range(input_ids.size(0)):
        ids = input_ids[index:index + 1]
        mask = attention_mask[index:index + 1]
        hidden, _, full_logits = backbone.run(
            input_ids=ids, attention_mask=mask, use_cache=False)
        start = ids.size(1) - logits_to_keep - 1
        scored = backbone.token_logps(
            hidden, full_logits, slice(start, start + logits_to_keep),
            ids[:, -logits_to_keep:], temperature=temperature,
            chunk_size=chunk_size, return_entropy=compute_entropy,
        )
        if compute_entropy:
            logps, entropy = scored
            rows.append(logps)
            entropies.append(entropy)
        else:
            rows.append(scored)
        del hidden, full_logits, scored
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logps = torch.cat(rows, dim=0)
    entropy = torch.cat(entropies, dim=0) if compute_entropy else None
    return logps, entropy, None


def _validate_behavior_logprobs(trainer: Any, packed: dict[str, Any]) -> None:
    """Reject traces from a different tokenizer, context, checkpoint, or sampling policy."""
    import torch

    trainer.model.eval()
    device = trainer.accelerator.device
    atol, mean_atol = LOGPROB_TOLERANCE["cuda" if getattr(device, "type", "") == "cuda" else "cpu"]
    worst = (0.0, 0.0)
    for prompt, completion, recorded, mask in zip(
        packed["prompt_ids"], packed["completion_ids"], packed["logprobs"], packed["env_mask"],
        strict=True,
    ):
        ids = torch.tensor([prompt + completion], device=device)
        if ids.max().item() >= trainer.model.get_input_embeddings().num_embeddings:
            raise PolicyError("rollout contains tokens outside the model vocabulary")
        with torch.no_grad():
            logps, _, _ = trainer._get_per_token_logps_and_entropies(
                trainer.model, ids, torch.ones_like(ids), len(completion), batch_size=1)
        generated = torch.tensor(mask, dtype=torch.bool, device=device)
        expected = torch.tensor(recorded, device=device, dtype=torch.float32)
        max_abs, mean_abs = logprob_drift(logps[0], expected, generated)
        if max_abs > atol or mean_abs > mean_atol:
            raise PolicyError(
                "behavior logprobs do not match the current training policy "
                f"(max_abs={max_abs:.4f}, mean_abs={mean_abs:.4f})"
            )
        worst = (max(worst[0], max_abs), max(worst[1], mean_abs))
        del logps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Logged so the CUDA tolerances can be recalibrated from real windows.
    print(f"behavior logprobs ok: max_abs={worst[0]:.4f} mean_abs={worst[1]:.4f}", flush=True)


def run_one_grpo_step(
    episodes: list[Episode], *, adapter_dir: str | Path, allow_scripted: bool = False,
    group_size: int | None = None, trainer: Any | None = None,
) -> dict[str, Any]:
    if trainer is None or not getattr(getattr(trainer, "model", None), "peft_config", None):
        raise PolicyError("training requires a real PEFT GRPOTrainer; no fallback is allowed")
    size = group_size or trainer.args.num_generations
    packed = pack_episodes(episodes, allow_scripted=allow_scripted, group_size=size)
    if packed["n"] < size:
        raise ValueError("GRPO needs a complete on-policy group")
    advantages = group_relative_advantages(packed["rewards"], group_size=size)
    if not any(advantages):
        raise ValueError("no within-group learning signal; policy must not advance")
    install_bittensor_rollout(trainer, episodes, allow_scripted=allow_scripted, group_size=size)
    _validate_behavior_logprobs(trainer, packed)
    before = _adapter_fingerprint(trainer.model)
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # From here the in-memory LoRA may equal no published policy (e.g. OOM mid-step).
    trainer.policy_checksum = None
    trainer.train()
    if _adapter_fingerprint(trainer.model) == before:
        raise PolicyError("training did not change the LoRA weights")
    target = Path(tempfile.mkdtemp(prefix="trained-", dir=Path(adapter_dir).parent))
    trainer.model.save_pretrained(str(target), safe_serialization=True)
    if not (target / "adapter_model.safetensors").is_file():
        raise PolicyError("trainer did not save a PEFT adapter")
    return {"packed": packed, "advantages": advantages, "adapter_dir": str(target),
            "checksum": adapter_checksum(target)}


def load_policy_model(*, model_id: str, adapter_dir: str | Path, use_cpu: bool = False) -> Any:
    """Trainable PEFT model for the published policy (base marker gets a fresh LoRA)."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    from gft_subnet.actor import policy_torch_dtype

    use_cuda = (not use_cpu) and torch.cuda.is_available()
    load_kwargs: dict[str, Any] = {"dtype": policy_torch_dtype("cuda" if use_cuda else "cpu")}
    if use_cuda:
        load_kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if use_cuda:
        model = model.to("cuda")
    root = Path(adapter_dir)
    if (root / "adapter_config.json").is_file():
        return PeftModel.from_pretrained(model, str(root), is_trainable=True)
    if (root / "base_policy.json").is_file():
        return get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=4, lora_alpha=8, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        ))
    raise PolicyError("current policy is neither a base policy nor a LoRA adapter")


def make_one_step_trainer(
    *, model_id: str, output_dir: str | Path, episodes: list[Episode],
    group_size: int, adapter_dir: str | Path, temperature: float = 1.0,
    use_cpu: bool = False, model: Any = None, optimizer: Any = None,
):
    """Single trainer process; one optimizer step starts from the miners' current policy.

    Pass the previous window's ``model`` and raw ``optimizer`` together so Adam
    moments carry over; a fresh Adam makes every step ``lr * sign(grad)``.
    """
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    packed = pack_episodes(episodes, group_size=group_size)
    if packed["n"] < group_size:
        raise ValueError("GRPO needs a complete on-policy group")

    use_cuda = (not use_cpu) and torch.cuda.is_available()
    if optimizer is not None and model is None:
        raise ValueError("a reused optimizer needs the model it was built for")
    if model is None:
        model = load_policy_model(model_id=model_id, adapter_dir=adapter_dir, use_cpu=use_cpu)
    if use_cuda:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if getattr(model.config, "use_cache", None) is not None:
            model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    keys = list(dict.fromkeys(packed["groups"]))
    dataset = Dataset.from_dict({"prompt": keys})
    args = GRPOConfig(
        output_dir=str(output_dir), max_steps=1, num_generations=group_size,
        per_device_train_batch_size=packed["n"], gradient_accumulation_steps=1,
        learning_rate=1e-5, lr_scheduler_type="constant",
        max_completion_length=max(map(len, packed["completion_ids"])),
        temperature=temperature, top_k=0, top_p=1.0,
        scale_rewards=False, beta=0.0, loss_type="dr_grpo",
        mask_truncated_completions=False, disable_dropout=True,
        report_to=[], logging_steps=1, save_strategy="no",
        # Model weights are already bf16 on CUDA. Accelerate bf16 mixed-precision
        # copies full LM logits to fp32 and OOMs a 48GB L40S.
        bf16=False, fp16=False, use_cpu=use_cpu,
        **({
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        } if use_cuda else {"gradient_checkpointing": False}),
    )

    def verified_reward(completions, verified_reward, **kwargs):
        del completions, kwargs
        return verified_reward

    install_portable_fused_lm_head()
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=verified_reward,
                          args=args, train_dataset=dataset, optimizers=(optimizer, None))
    if trainer.accelerator.num_processes != 1:
        raise ValueError("Bittensor MVP requires one trainer process")
    if use_cuda:
        import types

        def _chunked_logps(model, input_ids, attention_mask, logits_to_keep, batch_size=None,
                           compute_entropy=False, **kwargs):
            del batch_size, kwargs
            return chunked_completion_logps(
                model, input_ids, attention_mask, logits_to_keep,
                temperature=float(trainer.args.temperature or 1.0),
                compute_entropy=compute_entropy,
            )

        trainer._get_per_token_logps_and_entropies = _chunked_logps

        def _step(self, model, inputs, num_items_in_batch=None):
            import torch
            model.train()
            inputs = self._prepare_inputs(inputs)
            batch = inputs["prompt_ids"].size(0)
            total = None
            for index in range(batch):
                sliced = {
                    key: (
                        value[index:index + 1]
                        if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == batch
                        else value
                    )
                    for key, value in inputs.items()
                }
                loss = self.compute_loss(model, sliced, num_items_in_batch=num_items_in_batch)
                loss = loss / batch
                self.accelerator.backward(loss)
                total = loss.detach() if total is None else total + loss.detach()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return total

        trainer.training_step = types.MethodType(_step, trainer)
    return install_bittensor_rollout(trainer, episodes, group_size=group_size)


# TRL scores GRPO tokens with a Triton kernel and has no other path: without
# Triton (macOS) or on a CPU it cannot run. Off-GPU, compute the same outputs
# from full FP32 logits so CPU tests and local runs work.

def _logprob_from_logits(hidden, weight, bias, targets, temperature, chunk_size,
                         final_logit_softcapping=None, logit_scale=1.0,
                         outputs=("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")):
    """Same outputs as ``trl.kernels.ChunkedLogProbFunction.apply``."""
    import torch

    device = hidden.device.type
    dtype = (torch.get_autocast_dtype(device) if torch.is_autocast_enabled(device)
             else hidden.dtype)
    logits = hidden.to(dtype) @ weight.to(dtype).t()
    if bias is not None:
        logits = logits + bias.to(dtype)
    z = logits.float() * logit_scale
    if final_logit_softcapping is not None:
        z = final_logit_softcapping * torch.tanh(z / final_logit_softcapping)
    z = z / temperature
    log_z = z.logsumexp(-1)
    log_probs = z.gather(-1, targets[:, None]).squeeze(-1) - log_z
    entropy = log_z - (z.softmax(-1) * z).sum(-1) if "entropy" in outputs else None
    with torch.no_grad():
        log_sum_sq = ((2 * z).logsumexp(-1) - 2 * log_z
                      if "log_sum_sq_probs" in outputs else None)
        mean = z.mean(-1) if "mean_logits" in outputs else None
        top1 = z.argmax(-1) == targets if "is_top1" in outputs else None
    return log_probs, entropy, log_sum_sq, mean, top1


class _PortableChunkedLogProb:
    _gft_portable = True

    def __init__(self, kernel):
        self.kernel = kernel

    def apply(self, hidden, *args):
        if self.kernel is not None and hidden.device.type in ("cuda", "xpu"):
            return self.kernel.apply(hidden, *args)
        return _logprob_from_logits(hidden, *args)


def install_portable_fused_lm_head() -> bool:
    """Route off-GPU fused-head scoring to exact logits. Idempotent."""
    try:
        from trl.trainer import utils
    except ImportError:  # pragma: no cover - TRL absent
        return False
    if not hasattr(utils, "_ChunkedLogProbFunction"):
        return False
    if not getattr(utils._ChunkedLogProbFunction, "_gft_portable", False):
        utils._ChunkedLogProbFunction = _PortableChunkedLogProb(utils._ChunkedLogProbFunction)
    return True
