"""Forced sampling: public randomness fixes every sampled token.

The uniform for token ``t`` of turn ``k`` is ``sha256(key, k, t)``. The token is
the inverse-CDF pick of ``softmax(logits / T)`` in vocabulary order, so a trace
has exactly one legal continuation and the validator can check it by teacher
forcing. The key binds window randomness, task, policy checksum and rollout
index, and deliberately excludes the miner hotkey.
"""
from __future__ import annotations

import hashlib
from typing import Any

_FLOAT_BITS = 53


def sampling_key(
    window_randomness: str, family: str, task_seed: int, policy_checksum: str, rollout_index: int,
) -> str:
    material = f"gft-forced-v1|{window_randomness}|{family}|{int(task_seed)}|{policy_checksum}|{int(rollout_index)}"
    return hashlib.sha256(material.encode()).hexdigest()


def forced_uniform(key: str, turn: int, position: int) -> float:
    """Uniform in [0, 1) for one sampled token."""
    digest = hashlib.sha256(f"{key}|{int(turn)}|{int(position)}".encode()).digest()
    return (int.from_bytes(digest[:8], "big") >> (64 - _FLOAT_BITS)) / float(1 << _FLOAT_BITS)


def _distribution(logits: Any, temperature: float) -> tuple[Any, Any]:
    probs = (logits.float() / temperature).softmax(-1)
    return probs, probs.cumsum(-1)


def forced_pick(logits: Any, uniform: float, temperature: float) -> tuple[int, float]:
    """Token and its sampling logprob for one ``[vocab]`` logits row."""
    import torch

    probs, cdf = _distribution(logits, temperature)
    target = torch.tensor([uniform], dtype=cdf.dtype, device=cdf.device) * cdf[-1]
    token = int(torch.searchsorted(cdf, target, right=True).clamp(max=cdf.numel() - 1))
    return token, float(probs[token].log())


def forced_check(logits: Any, tokens: Any, uniforms: Any, temperature: float) -> tuple[Any, Any, Any]:
    """Per position: distance outside the recorded token's interval, its logprob, and the forced pick.

    ``logits`` is ``[n, vocab]``; ``tokens`` and ``uniforms`` are ``[n]``. A
    distance of zero means the recorded token is exactly the forced pick; the
    returned pick is the token the validator itself would sample there.
    """
    import torch

    probs, cdf = _distribution(logits, temperature)
    index = tokens.long().unsqueeze(-1)
    hi = cdf.gather(-1, index).squeeze(-1)
    p = probs.gather(-1, index).squeeze(-1)
    lo = hi - p
    target = uniforms.to(cdf.dtype) * cdf[:, -1]
    distance = (lo - target).clamp(min=0) + (target - hi).clamp(min=0)
    pick = torch.searchsorted(cdf, target.unsqueeze(-1), right=True).squeeze(-1)
    return distance, p.clamp(min=1e-45).log(), pick.clamp(max=cdf.shape[-1] - 1)


def forced_violation(logits: Any, tokens: Any, uniforms: Any, temperature: float) -> tuple[Any, Any]:
    """Distance of each forced uniform outside its token's CDF interval, and the token logprob."""
    distance, logp, _ = forced_check(logits, tokens, uniforms, temperature)
    return distance, logp
