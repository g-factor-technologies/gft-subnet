"""Trace replay: teacher-force every miner trace and check it against forced sampling.

This is the validator's honesty check. A trace passes when every sampled
token is the forced pick of the current policy within a calibrated numerical
tolerance, the behavior logprobs match, and every turn obeys the completion
contract. Failures are tagged ``TraceError`` (protocol violation) or
``TraceWarning`` (numerical miss) for scoring.
"""
from __future__ import annotations

from typing import Any

from gft_subnet.protocol import Episode


#: LM-head chunks. A competing_cycles 10k-token completion is 2.9GiB of fp32
#: Qwen3 logits; the same head in 128-token windows is ~78MiB. Same trick as
#: ``Backbone.token_logps``.
_LOGIT_CHUNK = 128


def _causal_lm(model: Any) -> Any:
    current = getattr(model, "module", model)
    getter = getattr(current, "get_base_model", None)
    if callable(getter):
        try:
            base = getter()
        except (AttributeError, TypeError):
            base = None
        if base is not None:
            current = base
    return current



#: Behavior-logprob tolerances (nats per generated token) as (max_abs, mean_abs).
#: bf16 batched, left-padded generation vs teacher forcing has rare large
#: single-token outliers (up to 1.96 on Qwen3-14B / L40S, batch 8; mean <= 0.03),
#: so CUDA keeps a loose max but a tight mean; another adapter, sampler
#: temperature or context shifts the mean. Token integrity is enforced by
#: FORCED_SAMPLING_TOLERANCE. CPU fp32 must match almost exactly.
LOGPROB_TOLERANCE = {"cuda": (5.0, 0.1), "cpu": (5e-3, 5e-3)}


def logprob_drift(computed: Any, recorded: Any, mask: Any) -> tuple[float, float]:
    """(max_abs, mean_abs) difference over generated tokens, in float32."""
    import torch

    generated = mask.to(dtype=torch.bool)
    delta = (computed[generated].float() - recorded[generated].float()).abs()
    if not delta.numel():
        return 0.0, 0.0
    return float(delta.max()), float(delta.mean())


def behavior_logprobs_match(computed: Any, recorded: Any, mask: Any, *, atol: float = 5e-3,
                            mean_atol: float | None = None) -> bool:
    """Compare miner traces to trainer logps in float32. CUDA GRPO is bf16."""
    max_abs, mean_abs = logprob_drift(computed, recorded, mask)
    return max_abs <= atol and mean_abs <= (atol if mean_atol is None else mean_atol)


#: Forced-sampling tolerances. ``distance`` is how far the public uniform falls
#: outside the recorded token's CDF interval (0 = exact forced pick). bf16
#: batched KV-cache generation vs teacher forcing shifts CDF boundaries, so
#: honest traces show some off-pick tokens, nearly all at tiny distance.
#: Calibrated on 228 honest Qwen3-14B traces / 71k tokens, L40S, batch 8
#: (testnet, 2026-09-23): 0.4% off-pick tokens, up to 33 in one trace, at most
#: 4 per trace beyond 0.01, largest distance 0.18. A trace passes when every
#: distance <= max_distance, off-pick tokens beyond ``near`` <= max(n, f * tokens)
#: (``far_flips``) and all off-pick tokens <= max(n, f * tokens) (``all_flips``).
FORCED_SAMPLING_TOLERANCE = {
    "cuda": {"max_distance": 0.4, "near": 0.01, "far_flips": (6, 0.01), "all_flips": (8, 0.05)},
    "cpu": {"max_distance": 1e-4, "near": 0.0, "far_flips": (0, 0.0), "all_flips": (0, 0.0)},
}


def completion_violation(token_ids: list[int], *, eos_token_id: int | None,
                         max_tokens: int | None) -> str | None:
    """Completion contract for one turn; ``None`` when the turn ended legally.

    A turn ends at its first end-of-sequence token or after exactly
    ``max_tokens`` tokens. Tokens after EOS, more than ``max_tokens`` tokens, or
    stopping early without EOS are protocol violations an honest miner cannot
    produce (numerical drift never changes where a trace stops).
    """
    if not token_ids:
        return "turn has no generated tokens"
    if max_tokens is not None and len(token_ids) > max_tokens:
        return f"turn has {len(token_ids)} tokens, limit is {max_tokens}"
    if eos_token_id is None:
        return None
    if eos_token_id in token_ids[:-1]:
        return "tokens continue after end-of-sequence"
    if max_tokens is not None and token_ids[-1] != eos_token_id and len(token_ids) != max_tokens:
        return f"turn stopped after {len(token_ids)} of {max_tokens} tokens without end-of-sequence"
    return None


def forced_sampling_ok(distance: Any, tolerance: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    """Apply FORCED_SAMPLING_TOLERANCE to one trace's per-token distances."""
    tokens = distance.numel()
    flips = int((distance > 0).sum())
    far = int((distance > tolerance["near"]).sum())
    stats = {"distance": float(distance.max()) if tokens else 0.0,
             "flips": float(flips), "far_flips": float(far)}
    far_min, far_share = tolerance["far_flips"]
    all_min, all_share = tolerance["all_flips"]
    ok = (stats["distance"] <= tolerance["max_distance"]
          and far <= max(far_min, far_share * tokens)
          and flips <= max(all_min, all_share * tokens))
    return ok, stats


def _trace_positions(episode: Episode) -> tuple[list[int], list[int], list[int], list[float], list[float]]:
    """Full sequence plus (position, token, uniform, recorded logprob) per generated token."""
    from gft_subnet.sampling import forced_uniform

    last = episode.turns[-1]
    ids = list(last.prompt_ids) + list(last.token_ids)
    positions, tokens, uniforms, recorded = [], [], [], []
    for turn in episode.turns:
        start = len(turn.prompt_ids)
        if ids[:start + len(turn.token_ids)] != list(turn.prompt_ids) + list(turn.token_ids):
            raise ValueError("turn context is not a prefix of the final sequence")
        for offset, (token, logprob) in enumerate(zip(turn.token_ids, turn.logprobs, strict=True)):
            positions.append(start + offset)
            tokens.append(int(token))
            uniforms.append(forced_uniform(episode.sampling_key, turn.turn, offset))
            recorded.append(float(logprob))
    return ids, positions, tokens, uniforms, recorded


def verify_episode_traces(
    model: Any, episodes: list[Episode], *, temperature: float, chunk_size: int = _LOGIT_CHUNK,
    eos_token_id: int | None = None, max_tokens: int | None = None,
    tolerance: dict[str, Any] | None = None, logprob_tolerance: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Teacher-force every miner trace; mark forged or off-policy episodes as infra errors.

    Two outcomes for a bad trace:

    - ``TraceError``: a protocol violation an honest miner cannot produce (no
      trace, rewritten context, broken completion contract).
    - ``TraceWarning``: tokens or logprobs outside the numerical tolerance,
      including an end-of-sequence that differs from the forced pick. The size
      of a numerical miss does not prove intent, so it is never escalated here.

    Runs before scoring, so a failed trace is unpaid and never reaches GRPO.
    ``tolerance`` / ``logprob_tolerance`` override the per-device defaults
    (used to test the CUDA tolerances on CPU).
    """
    import torch

    from gft_subnet.sampling import forced_check
    from gft_subnet._backbone import Backbone

    device = next(model.parameters()).device
    kind = "cuda" if device.type == "cuda" else "cpu"
    tolerance = tolerance or FORCED_SAMPLING_TOLERANCE[kind]
    atol, mean_atol = logprob_tolerance or LOGPROB_TOLERANCE[kind]
    if eos_token_id is None and isinstance(getattr(model.config, "eos_token_id", None), int):
        eos_token_id = model.config.eos_token_id
    backbone = Backbone(_causal_lm(model))
    was_training = model.training
    model.eval()
    worst = {"distance": 0.0, "flips": 0.0, "far_flips": 0.0, "eos_mismatch": 0.0,
             "max_abs": 0.0, "mean_abs": 0.0, "checked": 0.0, "rejected": 0.0}
    try:
        for episode in episodes:
            if episode.infra_error is not None:
                continue
            if episode.scripted or not episode.sampling_key or not episode.turns:
                episode.infra_error = "TraceError: episode has no verifiable forced-sampling trace"
                worst["rejected"] += 1
                continue
            try:
                ids, positions, tokens, uniforms, recorded = _trace_positions(episode)
            except ValueError as exc:
                episode.infra_error = f"TraceError: {exc}"
                worst["rejected"] += 1
                continue
            vocab = model.get_input_embeddings().num_embeddings
            if not positions or min(positions) < 1 or max(ids) >= vocab:
                episode.infra_error = "TraceError: trace does not fit the model"
                worst["rejected"] += 1
                continue
            broken = next((reason for reason in (
                completion_violation(list(turn.token_ids), eos_token_id=eos_token_id,
                                     max_tokens=max_tokens)
                for turn in episode.turns) if reason), None)
            if broken:
                episode.infra_error = f"TraceError: {broken}"
                worst["rejected"] += 1
                continue
            distances, logps, picks = [], [], []
            with torch.no_grad():
                input_ids = torch.tensor([ids], device=device)
                hidden, _, full_logits = backbone.run(
                    input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
                for start in range(0, len(positions), chunk_size):
                    index = torch.tensor(
                        [pos - 1 for pos in positions[start:start + chunk_size]], device=device)
                    logits = backbone.logits_at(hidden, full_logits, index)[0]
                    distance, logp, pick = forced_check(
                        logits,
                        torch.tensor(tokens[start:start + chunk_size], device=device),
                        torch.tensor(uniforms[start:start + chunk_size], device=device,
                                     dtype=torch.float64),
                        temperature,
                    )
                    distances.append(distance.float())
                    logps.append(logp.float())
                    picks.append(pick)
                del hidden, full_logits
            distance = torch.cat(distances)
            drift = (torch.cat(logps) - torch.tensor(recorded, device=device)).abs()
            forced_ok, stats = forced_sampling_ok(distance, tolerance)
            # Where a trace stops is not covered by the flip budget: an EOS must be
            # exactly the forced pick, or a miner could stretch or cut its output.
            claimed = torch.tensor(tokens, device=device)
            pick = torch.cat(picks)
            eos_mismatch = 0
            if eos_token_id is not None:
                at_eos = (claimed == eos_token_id) | (pick == eos_token_id)
                eos_mismatch = int((at_eos & (claimed != pick)).sum())
            stats.update(eos_mismatch=float(eos_mismatch), max_abs=float(drift.max()),
                         mean_abs=float(drift.mean()))
            worst.update({key: max(worst[key], value) for key, value in stats.items()})
            worst["checked"] += 1
            if (not forced_ok or eos_mismatch or stats["max_abs"] > atol
                    or stats["mean_abs"] > mean_atol):
                episode.infra_error = (
                    "TraceWarning: tokens are not the forced samples of the current policy "
                    + " ".join(f"{key}={value:.4f}" for key, value in stats.items()))
                worst["rejected"] += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        model.train(was_training)
    # Logged so both tolerances can be recalibrated from real windows.
    print("trace verification: " + " ".join(f"{k}={v:.4f}" for k, v in worst.items()), flush=True)
    return worst
