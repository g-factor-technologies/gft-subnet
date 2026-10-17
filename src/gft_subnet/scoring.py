"""Miner credit = accepted rollout slots; weights follow its moving average.

Contract (validator side):

- The validator assigns each slot (task, policy version, sampling key, token
  limit) before the miner sees it. Every miner runs the same published policy
  with forced sampling, so the miner controls only speed, cost and reliability.
- An accepted slot earns 1 credit, whatever its reward and whether or not the
  rest of its GRPO group arrived. Advantage is used only inside GRPO.
- A rejected slot earns nothing:
  * timeout / transport error: the slot is re-dealt and measured throughput
    drops, so the miner gets less work;
  * ``TraceWarning`` (numerical mismatch of any size): repeated warnings
    quarantine the miner (no new slots) until they age out of the window;
  * ``TraceError`` (protocol violation an honest miner cannot produce: broken
    completion contract, rewritten context, missing trace): moving credit is
    reset to 0 and the miner is quarantined for ``ERROR_QUARANTINE_WINDOWS``.
- Moving credit decays every window, so a miner without work fades out.
"""
from __future__ import annotations

import math

from gft_subnet.chain import MetagraphSnapshot
from gft_subnet.protocol import Episode

#: Weight of the newest window in a miner's moving credit.
CREDIT_EMA_ALPHA = 0.2
#: Windows a miner sits out after a ``TraceError``.
ERROR_QUARANTINE_WINDOWS = 10
#: Windows with a ``TraceWarning`` within the last ``WARNING_HORIZON`` windows
#: that quarantine a miner.
WARNING_LIMIT = 2
WARNING_HORIZON = 5

#: (window_id, credits, slashed hotkeys, warned hotkeys), oldest first.
CreditHistory = list[tuple[int, dict[str, float], set[str], set[str]]]


def episode_eligible(episode: Episode) -> bool:
    return episode.infra_error is None


def classify_trace_failures(episodes: list[Episode]) -> tuple[set[str], set[str]]:
    """``(slashed, warned)`` hotkeys for one window; timeouts and transport errors are neither."""
    slashed = {ep.hotkey for ep in episodes if (ep.infra_error or "").startswith("TraceError:")}
    warned = {ep.hotkey for ep in episodes if (ep.infra_error or "").startswith("TraceWarning:")}
    return slashed, warned - slashed


def miner_credits(episodes: list[Episode]) -> dict[str, float]:
    """One credit per accepted slot, independent of reward and group completeness."""
    credits: dict[str, float] = {}
    for episode in episodes:
        if episode_eligible(episode):
            credits[episode.hotkey] = credits.get(episode.hotkey, 0.0) + 1.0
    return credits


def update_credit_ema(
    previous: dict[str, float], credits: dict[str, float], slashed: set[str], *,
    alpha: float = CREDIT_EMA_ALPHA,
) -> dict[str, float]:
    """Every known miner decays; credit earned this window is added; slashed miners reset to 0."""
    updated = {
        hotkey: (1 - alpha) * previous.get(hotkey, 0.0) + alpha * credits.get(hotkey, 0.0)
        for hotkey in set(previous) | set(credits)
    }
    for hotkey in slashed:
        updated[hotkey] = 0.0
    return updated


def credit_scores(history: CreditHistory, *, alpha: float = CREDIT_EMA_ALPHA) -> dict[str, float]:
    """Replay the ledger's credit history into moving credit (weights are proportional to it)."""
    scores: dict[str, float] = {}
    for _, credits, slashed, _ in history:
        scores = update_credit_ema(scores, credits, slashed, alpha=alpha)
    return scores


def quarantined_miners(history: CreditHistory, *, window_id: int) -> set[str]:
    """Miners that get no new slots in ``window_id``.

    A ``TraceError`` benches a miner for ``ERROR_QUARANTINE_WINDOWS``; at least
    ``WARNING_LIMIT`` windows with a ``TraceWarning`` among the last
    ``WARNING_HORIZON`` windows bench it until the warnings age out.
    """
    benched: set[str] = set()
    warnings: dict[str, int] = {}
    for past, _, slashed, warned in history:
        age = window_id - past
        if age <= 0:
            continue
        if age <= ERROR_QUARANTINE_WINDOWS:
            benched |= slashed
        if age <= WARNING_HORIZON:
            for hotkey in warned:
                warnings[hotkey] = warnings.get(hotkey, 0) + 1
    return benched | {hotkey for hotkey, count in warnings.items() if count >= WARNING_LIMIT}


def scores_to_weights(
    scores: dict[str, float],
    metagraph: MetagraphSnapshot,
) -> tuple[list[int], list[float]]:
    """Map hotkey scores onto metagraph UIDs. Missing miners get 0. Do not dummy-fill."""
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("miner scores must be finite")
    by_hotkey = {neuron.hotkey: neuron.uid for neuron in metagraph.neurons}
    pairs = sorted(
        (
            (by_hotkey[hotkey], max(0.0, float(score)))
            for hotkey, score in scores.items()
            if hotkey in by_hotkey
        ),
        key=lambda item: item[0],
    )
    if not pairs:
        return [], []
    uids = [uid for uid, _ in pairs]
    weights = [weight for _, weight in pairs]
    total = sum(weights)
    if total <= 0:
        return [], []
    return uids, [value / total for value in weights]
