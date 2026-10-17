"""One Bittensor window: validator tasks, rollouts dealt to miners, trace replay, slot credit, GRPO, weights."""
from __future__ import annotations

import json
import random
import secrets
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gft_subnet.chain import ChainClient
from gft_subnet.gym import Gym
from gft_subnet.gym_loop import MinerClient, RolloutSpec, run_rollouts
from gft_subnet.ledger import Ledger
from gft_subnet.policy import PolicyError, PolicyRegistry
from gft_subnet.protocol import Episode, PolicyManifest
from gft_subnet.sampling import sampling_key
from gft_subnet.scoring import (
    classify_trace_failures,
    credit_scores,
    miner_credits,
    quarantined_miners,
    scores_to_weights,
)


@dataclass
class WindowResult:
    window_id: int
    policy_version: int
    episodes: list[Episode]
    scores: dict[str, float]
    uids: list[int]
    weights: list[float]
    block: int
    next_policy: PolicyManifest | None = None
    packed: dict[str, Any] | None = None
    dummy_weights: bool = False
    extras: dict[str, Any] = field(default_factory=dict)
    trainer: Any = None


def ack_miners(policy: PolicyRegistry, miners: list[MinerClient]) -> PolicyManifest:
    manifest = policy.manifest()
    for miner in miners:
        miner.ack_policy(manifest.version, manifest.checksum, gym_id=manifest.gym_id)
        policy.ack(miner.hotkey, manifest.version, manifest.checksum, gym_id=manifest.gym_id)
    if not policy.can_advance([miner.hotkey for miner in miners], gym_id=manifest.gym_id):
        raise PolicyError("not every miner acked the current policy")
    return manifest


#: Extra rounds that hand rollouts of failed miners to healthy ones.
REASSIGN_ROUNDS = 1
#: Weight of the newest window in a miner's measured throughput (tokens/s).
THROUGHPUT_EMA_ALPHA = 0.3


def choose_window_tasks(
    *, families: list[str], count: int, window_id: int, used_tasks: set[tuple[str, int]],
) -> list[tuple[str, int]]:
    """Validator-chosen tasks: families in rotation, the next unplayed seed of each."""
    if not families or count < 1:
        raise ValueError("need at least one family and one task per window")
    taken = set(used_tasks)
    tasks = []
    for offset in range(count):
        family = families[(window_id * count + offset) % len(families)]
        seed = 1 + max((seed for name, seed in taken if name == family), default=0)
        taken.add((family, seed))
        tasks.append((family, seed))
    return tasks


def assign_rollouts(
    specs: list[RolloutSpec], miners: list[MinerClient], throughput: dict[str, float],
    rng: random.Random,
) -> dict[str, list[RolloutSpec]]:
    """Shuffle rollouts and split them in proportion to measured throughput.

    Unmeasured miners count as the average of measured ones. Largest-remainder
    rounding with random tie-breaks; a slow miner may get none when rollouts
    are scarce.
    """
    if not miners or not specs:
        return {}
    known = [throughput[m.hotkey] for m in miners if throughput.get(m.hotkey, 0.0) > 0]
    default = sum(known) / len(known) if known else 1.0
    capacity = [throughput.get(m.hotkey) or default for m in miners]
    exact = [len(specs) * value / sum(capacity) for value in capacity]
    quotas = [int(value) for value in exact]
    ties = [rng.random() for _ in miners]
    order = sorted(range(len(miners)), key=lambda i: (exact[i] - quotas[i], ties[i]), reverse=True)
    for index in order[:len(specs) - sum(quotas)]:
        quotas[index] += 1
    shuffled = list(specs)
    rng.shuffle(shuffled)
    assignment: dict[str, list[RolloutSpec]] = {}
    start = 0
    for miner, quota in zip(miners, quotas, strict=True):
        if quota:
            assignment[miner.hotkey] = shuffled[start:start + quota]
            start += quota
    return assignment


def collect_rollouts(
    *,
    gym: Gym,
    miners: list[MinerClient],
    policy: PolicyRegistry,
    assignment: dict[str, list[RolloutSpec]],
    turn_timeout_seconds: float = 30.0,
    verifier: Any = None,
) -> tuple[list[tuple[RolloutSpec, Episode]], dict[str, float]]:
    """Every assigned miner plays its rollouts (batched per turn); miners run in parallel.

    Returns (spec, episode) pairs and tokens/second per miner that finished
    cleanly. A miner-side protocol violation fails that miner's rollouts
    instead of aborting the window.
    """
    active = [miner for miner in miners if assignment.get(miner.hotkey)]

    def play(miner: MinerClient) -> tuple[list[tuple[RolloutSpec, Episode]], float | None]:
        specs = assignment[miner.hotkey]
        started = time.monotonic()
        try:
            episodes = run_rollouts(
                gym=gym, miner=miner, rollouts=specs, policy=policy,
                turn_timeout_seconds=turn_timeout_seconds, verifier=verifier,
            )
        except PolicyError as exc:
            episodes = [_failed_episode(miner.hotkey, spec, policy, f"PolicyError: {exc}")
                        for spec in specs]
        elapsed = time.monotonic() - started
        tokens = sum(len(turn.token_ids) for episode in episodes for turn in episode.turns)
        clean = all(episode.infra_error is None for episode in episodes)
        return list(zip(specs, episodes, strict=True)), (tokens / elapsed if clean and elapsed > 0 else None)

    if not active:
        return [], {}
    with ThreadPoolExecutor(max_workers=len(active)) as pool:
        results = list(pool.map(play, active))
    pairs = [pair for played, _ in results for pair in played]
    rates = {miner.hotkey: rate for miner, (_, rate) in zip(active, results, strict=True)
             if rate is not None}
    return pairs, rates


def _failed_episode(hotkey: str, spec: RolloutSpec, policy: PolicyRegistry, error: str) -> Episode:
    return Episode(
        episode_id=f"failed-{spec.family}-{spec.task_seed}-{spec.index}-{hotkey}-{secrets.token_hex(4)}",
        hotkey=hotkey, family=spec.family, task_seed=spec.task_seed, generation_seed=spec.index,
        policy_version=policy.current_version, turns=[], reward=0.0, complete=False,
        infra_error=error, sampling_key=spec.sampling_key,
    )


def run_window(
    *,
    gym: Gym,
    chain: ChainClient,
    validator_hotkey: str,
    miners: list[MinerClient],
    policy: PolicyRegistry,
    ledger: Ledger,
    families: list[str] | None = None,
    group_size: int = 8,
    tasks_per_window: int = 2,
    dummy_weights: bool = False,
    train: bool = True,
    turn_timeout_seconds: float = 30.0,
    trainer: Any | None = None,
    throughput: dict[str, float] | None = None,
    version_key: int = 0,
    model_id: str | None = None,
    trainer_output_dir: str | None = None,
    require_confidential: bool = False,
    verifier: Any = None,
) -> WindowResult:
    """One round on the current policy.

    The validator picks ``tasks_per_window`` tasks with ``group_size`` rollouts
    each, deals the rollouts to miners by measured throughput, verifies every
    trace (training mode), re-deals rollouts of failed miners, then pays one
    credit per accepted slot and takes one GRPO step on groups with reward spread.
    ``families`` defaults to every family of ``gym``.

    ``throughput`` (tokens/s per miner) is updated in place. ``trainer`` is
    reused only while its in-memory LoRA equals the published policy.
    """
    from gft_subnet.training import informative_episodes, pack_episodes, run_one_grpo_step

    if group_size < 2:
        raise ValueError(f"group_size {group_size} must be at least 2")
    throughput = {} if throughput is None else throughput
    _push_policy(miners, policy.manifest(), require_confidential=require_confidential)
    manifest = ack_miners(policy, miners)
    window_id = ledger.open_window(manifest.version, manifest.checksum)
    for miner in miners:
        ledger.ack(window_id, miner.hotkey, manifest.checksum)
    window_randomness = secrets.token_hex(16)
    tasks = choose_window_tasks(
        families=list(families or gym.families), count=tasks_per_window, window_id=window_id,
        used_tasks=ledger.used_tasks())
    specs = [
        RolloutSpec(family, seed, index,
                    sampling_key(window_randomness, family, seed, manifest.checksum, index))
        for family, seed in tasks for index in range(group_size)
    ]
    temperature = float(manifest.sampling["temperature"])
    active_trainer = trainer if _trainer_matches(trainer, manifest) else None
    trainer = None
    training_skipped = None
    if train:
        try:
            if active_trainer is None:
                # Holder with the same shape as a trainer, so the model survives idle windows.
                active_trainer = SimpleNamespace(
                    model=_load_policy_model(model_id, manifest.adapter_dir),
                    optimizer=None, policy_checksum=manifest.checksum)
        except PolicyError as exc:
            print(f"policy model unavailable: {exc}", flush=True)
            training_skipped = "policy_error"
    rng = random.Random(window_randomness)
    quarantined = quarantined_miners(ledger.credit_history(), window_id=window_id)
    max_tokens = int(manifest.sampling["max_tokens"])
    eos_token_id = getattr(getattr(verifier, "tokenizer", None), "eos_token_id", None)
    accepted: dict[RolloutSpec, Episode] = {}
    attempts: list[Episode] = []
    failed: set[str] = set()
    dealt: dict[str, int] = defaultdict(int)
    trace: dict[str, float] = {}
    pending = specs
    for _ in range(1 + REASSIGN_ROUNDS):
        healthy = [miner for miner in miners
                   if miner.hotkey not in failed and miner.hotkey not in quarantined
                   and not getattr(miner, "_timed_out", False)]
        if not pending or not healthy:
            break
        assignment = assign_rollouts(pending, healthy, throughput, rng)
        for hotkey, assigned in assignment.items():
            dealt[hotkey] += len(assigned)
        pairs, rates = collect_rollouts(
            gym=gym, miners=healthy, policy=policy, assignment=assignment,
            turn_timeout_seconds=turn_timeout_seconds, verifier=verifier)
        episodes = [episode for _, episode in pairs]
        if train and training_skipped is None:
            try:
                stats = verify_episode_traces(
                    _unwrap_trainer_model(active_trainer), episodes, temperature=temperature,
                    eos_token_id=eos_token_id, max_tokens=max_tokens)
                trace = {key: max(trace.get(key, 0.0), value) for key, value in stats.items()}
            except Exception as exc:
                if not _is_cuda_oom(exc):
                    raise
                print(f"trace verification skipped: {exc}", flush=True)
                training_skipped = "oom"
                active_trainer = None
        attempts.extend(episodes)
        for spec, episode in pairs:
            if episode.infra_error is None:
                accepted[spec] = episode
            else:
                failed.add(episode.hotkey)
        _update_throughput(throughput, rates, failed)
        pending = [spec for spec in pending if spec not in accepted]
    for episode in attempts:
        ledger.record_episode(window_id, episode)
    final = [accepted[spec] for spec in specs if spec in accepted]
    # Unverified traces must not earn: without a verification pass in training mode,
    # record nothing and submit no weights this window.
    if training_skipped:
        credits: dict[str, float] = {}
        slashed: set[str] = set()
        warned: set[str] = set()
        failed = set()
    else:
        credits = miner_credits(final)
        slashed, warned = classify_trace_failures(attempts)
    ledger.record_credits(window_id, credits, slashed, warned)
    scores = credit_scores(ledger.credit_history())
    for hotkey in failed:
        scores.setdefault(hotkey, 0.0)
    snapshot = chain.metagraph()
    for hotkey, score in scores.items():
        uid = snapshot.uid_for(hotkey)
        if uid is not None:
            ledger.record_score(window_id, hotkey, uid, score)
    weights_skipped = None
    try:
        if dummy_weights:
            miners_on_chain = [neuron for neuron in snapshot.neurons if neuron.axon is not None]
            uids = [neuron.uid for neuron in miners_on_chain]
            weights = [1.0 / len(uids) for _ in uids] if uids else []
            block = chain.set_weights(validator_hotkey, uids, weights) if uids else snapshot.block
        else:
            uids, weights, block = set_miner_weights(
                chain=chain,
                validator_hotkey=validator_hotkey,
                scores=scores,
                version_key=version_key,
            )
    except Exception as exc:
        if not _is_weights_rate_limit(exc):
            raise
        print(f"set_weights skipped: {exc}", flush=True)
        uids, weights, block = [], [], snapshot.block
        weights_skipped = "rate_limit"
    packed = None
    next_policy = None
    push_error = None
    if train and training_skipped is None:
        train_episodes = informative_episodes(final, group_size=group_size)
        if not pack_episodes(final, allow_scripted=False, group_size=group_size)["n"]:
            training_skipped = "no_groups"
        elif not train_episodes:
            training_skipped = "zero_advantages"
        else:
            try:
                # Unset until the new trainer exists, so a failed build keeps nothing.
                previous, active_trainer = active_trainer, None
                active_trainer = _make_trainer(
                    model_id=model_id,
                    output_dir=trainer_output_dir or (manifest.adapter_dir + "-trainer"),
                    episodes=train_episodes,
                    group_size=group_size,
                    adapter_dir=manifest.adapter_dir,
                    temperature=temperature,
                    model=_unwrap_trainer_model(previous),
                    optimizer=_unwrap_trainer_optimizer(previous),
                )
                active_trainer.policy_checksum = manifest.checksum
                trained = run_one_grpo_step(
                    train_episodes,
                    adapter_dir=manifest.adapter_dir,
                    allow_scripted=False,
                    group_size=group_size,
                    trainer=active_trainer,
                )
                next_policy = policy.publish_from_dir(
                    version=manifest.version + 1, adapter_dir=trained["adapter_dir"],
                    gym_id=manifest.gym_id,
                )
                active_trainer.policy_checksum = next_policy.checksum
                packed = trained["packed"]
            except PolicyError as exc:
                print(f"training skipped: {exc}", flush=True)
                training_skipped = "policy_error"
            except Exception as exc:
                if not _is_cuda_oom(exc):
                    raise
                print(f"training skipped: {exc}", flush=True)
                training_skipped = "oom"
        if next_policy is not None:
            # v+1 is already published; the next window re-pushes it if this fails.
            try:
                _push_policy(miners, next_policy, require_confidential=require_confidential)
                ack_miners(policy, miners)
            except PolicyError as exc:
                print(f"policy push failed: {exc}", flush=True)
                push_error = str(exc)
    if not _trainer_matches(active_trainer, policy.manifest()):
        # Weights may be mid-update or unpublished; never train the next window from them.
        # run_windows drops its reference and frees GPU memory before the next window.
        active_trainer = None
    return WindowResult(
        window_id=window_id,
        policy_version=manifest.version,
        episodes=final,
        scores=scores,
        uids=uids,
        weights=weights,
        block=block,
        next_policy=next_policy,
        packed=packed,
        dummy_weights=dummy_weights,
        extras={"explorer_url": getattr(chain, "explorer_url", lambda: None)(),
                "weights_skipped": weights_skipped or (not bool(uids)),
                "training_skipped": training_skipped,
                "push_error": push_error,
                "tasks": [list(task) for task in tasks],
                "dealt": dict(dealt),
                "failed_miners": sorted(failed),
                "slashed_miners": sorted(slashed),
                "warned_miners": sorted(warned),
                "quarantined_miners": sorted(quarantined),
                "missing_rollouts": len(specs) - len(final),
                "credits": credits,
                "throughput": dict(throughput),
                "trace": trace or None},
        trainer=active_trainer,
    )


def _update_throughput(throughput: dict[str, float], rates: dict[str, float], failed: set[str]) -> None:
    """Clean miners update their measured tokens/s; a failure counts as a zero-rate sample."""
    for hotkey, rate in rates.items():
        if hotkey in failed:
            continue
        old = throughput.get(hotkey)
        throughput[hotkey] = rate if old is None else (
            (1 - THROUGHPUT_EMA_ALPHA) * old + THROUGHPUT_EMA_ALPHA * rate)
    for hotkey in failed:
        if hotkey in throughput:
            throughput[hotkey] *= 1 - THROUGHPUT_EMA_ALPHA


def run_windows(
    *,
    steps: int = 1,
    log_path: str | Path | None = None,
    **kwargs: Any,
) -> list[WindowResult]:
    """On-policy loop: one collect+optional GRPO step per window. Reuses GPU weights."""
    if steps < 1:
        raise ValueError("steps must be positive")
    trainer = kwargs.pop("trainer", None)
    throughput = kwargs.pop("throughput", None)
    throughput = {} if throughput is None else throughput
    results: list[WindowResult] = []
    log_file = Path(log_path) if log_path else None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    for index in range(steps):
        result = run_window(trainer=trainer, throughput=throughput, **kwargs)
        trainer = result.trainer
        result.trainer = None
        payload = {
            "step": index + 1,
            "steps": steps,
            "window_id": result.window_id,
            "policy_version": result.policy_version,
            "rewards": _task_rewards(result.episodes),
            "scores": result.scores,
            "uids": result.uids,
            "weights": result.weights,
            "block": result.block,
            "next_version": None if result.next_policy is None else result.next_policy.version,
            **{key: result.extras.get(key) for key in (
                "tasks", "dealt", "failed_miners", "missing_rollouts", "credits",
                "throughput", "training_skipped", "trace")},
        }
        print(
            f"window {index + 1}/{steps} tasks={result.extras.get('tasks')} "
            f"dealt={result.extras.get('dealt')} failed={result.extras.get('failed_miners')} "
            f"trained={result.next_policy is not None} "
            f"skipped={result.extras.get('training_skipped')}",
            flush=True,
        )
        if log_file is not None:
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        results.append(result)
        _empty_cuda_cache()
    if results:
        results[-1].trainer = trainer
    return results


def _task_rewards(episodes: list[Episode]) -> dict[str, list[list[Any]]]:
    rewards: dict[str, list[list[Any]]] = {}
    for episode in episodes:
        rewards.setdefault(f"{episode.family}/{episode.task_seed}", []).append(
            [episode.hotkey, round(float(episode.reward), 4)])
    return rewards


def _load_policy_model(model_id: str | None, adapter_dir: str) -> Any:
    if not model_id:
        raise PolicyError("training requires --model")
    from gft_subnet.training import load_policy_model
    return load_policy_model(model_id=model_id, adapter_dir=adapter_dir)


def verify_episode_traces(model: Any, episodes: list[Episode], *, temperature: float,
                          **limits: Any) -> dict:
    from gft_subnet.verify import verify_episode_traces as verify
    return verify(model, episodes, temperature=temperature, **limits)


def _is_weights_rate_limit(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "rate limit" in text and "weight" in text


def _is_cuda_oom(exc: BaseException) -> bool:
    if type(exc).__name__ in {"OutOfMemoryError", "CudaOutOfMemoryError"}:
        return True
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "cublas" in text)


def _empty_cuda_cache() -> None:
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_miner_weights(
    *,
    chain: ChainClient,
    validator_hotkey: str,
    scores: dict[str, float],
    version_key: int = 0,
) -> tuple[list[int], list[float], int]:
    snapshot = chain.metagraph()
    uids, weights = scores_to_weights(scores, snapshot)
    if not uids:
        return [], [], snapshot.block
    try:
        block = chain.set_weights(validator_hotkey, uids, weights, version_key=version_key)
    except TypeError:
        block = chain.set_weights(validator_hotkey, uids, weights)
    return uids, weights, block


def _push_policy(
    miners: list[MinerClient], manifest: PolicyManifest, *, require_confidential: bool = False,
) -> None:
    from gft_subnet.attestation import MinerAttestation, assert_may_push_lora
    from gft_subnet.policy import read_adapter_files

    files = read_adapter_files(manifest.adapter_dir) if manifest.adapter_dir else None
    for miner in miners:
        reporter = getattr(miner, "attestation", None)
        report = reporter() if callable(reporter) else MinerAttestation()
        assert_may_push_lora(report, require_confidential=require_confidential)
        loader = getattr(miner, "load_policy", None)
        if callable(loader):
            loader(manifest, files)


def _trainer_matches(trainer: Any, manifest: PolicyManifest) -> bool:
    return trainer is not None and getattr(trainer, "policy_checksum", None) == manifest.checksum


def _unwrap_trainer_optimizer(trainer: Any) -> Any:
    """Raw torch optimizer, so Adam moments survive across one-step trainers."""
    optimizer = getattr(trainer, "optimizer", None)
    return getattr(optimizer, "optimizer", optimizer)


def _unwrap_trainer_model(trainer: Any) -> Any:
    if trainer is None:
        return None
    model = trainer.model
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None:
        return accelerator.unwrap_model(model)
    return getattr(model, "module", model)


def _make_trainer(
    *,
    model_id: str | None,
    output_dir: str,
    episodes: list[Episode],
    group_size: int,
    adapter_dir: str,
    temperature: float,
    model: Any = None,
    optimizer: Any = None,
) -> Any:
    if not model_id:
        raise PolicyError("training requires --model")
    from gft_subnet.training import make_one_step_trainer
    return make_one_step_trainer(
        model_id=model_id, output_dir=output_dir, episodes=episodes, group_size=group_size,
        adapter_dir=adapter_dir, temperature=temperature, model=model, optimizer=optimizer,
    )
