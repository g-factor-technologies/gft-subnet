"""Validator-owned gym loop. One HTTP turn per gym step. Miners do not import gyms."""
from __future__ import annotations

import json
import re
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Thread
from typing import Any, Protocol

from gft_subnet.gym import Gym
from gft_subnet.policy import PolicyError, PolicyRegistry
from gft_subnet.protocol import (
    MAX_STEPS,
    Episode,
    EpisodeTurn,
    TurnRequest,
    TurnResponse,
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
class MinerClient(Protocol):
    hotkey: str
    scripted: bool

    def generate_batch(self, requests: list[TurnRequest]) -> list[TurnResponse]: ...

    def ack_policy(self, version: int, checksum: str, gym_id: str = "") -> None: ...


def parse_json_action(text: str) -> dict[str, Any]:
    """Validator-side parse. Miners return text; ``env.step`` rejects unknown actions."""
    text = _THINK_BLOCK.sub("", text or "")
    match = _JSON_OBJECT.search(text)
    if not match:
        return {"type": "invalid"}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"type": "invalid"}
    if payload.get("name") == "act":
        payload = payload.get("arguments")
    if isinstance(payload, dict) and isinstance(payload.get("action"), dict):
        payload = payload["action"]
    if not isinstance(payload, dict) or not payload:
        return {"type": "invalid"}
    return payload


def _observation_text(observation: dict[str, Any]) -> str:
    return json.dumps(observation, sort_keys=True, separators=(",", ":"))


def _first_messages(gym: Gym, observation: dict[str, Any], max_steps: int) -> list[dict[str, Any]]:
    action_schema = json.dumps(gym.action_schema, sort_keys=True, separators=(",", ":"))
    instructions = gym.instructions(observation)
    task_instructions = f"\nTask instructions: {instructions}" if instructions else ""
    return [
        {
            "role": "system",
            "content": (
                "You are an agent controlling one stateful OpenEnv episode. "
                "Call the tool as act(action=<ACTION_OBJECT>), where only the "
                "nested ACTION_OBJECT matches the supplied JSON Schema. Never "
                "place action fields beside action. Continue until the environment says "
                "done or the step budget is exhausted. Maximize the environment "
                "reward; do not invent observations or tool results."),
        },
        {
            "role": "user",
            "content": (
                f"Environment: {gym.gym_id}\n"
                "Action type: typed\n"
                f"Maximum steps: {max_steps}\n"
                f"Action JSON Schema: {action_schema}"
                f"{task_instructions}"),
        },
    ]


def _messages(
    gym: Gym, observation: dict[str, Any], previous: dict[str, Any] | None = None,
    *, max_steps: int = MAX_STEPS,
) -> list[dict[str, Any]]:
    """First turn: instructions + full observation. Later turns: only changed fields.

    Earlier turns stay in the miner's token prefix, so resending the full
    snapshot every step only inflates completions (and trainer memory).
    """
    if previous is not None:
        changed = {key: value for key, value in observation.items() if previous.get(key) != value}
        return [{
            "role": "user",
            "content": (
                "Observation update (changed fields only):\n"
                + _observation_text(changed) + "\n" + gym.reply_hint),
        }]
    messages = _first_messages(gym, observation, max_steps)
    messages[-1]["content"] += "\n" + _observation_text(observation) + "\n" + gym.reply_hint
    return messages


def run_episode(
    *,
    gym: Gym,
    miner: MinerClient,
    family: str,
    task_seed: int,
    generation_seed: int,
    policy: PolicyRegistry,
    max_steps: int = MAX_STEPS,
    split: str = "train",
    turn_timeout_seconds: float = 30.0,
    verifier: Any = None,
    sampling_key: str = "",
) -> Episode:
    """Play one rollout (see ``run_rollouts``)."""
    return run_rollouts(
        gym=gym, miner=miner,
        rollouts=[RolloutSpec(family, task_seed, generation_seed, sampling_key)],
        policy=policy, max_steps=max_steps, split=split,
        turn_timeout_seconds=turn_timeout_seconds, verifier=verifier,
    )[0]


@dataclass(frozen=True)
class RolloutSpec:
    """One rollout the validator assigns: task, index inside the task's group, forced-sampling key."""

    family: str
    task_seed: int
    index: int
    sampling_key: str


@dataclass
class _Rollout:
    spec: RolloutSpec
    episode_id: str
    env: Any
    observation: dict[str, Any]
    prompt: str
    turns: list[EpisodeTurn] = field(default_factory=list)
    prefix: list[int] = field(default_factory=list)
    previous: dict[str, Any] | None = None
    reward: float = 0.0
    scripted: bool = False
    finished: bool = False
    error: str | None = None


def run_rollouts(
    *,
    gym: Gym,
    miner: MinerClient,
    rollouts: list[RolloutSpec],
    policy: PolicyRegistry,
    max_steps: int = MAX_STEPS,
    split: str = "train",
    turn_timeout_seconds: float = 30.0,
    verifier: Any = None,
) -> list[Episode]:
    """Play the rollouts assigned to one miner in lockstep: one batched miner call per turn.

    Rollouts may belong to different tasks; each ``sampling_key`` fixes every
    sampled token (see ``sampling.py``). The validator steps each gym and drops
    episodes from the batch as they finish. A failed batch call (timeout,
    transport) is an infrastructure error for every rollout still running.

    With a ``TokenTraceVerifier`` the validator rebuilds each turn context and
    decodes the reward-bearing text from the miner's token IDs itself.
    """
    gym.prepare()
    manifest = policy.manifest()
    if miner.hotkey not in policy.acked(manifest.version, gym_id=manifest.gym_id):
        raise PolicyError(f"{miner.hotkey} has not acked {manifest.slot}")
    scripted = bool(getattr(miner, "scripted", False))
    states = []
    for spec in rollouts:
        env = gym.make(spec.family, spec.task_seed, split)
        observation = env.reset_observation()
        states.append(_Rollout(
            spec=spec,
            episode_id=(f"ep-{manifest.version}-{spec.family}-{spec.task_seed}-{spec.index}-"
                        f"{miner.hotkey}-{uuid.uuid4().hex[:8]}"),
            env=env, observation=observation,
            prompt=_messages(gym, observation, max_steps=max_steps)[0]["content"],
            scripted=scripted,
        ))
    for step in range(max_steps):
        active = [state for state in states if not state.finished and not state.observation.get("done")]
        if not active:
            break
        requests = [
            TurnRequest(
                policy_version=manifest.version, episode_id=state.episode_id, turn=step,
                family=state.spec.family, task_seed=state.spec.task_seed,
                generation_seed=state.spec.index, observation=state.observation,
                messages=_messages(gym, state.observation, state.previous, max_steps=max_steps),
                action_schema=gym.action_schema, prefix_token_ids=state.prefix,
                gym_id=manifest.gym_id, sampling_key=state.spec.sampling_key,
            )
            for state in active
        ]
        try:
            responses = _generate_batch(miner, requests, timeout=turn_timeout_seconds)
            if len(responses) != len(requests):
                raise ValueError("miner returned a different number of turns than requested")
        except PolicyError:
            raise
        except _INFRA_ERRORS as exc:
            for state in active:
                state.error, state.finished = f"{type(exc).__name__}: {exc}", True
            break
        for state, request, response in zip(active, requests, responses, strict=True):
            if response.policy_version != manifest.version:
                raise PolicyError("miner replied with a mixed policy version")
            if response.episode_id != state.episode_id:
                raise PolicyError("miner replied for a different episode")
            if response.hotkey != miner.hotkey:
                raise PolicyError("miner replied with a different hotkey")
            _advance(gym, state, request, response, verifier)
    return [
        Episode(
            episode_id=state.episode_id, hotkey=miner.hotkey, family=state.spec.family,
            task_seed=state.spec.task_seed, generation_seed=state.spec.index,
            policy_version=manifest.version, turns=state.turns,
            reward=0.0 if state.error else state.reward,
            complete=state.error is None and bool(state.env.done),
            prompt=state.prompt, scripted=state.scripted, infra_error=state.error,
            sampling_key=state.spec.sampling_key,
        )
        for state in states
    ]


_INFRA_ERRORS = (OSError, TimeoutError, TypeError, ValueError, KeyError, RuntimeError)


def _advance(gym: Gym, state: _Rollout, request: TurnRequest, response: TurnResponse,
             verifier: Any) -> None:
    """Apply one miner turn to one rollout and step its gym."""
    state.scripted = state.scripted or response.scripted
    try:
        text = response.text if verifier is None else verifier.verify(request, response)
    except ValueError as exc:
        state.error, state.finished = f"TraceError: {exc}", True
        return
    action = gym.normalize_action(parse_json_action(text) or dict(response.action or {}))
    state.turns.append(EpisodeTurn(
        turn=request.turn, action=action, observation={},
        token_ids=list(response.token_ids), logprobs=list(response.logprobs),
        env_token_ids=[], prompt_ids=list(response.prompt_ids),
    ))
    state.prefix = list(response.prompt_ids) + list(response.token_ids)
    state.previous = state.observation
    try:
        result = state.env.step(action)
    except gym.illegal_action_errors:
        # An illegal action is a model failure, not an infrastructure failure.
        state.reward, state.finished = 0.0, True
        return
    except _INFRA_ERRORS as exc:
        state.error, state.finished = f"{type(exc).__name__}: {exc}", True
        return
    state.observation = dict(result.get("observation") or {})
    state.turns[-1].observation = state.observation
    state.reward = float(result.get("reward") or 0.0)
    state.finished = bool(result.get("done"))


def _generate_batch(
    miner: MinerClient, requests: list[TurnRequest], *, timeout: float,
) -> list[TurnResponse]:
    if getattr(miner, "_timed_out", False):
        raise TimeoutError("miner client is quarantined after a timed-out turn")
    if timeout <= 0:
        return miner.generate_batch(requests)
    # A daemon cannot hold up shutdown, unlike ThreadPoolExecutor.__exit__.
    # On timeout the miner is quarantined; do not start more work on a busy miner.
    future: Future = Future()
    def invoke() -> None:
        try:
            future.set_result(miner.generate_batch(requests))
        except BaseException as exc:  # noqa: BLE001 - forward worker failures to the caller's Future
            future.set_exception(exc)
    Thread(target=invoke, daemon=True).start()
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        miner._timed_out = True
        raise
