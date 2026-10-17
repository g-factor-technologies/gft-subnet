"""Wire types for validator ↔ miner turns. JSON only; no Axon/Synapse."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


PROTOCOL_VERSION = 4
MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_STEPS = 6
DEFAULT_GYM_ID = "dex_arbitrage"


def policy_slot(gym_id: str, version: int) -> str:
    return f"{gym_id}-v{int(version)}"


def dumps(payload: dict[str, Any]) -> bytes:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    if len(raw) > MAX_BODY_BYTES:
        raise ValueError(f"payload exceeds {MAX_BODY_BYTES} bytes")
    return raw


def loads(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_BODY_BYTES:
        raise ValueError(f"payload exceeds {MAX_BODY_BYTES} bytes")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    return payload


@dataclass(frozen=True)
class PolicyManifest:
    version: int
    base_model: str
    revision: str | None
    checksum: str
    sampling: dict[str, Any] = field(default_factory=dict)
    adapter_dir: str = ""
    gym_id: str = DEFAULT_GYM_ID

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def slot(self) -> str:
        return policy_slot(self.gym_id, self.version)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PolicyManifest":
        return cls(
            version=int(payload["version"]),
            base_model=str(payload.get("base_model") or ""),
            revision=payload.get("revision"),
            checksum=str(payload["checksum"]),
            sampling=dict(payload.get("sampling") or {}),
            adapter_dir=str(payload.get("adapter_dir") or ""),
            gym_id=str(payload.get("gym_id") or DEFAULT_GYM_ID),
        )


@dataclass
class TurnRequest:
    policy_version: int
    episode_id: str
    turn: int
    family: str
    task_seed: int
    generation_seed: int
    observation: dict[str, Any]
    messages: list[dict[str, Any]]
    action_schema: dict[str, Any] | None = None
    prefix_token_ids: list[int] = field(default_factory=list)
    gym_id: str = DEFAULT_GYM_ID
    sampling_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TurnRequest":
        return cls(
            policy_version=int(payload["policy_version"]),
            episode_id=str(payload["episode_id"]),
            turn=int(payload["turn"]),
            family=str(payload["family"]),
            task_seed=int(payload["task_seed"]),
            generation_seed=int(payload["generation_seed"]),
            observation=dict(payload.get("observation") or {}),
            messages=list(payload.get("messages") or []),
            action_schema=payload.get("action_schema"),
            prefix_token_ids=[int(v) for v in payload.get("prefix_token_ids", [])],
            gym_id=str(payload.get("gym_id") or DEFAULT_GYM_ID),
            sampling_key=str(payload.get("sampling_key") or ""),
        )


@dataclass
class TurnResponse:
    policy_version: int
    episode_id: str
    hotkey: str
    action: dict[str, Any]
    text: str
    token_ids: list[int]
    logprobs: list[float]
    prompt_ids: list[int] = field(default_factory=list)
    scripted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TurnResponse":
        token_ids = [int(value) for value in payload.get("token_ids") or []]
        logprobs = [float(value) for value in payload.get("logprobs") or []]
        if len(logprobs) != len(token_ids) or any(not math.isfinite(v) for v in logprobs):
            raise ValueError("logprobs must match token_ids")
        action = payload.get("action")
        if not isinstance(action, dict):
            action = {}
        return cls(
            policy_version=int(payload["policy_version"]),
            episode_id=str(payload["episode_id"]),
            hotkey=str(payload.get("hotkey") or ""),
            action=action,
            text=str(payload.get("text") or ""),
            token_ids=token_ids,
            logprobs=logprobs,
            prompt_ids=[int(v) for v in payload.get("prompt_ids", [])],
            scripted=bool(payload.get("scripted", True)),
        )


@dataclass
class EpisodeTurn:
    turn: int
    action: dict[str, Any]
    observation: dict[str, Any]
    token_ids: list[int]
    logprobs: list[float]
    env_token_ids: list[int]
    prompt_ids: list[int] = field(default_factory=list)


@dataclass
class Episode:
    episode_id: str
    hotkey: str
    family: str
    task_seed: int
    generation_seed: int
    policy_version: int
    turns: list[EpisodeTurn]
    reward: float
    complete: bool
    prompt: str = ""
    scripted: bool = False
    infra_error: str | None = None
    sampling_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "hotkey": self.hotkey,
            "family": self.family,
            "task_seed": self.task_seed,
            "generation_seed": self.generation_seed,
            "policy_version": self.policy_version,
            "reward": self.reward,
            "complete": self.complete,
            "prompt": self.prompt,
            "scripted": self.scripted,
            "infra_error": self.infra_error,
            "sampling_key": self.sampling_key,
            "turns": [asdict(turn) for turn in self.turns],
        }
