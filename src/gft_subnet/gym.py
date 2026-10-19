"""Gym plug-in contract. The validator runs the gym; miners never see gym code.

A gym is any deterministic, verifiable environment: the validator builds a
task from ``(family, task_seed, split)``, sends observations to miners and
steps the environment with their actions. Scoring and trace verification do
not depend on which gym is plugged in.
"""
from __future__ import annotations

import importlib
from typing import Any, Protocol


class GymEnv(Protocol):
    """One episode of one task."""

    done: bool

    def reset_observation(self) -> dict[str, Any]: ...

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """Return ``{"observation": dict, "reward": float, "done": bool}``."""
        ...


class IllegalAction(ValueError):
    """An action the gym refuses: a model failure (reward 0), not an infrastructure error."""


class Gym:
    """Base class for validator gyms. Set the class attributes and implement ``make``."""

    gym_id: str = ""
    families: tuple[str, ...] = ()
    action_schema: dict[str, Any] = {}
    #: Exceptions from ``env.step`` that mean an illegal action, not an outage.
    illegal_action_errors: tuple[type[BaseException], ...] = (IllegalAction,)
    #: Appended to every user turn: how the miner's model must format its reply.
    reply_hint: str = "Reply with a single JSON object on one line and nothing else."

    def make(self, family: str, task_seed: int, split: str) -> GymEnv:
        raise NotImplementedError

    def prepare(self) -> None:
        """Called before each batch of rollouts (e.g. to pin a grading mode)."""

    def instructions(self, observation: dict[str, Any]) -> str:
        return str(observation.get("instruction", ""))

    def normalize_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Validator-side cleanup of a parsed action before ``env.step``."""
        return action

    def scripted_actions(self, observation: dict[str, Any]) -> list[dict[str, Any]]:
        """Reference solver for labeled scripted miners (local smoke only, never trained on)."""
        raise NotImplementedError(f"gym {self.gym_id!r} has no scripted solver")


def load_gym(spec: str) -> Gym:
    """``"dex_arbitrage"`` (the built-in test gym) or ``"package.module:ClassName"`` (plug-in)."""
    if spec in {"", "dex_arbitrage"}:
        from gft_subnet.gyms.dex_arbitrage import DexArbitrageTestGym

        return DexArbitrageTestGym()
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"gym spec {spec!r} must be 'dex_arbitrage' or 'module:ClassName'")
    gym = getattr(importlib.import_module(module), name)()
    if not isinstance(gym, Gym):
        raise TypeError(f"{spec} is not a gft_subnet.gym.Gym")
    return gym
