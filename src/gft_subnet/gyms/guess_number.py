"""Reference gym: find a hidden integer with higher/lower feedback.

Small, deterministic and verifiable, so the full window (tasks, rollouts,
trace replay, scoring, GRPO) runs without any private environment. Fewer
guesses earn more reward, so rollouts of one task differ and GRPO has a
learning signal.
"""
from __future__ import annotations

import hashlib
from typing import Any

from gft_subnet.gym import Gym, IllegalAction

RANGES = {"guess_16": 16, "guess_64": 64}


def hidden_target(family: str, task_seed: int, split: str) -> int:
    size = RANGES[family]
    digest = hashlib.sha256(f"guess-number|{family}|{int(task_seed)}|{split}".encode()).digest()
    return 1 + int.from_bytes(digest[:8], "big") % size


class GuessNumberEnv:
    def __init__(self, family: str, task_seed: int, split: str = "train") -> None:
        if family not in RANGES:
            raise ValueError(f"unknown guess_number family {family!r}")
        self.size = RANGES[family]
        self.target = hidden_target(family, task_seed, split)
        self.low, self.high = 1, self.size
        self.guesses = 0
        self.last_guess: int | None = None
        self.feedback: str | None = None
        self.done = False

    def _observation(self) -> dict[str, Any]:
        return {
            "instruction": (f"Find the hidden integer between 1 and {self.size}. "
                            "Each guess is answered with higher, lower or correct. "
                            "Fewer guesses earn more reward. Finalize when done."),
            "low": self.low, "high": self.high, "guesses": self.guesses,
            "last_guess": self.last_guess, "feedback": self.feedback, "done": self.done,
        }

    def reset_observation(self) -> dict[str, Any]:
        return self._observation()

    def _reward(self) -> float:
        if self.feedback != "correct":
            return 0.0
        return max(0.1, 1.0 - 0.15 * (self.guesses - 1))

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.done:
            raise IllegalAction("episode is already done")
        kind = action.get("type")
        if kind == "guess":
            value = action.get("value")
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= self.size:
                raise IllegalAction(f"guess must be an integer in [1, {self.size}]")
            self.guesses += 1
            self.last_guess = value
            if value == self.target:
                self.feedback, self.done = "correct", True
            elif value < self.target:
                self.feedback, self.low = "higher", max(self.low, value + 1)
            else:
                self.feedback, self.high = "lower", min(self.high, value - 1)
        elif kind == "finalize":
            self.done = True
        elif kind != "inspect_task":
            raise IllegalAction(f"unknown action type {kind!r}")
        return {"observation": self._observation(), "reward": self._reward(), "done": self.done}


class GuessNumberGym(Gym):
    gym_id = "guess_number"
    families = tuple(RANGES)
    action_schema = {
        "type": "object",
        "properties": {
            "type": {"enum": ["inspect_task", "guess", "finalize"]},
            "value": {"type": "integer", "minimum": 1},
        },
        "required": ["type"],
        "additionalProperties": False,
    }

    def make(self, family: str, task_seed: int, split: str) -> GuessNumberEnv:
        return GuessNumberEnv(family, task_seed, split)

    def scripted_actions(self, observation: dict[str, Any]) -> list[dict[str, Any]]:
        """Binary search, one guess per turn."""
        if observation.get("done") or observation.get("feedback") == "correct":
            return [{"type": "finalize"}]
        low, high = int(observation["low"]), int(observation["high"])
        return [{"type": "guess", "value": (low + high) // 2}]

