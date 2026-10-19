"""DEX arbitrage routing: the TEST gym for the subnet protocol.

This gym exists to exercise the protocol end to end (validator-chosen tasks,
multi-turn rollouts, trace replay, slot credit, GRPO) on a local chain and on
testnet. It is not a product and not a trading strategy: tasks are synthetic,
offline pool snapshots, graded by exact integer AMM math in ``reference``
mode, with no live market data or execution. Production gyms are private
plug-ins (see ``gft_subnet.gym.Gym``).

The environment code is copied from GFT Studio's DEX vertical; keep it in sync
when the gym changes there.
"""
from __future__ import annotations

import os
from typing import Any

from gft_subnet.gym import Gym
from gft_subnet.gyms.dex_arbitrage._support import ContractError
from gft_subnet.gyms.dex_arbitrage.environment import DexArbitrageGym, solve_from_observation
from gft_subnet.gyms.dex_arbitrage.safety import UnsafeDexSubmission
from gft_subnet.gyms.dex_arbitrage.tasks import ACTION_SCHEMA, FAMILIES, build_family_task

JSON_ACTION_REPLY = (
    "Reply with a single JSON object on one line and nothing else. "
    "Do not pretty-print. Do not add extra keys. "
    "Size amount_in_wei from the pool reserves and the AMM formula; "
    "do not copy constraints.min_amount_in_wei. "
    "After submit_route, inspect_evaluation and try another size if net "
    "profit is missing or not positive, then finalize."
)

#: The subnet always grades in reference mode: exact integer AMM math, no Foundry replay.
REFERENCE_MODE = "reference"
ENV_MODE_KEY = "SKILLHUB_DEX_ARBITRAGE_MODE"


class DexArbitrageTestGym(Gym):
    """Test gym: synthetic DEX routing tasks for protocol runs, not a production gym."""

    gym_id = "dex_arbitrage"
    families = tuple(FAMILIES)
    action_schema = ACTION_SCHEMA
    illegal_action_errors = (ContractError, UnsafeDexSubmission)
    reply_hint = JSON_ACTION_REPLY

    def make(self, family: str, task_seed: int, split: str) -> DexArbitrageGym:
        return DexArbitrageGym(build_family_task(family, task_seed, split))

    def prepare(self) -> None:
        current = os.environ.get(ENV_MODE_KEY)
        if current and current != REFERENCE_MODE:
            raise RuntimeError(f"{ENV_MODE_KEY}={current}; the subnet test gym needs {REFERENCE_MODE}")
        os.environ[ENV_MODE_KEY] = REFERENCE_MODE

    def normalize_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if action.get("amount_in_wei") is not None:
            action["amount_in_wei"] = str(action["amount_in_wei"])
        return action

    def scripted_actions(self, observation: dict[str, Any]) -> list[dict[str, Any]]:
        return solve_from_observation(observation)
