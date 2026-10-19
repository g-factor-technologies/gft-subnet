"""OpenEnv loop for typed, offline DEX arbitrage routing."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from gft_subnet.gyms.dex_arbitrage._support import ContractError
from gft_subnet.gyms.dex_arbitrage._support import TypedSimulator, build_terminal_evidence
from gft_subnet.gyms.dex_arbitrage.backend import (
    DexArbitrageBackend,
    public_evaluation,
)
from gft_subnet.gyms.dex_arbitrage.reward import (
    profile_with_digest,
    score_episode,
)
from gft_subnet.gyms.dex_arbitrage.safety import UnsafeDexSubmission
from gft_subnet.gyms.dex_arbitrage.tasks import gold_route_from_public

_FORBIDDEN = {
    "execute_shell", "read_file", "install_package", "submit_code",
    "submit_solidity", "rpc_call", "sign_transaction", "submit_metrics",
}


class DexArbitrageGym(TypedSimulator):
    def __init__(self, task: dict[str, Any],
                 reward_profile: dict[str, Any] | None = None):
        super().__init__(task, reward_profile or profile_with_digest())
        self.backend = DexArbitrageBackend(task)
        self.state["runtime"] = {
            "mode": self.backend.runtime["mode"],
            "native": self.backend.runtime["native"],
            "reference_verifier": self.backend.runtime["reference_verifier"],
            "native_verifier": self.backend.runtime["native_verifier"],
        }
        self.initial = deepcopy(self.state)

    def public_summary(self) -> dict[str, Any]:
        public = self.task["public"]
        return {
            "family": self.task["family"],
            "base_asset": public["base_asset"],
            "snapshot": deepcopy(public["snapshot"]),
            "constraints": deepcopy(public["constraints"]),
            "gas_model": deepcopy(public["gas_model"]),
            "fee_model": deepcopy(public["fee_model"]),
            "semantics": public["semantics"],
            "runtime": deepcopy(self.state["runtime"]),
            "evaluations": len(self.state["evaluations"]),
            "best": public_evaluation(self.state["best_evaluation"])
            if self.state.get("best_evaluation") else None,
        }

    def public_state(self) -> dict[str, Any]:
        return self.public_summary()

    def step(self, action: dict[str, Any], *, cost_micros: int = 0) -> dict[str, Any]:
        # Score after every step like the sql vertical, so a trainer that
        # takes the best step of an unfinished episode sees the submit_route
        # progress (capped at 0.25 x progress by the soft `finalized` gate)
        # instead of a flat 0 until finalize.
        result = super().step(action, cost_micros=cost_micros)
        if result["done"]:
            return result
        scored = self.score()
        result["reward"] = float(scored.get("native") or 0.0)
        result["metadata"] = {
            **result["metadata"],
            "base_reward": scored.get("base", 0.0),
            "task_progress": scored.get("task_progress", 0.0),
            "reward_gate": scored.get("reward_gate", 0.0),
            "failed_gates": scored.get("failed_gates", []),
        }
        return result

    def handle(self, action: dict[str, Any]) -> tuple[Any, str | None]:
        kind = action["type"]
        if kind == "inspect_task":
            return self.public_summary(), None
        if kind == "submit_route":
            return self._submit({
                "pools": action.get("pools"),
                "amount_in_wei": action.get("amount_in_wei"),
            })
        if kind == "inspect_evaluation":
            best = self.state.get("best_evaluation")
            return {"best": public_evaluation(best) if best else None}, None
        if kind == "finalize":
            self.receipts.append({"operation": "finalize", "effective": True})
            self.done = True
            best = self.state.get("best_evaluation")
            return {
                "finalized": True,
                "best": public_evaluation(best) if best else None,
            }, None
        if kind in _FORBIDDEN:
            return self.mark_unsafe(
                kind,
                "only typed pool addresses and a decimal input amount are "
                "allowed; code, shell, RPC, and signing are forbidden",
            )
        raise ContractError(f"unknown action type {kind!r}")

    def _submit(self, candidate: dict[str, Any]) -> tuple[Any, str | None]:
        try:
            result = self.backend.evaluate(candidate)
        except UnsafeDexSubmission as exc:
            return self.mark_unsafe("submit_route", str(exc))
        except (ContractError, OSError, TimeoutError, ValueError) as exc:
            self.receipts.append({
                "operation": "submit_route", "effective": False,
                "verified": False,
            })
            return {}, str(exc)
        stored = deepcopy(result)
        self.state["evaluations"].append(stored)
        best = self.state.get("best_evaluation")
        if best is None or self._rank(stored) > self._rank(best):
            self.state["best_evaluation"] = stored
        self.receipts.append({
            "operation": "submit_route", "effective": True,
            "candidate_digest": stored["candidate_digest"],
            "verified": stored["verified_protocol"],
        })
        return {"evaluation": public_evaluation(stored), "can_refine": True}, None

    @staticmethod
    def _rank(result: dict[str, Any]) -> tuple[int, int, float]:
        measurements = result.get("measurements") or {}
        quality = result.get("quality") or {}
        return (
            int(result.get("correctness_unlocked") is True),
            int(measurements.get("net_profit_wei") or -(10**30)),
            float(quality.get("stress_robustness") or 0.0),
        )

    def score(self) -> dict[str, Any]:
        finalized = any(
            row.get("operation") == "finalize" for row in self.receipts)
        evidence = build_terminal_evidence(
            self.task, self.receipts, self.artifact_payloads())
        return score_episode(
            self.state, self.task, self.receipts, finalized=finalized,
            calls=self.calls, latency_ms=self.latency_ms,
            cost_micros=self.cost_micros, evidence=evidence,
            profile=self.reward_profile,
        )

    def artifact_payloads(self) -> dict[str, dict[str, Any]]:
        return self.backend.artifacts(self.state.get("best_evaluation"))

    def close(self) -> None:
        pass


def solve_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    return [{
        "type": "submit_route",
        "pools": deepcopy(task["hidden"]["oracle_route"]),
        "amount_in_wei": str(task["hidden"]["oracle_amount_in_wei"]),
        "idempotency_key": "dex-arbitrage-gold-v1",
    }, {"type": "finalize"}]


def solve_from_observation(observation: dict[str, Any]) -> list[dict[str, Any]]:
    solved = gold_route_from_public(observation)
    return [{
        "type": "submit_route",
        **solved,
        "idempotency_key": "dex-arbitrage-public-solver-v1",
    }, {"type": "finalize"}]


DexArbitrageSimulator = DexArbitrageGym
