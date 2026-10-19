"""Evidence-bound reward for typed DEX arbitrage routes."""
from __future__ import annotations

from typing import Any

from gft_subnet.gyms.dex_arbitrage._support import (
    efficiency_score,
    finish_score,
    make_reward_profile,
)

GATES = [
    "source_safe", "route_valid", "amount_bounded", "execution_success",
    "accounting_conserved", "positive_net_profit", "stress_positive",
    "native_replay", "verified_protocol", "evidence_bound", "finalized",
    "no_unsafe_submission",
]

#: Zero set = source/unsafe + protocol/evidence.  ``positive_net_profit``
#: and ``stress_positive`` are *the task*: failing them still blocks
#: ``hard_gated_success`` and caps the reward at ``0.25 * progress`` (a losing
#: route can never out-score a profitable one), but a valid, executable route
#: keeps a bounded shaping signal instead of collapsing the group to zero.
#: ``finalized`` is a soft gate like in the sql vertical: an episode that
#: submits a route but never calls finalize earns at most ``0.25 * progress``
#: (the trainer scores an unfinished episode by its best step), the full
#: reward still needs finalize. With it in the zero set Qwen3-8B scored 0.0 on
#: 128 rollouts (coverage attempt4-1788392772, mean_turns = max_steps, never
#: finalized), i.e. no GRPO signal at all.
ZERO_REWARD_GATES = {
    "source_safe", "verified_protocol", "evidence_bound",
    "no_unsafe_submission",
}

# The exact integer AMM math *is* the ground truth for the declared pool
# semantics; the Foundry/revm replay only cross-checks the implementation.
# A machine without the pinned Foundry image therefore trains against the
# full reward, but its result cannot be promoted.
PROMOTION_ONLY_GATES = {"native_replay"}


def profile_with_digest() -> dict[str, Any]:
    return make_reward_profile(
        "dex-arbitrage-routing-reward-v1",
        {
            "route_validity": 0.15,
            "execution_success": 0.15,
            "profit_quality": 0.55,
            "stress_robustness": 0.10,
            "iteration_efficiency": 0.05,
        },
        GATES,
        "dex-arbitrage-exact-integer-amm-foundry-replay-grader-v2",
    )


def score_episode(
    state: dict[str, Any], task: dict[str, Any], receipts: list[dict[str, Any]],
    *, finalized: bool, calls: int, latency_ms: float, cost_micros: int,
    evidence: dict[str, Any] | None, profile: dict[str, Any],
) -> dict[str, Any]:
    result = state.get("best_evaluation") or {}
    gate_results = result.get("gates") or {}
    quality = result.get("quality") or {}
    correctness_unlocked = bool(result.get("correctness_unlocked"))

    components = {
        "route_validity": float(quality.get("route_validity") or 0.0),
        "execution_success": float(quality.get("execution_success") or 0.0),
        # Profit cannot compensate for an invalid or economically losing route.
        "profit_quality": float(quality.get("profit_quality") or 0.0)
        if correctness_unlocked else 0.0,
        "stress_robustness": float(quality.get("stress_robustness") or 0.0),
        "iteration_efficiency": efficiency_score(
            calls, cost_micros, task["public"]["budgets"]),
    }
    gates = {
        "source_safe": result.get("source_safe") is True,
        **{
            name: gate_results.get(name) is True
            for name in (
                "route_valid", "amount_bounded", "execution_success",
                "accounting_conserved", "positive_net_profit",
                "stress_positive", "native_replay",
            )
        },
        "verified_protocol": result.get("verified_protocol") is True,
        "evidence_bound": result.get("evidence_bound") is True,
        "finalized": finalized,
        "no_unsafe_submission": not any(
            row.get("rejected_unsafe") for row in receipts),
    }
    progress = min(
        0.99,
        0.35 * components["route_validity"]
        + 0.35 * components["execution_success"]
        + 0.30 * components["stress_robustness"],
    )
    scored = finish_score(
        components, gates, profile,
        task_success=1.0 if correctness_unlocked else progress,
        evidence_id=task["hidden"]["evidence_id"], evidence=evidence,
        effective_work=any(
            row.get("effective") and row.get("operation") != "finalize"
            for row in receipts
        ),
        zero_reward_gates=ZERO_REWARD_GATES,
        promotion_only_gates=PROMOTION_ONLY_GATES,
    )
    native = result.get("native_replay") or {}
    scored["verification_level"] = (
        "foundry_revm_arithmetic_replay"
        if native.get("matched") is True
        else "reference_exact_integer_amm"
    )
    scored["promotion_eligible"] = bool(
        scored["hard_gated_success"]
        and result.get("promotion_eligible") is True
    )
    scored["training_reward_only"] = not scored["promotion_eligible"]
    scored["quality_unlocked"] = correctness_unlocked
    return scored
