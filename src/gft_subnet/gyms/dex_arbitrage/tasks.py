"""Frozen synthetic Uniswap V2-style routing tasks with a hidden oracle."""
from __future__ import annotations

import hashlib
import random
from copy import deepcopy
from typing import Any

from gft_subnet.gyms.dex_arbitrage._support import sha256_digest
from gft_subnet.gyms.dex_arbitrage._support import default_budgets
from gft_subnet.gyms.dex_arbitrage.math import solve_best_route

VERTICAL = "dex_arbitrage"
WAD = 10**18
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
FAMILIES = (
    "two_pool_cycle",
    "triangular_cycle",
    "competing_cycles",
    "four_hop_cycle",
    "size_sensitive_cycle",
)

ACTION_SCHEMA = {
    "title": "DexArbitrageAction",
    "type": "object",
    "additionalProperties": False,
    "required": ["type"],
    "properties": {
        "type": {"type": "string", "enum": [
            "inspect_task", "submit_route", "inspect_evaluation", "finalize",
        ]},
        "pools": {
            "type": "array", "minItems": 2, "maxItems": 4,
            "items": {"type": "string", "pattern": "^0x[0-9a-fA-F]{40}$"},
        },
        "amount_in_wei": {
            "type": "string", "pattern": "^(0|[1-9][0-9]{0,77})$"},
        "idempotency_key": {"type": "string", "maxLength": 128},
        "summary": {"type": "string", "maxLength": 1000},
    },
}

OBSERVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "instruction": {"type": "string"},
        "family": {"type": "string"},
        "vertical": {"type": "string"},
        "base_asset": {"type": "string"},
        "snapshot": {"type": "object"},
        "constraints": {"type": "object"},
        "gas_model": {"type": "object"},
        "fee_model": {"type": "object"},
        "semantics": {"type": "string"},
        "result": {"type": "object"},
        "ok": {"type": "boolean"},
        "done": {"type": "boolean"},
    },
}

_INSTRUCTIONS = {
    "two_pool_cycle": (
        "Find a profitable WETH round trip across two constant-product pools "
        "that quote the same token at different prices."
    ),
    "triangular_cycle": (
        "Find and size a profitable triangular WETH route across three pools."
    ),
    "competing_cycles": (
        "Choose the better of multiple valid WETH cycles and size the trade."
    ),
    "four_hop_cycle": (
        "Find and size a profitable four-pool WETH cycle without repeating pools."
    ),
    "size_sensitive_cycle": (
        "Find a profitable WETH cycle in a shallow graph where price impact makes "
        "input sizing as important as path selection."
    ),
}

_SEMANTICS = (
    "Every pool is a frozen Uniswap V2-style constant-product state with 18-decimal "
    "tokens. Start with WETH, traverse each submitted pool once, and finish in WETH. "
    "For each hop: amount_with_fee = amount_in * (10000 - fee_bps); "
    "amount_out = floor(amount_with_fee * reserve_out / "
    "(reserve_in * 10000 + amount_with_fee)). Reserves update after the hop. "
    "Net profit is final WETH minus input WETH, EIP-1559 execution fee from the "
    "published deterministic gas model, and any explicit coinbase payment. "
    "The snapshot is immutable and evaluation performs no network calls."
)


def _address(namespace: str, seed: int) -> str:
    digest = hashlib.sha256(f"{namespace}:{seed}".encode()).hexdigest()
    return "0x" + digest[-40:]


def _token(name: str, seed: int) -> str:
    return _address(f"token:{name}", seed)


def _pool(
    name: str, seed: int, asset0: str, asset1: str,
    reserve0: int, reserve1: int, *, fee_bps: int = 30,
) -> dict[str, Any]:
    return {
        "id": _address(f"pool:{name}", seed),
        "asset0": asset0,
        "asset1": asset1,
        "reserve0": str(reserve0),
        "reserve1": str(reserve1),
        "fee_bps": fee_bps,
    }


def _cycle(
    label: str, seed: int, hops: int, *, premium_bps: int,
    weth_liquidity: int, token_liquidity: int,
) -> list[dict[str, Any]]:
    intermediates = [_token(f"{label}:{index}", seed) for index in range(hops - 1)]
    tokens = [WETH, *intermediates, WETH]
    pools = []
    for index, (asset0, asset1) in enumerate(zip(tokens, tokens[1:])):
        if index == 0:
            reserve0 = weth_liquidity
            reserve1 = token_liquidity
        elif index == hops - 1:
            reserve0 = token_liquidity
            reserve1 = weth_liquidity * premium_bps // 10_000
        else:
            reserve0 = token_liquidity
            reserve1 = token_liquidity
        pools.append(_pool(
            f"{label}:{index}", seed, asset0, asset1, reserve0, reserve1))
    return pools


def _graph(family: str, seed: int) -> tuple[list[dict[str, Any]], int]:
    rng = random.Random(seed * 1009 + FAMILIES.index(family))
    jitter = rng.randint(0, 35)
    if family == "two_pool_cycle":
        return _cycle(
            family, seed, 2, premium_bps=10_900 + jitter,
            weth_liquidity=1_200 * WAD, token_liquidity=1_200_000 * WAD,
        ), 40 * WAD
    if family == "triangular_cycle":
        return _cycle(
            family, seed, 3, premium_bps=11_050 + jitter,
            weth_liquidity=1_500 * WAD, token_liquidity=1_500_000 * WAD,
        ), 40 * WAD
    if family == "four_hop_cycle":
        return _cycle(
            family, seed, 4, premium_bps=11_250 + jitter,
            weth_liquidity=1_800 * WAD, token_liquidity=1_800_000 * WAD,
        ), 45 * WAD
    if family == "size_sensitive_cycle":
        pools = _cycle(
            family, seed, 3, premium_bps=12_000 + jitter,
            weth_liquidity=90 * WAD, token_liquidity=90_000 * WAD,
        )
        return pools, 25 * WAD
    if family == "competing_cycles":
        strong = _cycle(
            f"{family}:strong", seed, 3, premium_bps=11_100 + jitter,
            weth_liquidity=1_300 * WAD, token_liquidity=1_300_000 * WAD,
        )
        weak = _cycle(
            f"{family}:weak", seed, 2, premium_bps=10_350 + jitter,
            weth_liquidity=1_000 * WAD, token_liquidity=1_000_000 * WAD,
        )
        return strong + weak, 40 * WAD
    raise ValueError(f"unknown DEX arbitrage family {family!r}")


def _stress_snapshots(
    pools: list[dict[str, Any]], fee_model: dict[str, Any],
) -> list[dict[str, Any]]:
    higher_fee = deepcopy(pools)
    for pool in higher_fee:
        pool["fee_bps"] = int(pool["fee_bps"]) + 5
    adverse_reserves = deepcopy(pools)
    for index, pool in enumerate(adverse_reserves):
        if index % 2:
            pool["reserve0"] = str(int(pool["reserve0"]) * 998 // 1000)
        else:
            pool["reserve1"] = str(int(pool["reserve1"]) * 998 // 1000)
    expensive_fee_model = {
        **fee_model,
        "base_fee_per_gas_wei": str(
            int(fee_model["base_fee_per_gas_wei"]) * 3 // 2),
    }
    return [
        {"id": "higher_pool_fee", "pools": higher_fee,
         "fee_model": deepcopy(fee_model)},
        {"id": "adverse_reserves", "pools": adverse_reserves,
         "fee_model": deepcopy(fee_model)},
        {"id": "higher_base_fee", "pools": deepcopy(pools),
         "fee_model": expensive_fee_model},
    ]


def gold_route_from_public(public: dict[str, Any]) -> dict[str, Any]:
    solved = solve_best_route(public)
    return {
        "pools": solved["route"],
        "amount_in_wei": str(solved["amount_in_wei"]),
    }


def build_family_task(family: str, seed: int, split: str) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError(f"unknown DEX arbitrage family {family!r}")
    if split not in {"train", "benchmark", "qualification"}:
        raise ValueError(f"unknown split {split!r}")
    # The pool graph must differ across splits, not just across seeds:
    # otherwise train seed 1 and benchmark seed 1 share addresses, reserves
    # and the optimal route, and the benchmark measures memorisation.
    graph_seed = int(hashlib.sha256(
        f"dex-graph:{split}:{family}:{seed}".encode()).hexdigest()[:12], 16)
    pools, max_amount = _graph(family, graph_seed)
    block_number = 22_000_000 + seed * 17 + FAMILIES.index(family)
    block_hash = hashlib.sha256(
        f"dex-block:{split}:{family}:{seed}".encode()).hexdigest()
    gas_model = {
        "base_gas": 65_000,
        "per_hop_gas": 45_000,
        "model": "bounded-v2-router-v1",
    }
    fee_model = {
        "base_fee_per_gas_wei": str((24 + seed % 11) * 10**9),
        "max_fee_per_gas_wei": str(80 * 10**9),
        "max_priority_fee_per_gas_wei": str(2 * 10**9),
        "explicit_coinbase_payment_wei": "0",
    }
    public = {
        "instruction": _INSTRUCTIONS[family],
        "family": family,
        "vertical": VERTICAL,
        "base_asset": WETH,
        "snapshot": {
            "kind": "synthetic_uniswap_v2",
            "chain_id": 1,
            "block_number": block_number,
            "block_hash": "0x" + block_hash,
            "timestamp": 1_775_000_000 + seed * 12,
            "evm_spec": "cancun",
            "pools": pools,
        },
        "constraints": {
            "min_hops": 2,
            "max_hops": 4,
            "min_amount_in_wei": str(10**15),
            "max_amount_in_wei": str(max_amount),
            "simple_pool_route": True,
            "start_and_end_asset": WETH,
        },
        "gas_model": gas_model,
        "fee_model": fee_model,
        "semantics": _SEMANTICS,
        "policy": (
            "Submit only typed pool addresses and a canonical decimal amount. "
            "Python, Solidity, shell, RPC URLs, private keys, signed transactions, "
            "and candidate-supplied PnL are rejected."
        ),
        "permissions": list(ACTION_SCHEMA["properties"]["type"]["enum"]),
        "budgets": {
            **default_budgets(), "calls_soft": 4, "calls_hard": 8,
            "latency_soft_ms": 20_000, "latency_hard_ms": 180_000,
        },
    }
    oracle = solve_best_route(public)
    if oracle["net_profit_wei"] <= 0:
        raise ValueError("trusted DEX task has no profitable route")
    stress_snapshots = _stress_snapshots(pools, fee_model)
    hidden = {
        "evidence_id": f"dex_evd_{split}_{family}_{seed}",
        "oracle_route": oracle["route"],
        "oracle_amount_in_wei": oracle["amount_in_wei"],
        "oracle_net_profit_wei": oracle["net_profit_wei"],
        "stress_snapshots": stress_snapshots,
        "oracle_engine": "bounded-simple-cycle-integer-search-v1",
    }
    task_id = f"task_{split}_{family}_{seed}"
    snapshot = {
        "task_id": task_id,
        "family": family,
        "runtime": {},
        "evaluations": [],
        "best_evaluation": None,
    }
    body = {
        "vertical": VERTICAL,
        "family": family,
        "split": split,
        "seed": seed,
        "source_lineage": f"dex-arbitrage.synthetic.{split}.{family}.{seed}",
        "entity_identity": f"dex-arbitrage-{split}-{family}-{seed}",
        "public": public,
        "snapshot": snapshot,
    }
    return {
        "id": task_id,
        "digest": sha256_digest({**body, "hidden": hidden}),
        "public_digest": sha256_digest(body),
        "split": split,
        "family": family,
        "vertical": VERTICAL,
        "source_lineage": body["source_lineage"],
        "entity_identity": body["entity_identity"],
        "public": public,
        "hidden": hidden,
        "snapshot": snapshot,
        "template_family": family,
        "semantic_digest": sha256_digest({
            "family": family,
            "snapshot": public["snapshot"],
            "constraints": public["constraints"],
            "gas_model": gas_model,
            "fee_model": fee_model,
        }),
    }


def build_taskset(
    split: str, *, per_family: int = 1, start_seed: int = 1,
) -> list[dict[str, Any]]:
    tasks = []
    seed = start_seed
    for family in FAMILIES:
        for _ in range(per_family):
            tasks.append(build_family_task(family, seed, split))
            seed += 1
    return tasks
