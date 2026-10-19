"""Exact integer constant-product routing and bounded oracle search."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from gft_subnet.gyms.dex_arbitrage._support import ContractError

FEE_DENOMINATOR = 10_000


def amount_out(
    amount_in: int, reserve_in: int, reserve_out: int, fee_bps: int,
) -> int:
    """Uniswap V2-style exact-input output with integer floor semantics."""
    if amount_in <= 0:
        raise ContractError("amount_in must be positive")
    if reserve_in <= 0 or reserve_out <= 0:
        raise ContractError("pool reserves must be positive")
    if not 0 <= fee_bps < FEE_DENOMINATOR:
        raise ContractError("pool fee_bps is outside [0, 10000)")
    amount_with_fee = amount_in * (FEE_DENOMINATOR - fee_bps)
    denominator = reserve_in * FEE_DENOMINATOR + amount_with_fee
    output = amount_with_fee * reserve_out // denominator
    if output <= 0 or output >= reserve_out:
        raise ContractError("swap output is empty or exhausts the pool")
    return output


def _pool_map(pools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(pool["id"]).lower(): deepcopy(pool) for pool in pools}


def execute_route(
    pools: list[dict[str, Any]], route: list[str], amount_in_wei: int,
    base_asset: str,
) -> dict[str, Any]:
    """Execute a simple pool route and return an auditable reserve trace."""
    state = _pool_map(pools)
    token = base_asset.lower()
    amount = int(amount_in_wei)
    trace = []
    seen: set[str] = set()
    for hop, raw_pool_id in enumerate(route):
        pool_id = raw_pool_id.lower()
        if pool_id in seen:
            raise ContractError(f"pool {raw_pool_id} is repeated")
        seen.add(pool_id)
        pool = state.get(pool_id)
        if pool is None:
            raise ContractError(f"unknown pool {raw_pool_id}")
        asset0 = str(pool["asset0"]).lower()
        asset1 = str(pool["asset1"]).lower()
        reserve0 = int(pool["reserve0"])
        reserve1 = int(pool["reserve1"])
        k_before = reserve0 * reserve1
        if token == asset0:
            next_asset = asset1
            output = amount_out(amount, reserve0, reserve1, int(pool["fee_bps"]))
            pool["reserve0"] = reserve0 + amount
            pool["reserve1"] = reserve1 - output
            reserve_in, reserve_out = reserve0, reserve1
        elif token == asset1:
            next_asset = asset0
            output = amount_out(amount, reserve1, reserve0, int(pool["fee_bps"]))
            pool["reserve1"] = reserve1 + amount
            pool["reserve0"] = reserve0 - output
            reserve_in, reserve_out = reserve1, reserve0
        else:
            raise ContractError(
                f"pool {raw_pool_id} does not accept route token {token}")
        k_after = int(pool["reserve0"]) * int(pool["reserve1"])
        if k_after < k_before:
            raise ContractError(f"pool {raw_pool_id} violated constant-product accounting")
        trace.append({
            "hop": hop,
            "pool": pool_id,
            "asset_in": token,
            "asset_out": next_asset,
            "amount_in": amount,
            "amount_out": output,
            "reserve_in_before": reserve_in,
            "reserve_out_before": reserve_out,
            "fee_bps": int(pool["fee_bps"]),
            "k_before": k_before,
            "k_after": k_after,
        })
        token, amount = next_asset, output
    return {
        "final_asset": token,
        "final_amount_wei": amount,
        "trace": trace,
        "accounting_conserved": all(
            row["k_after"] >= row["k_before"] for row in trace),
    }


def gas_used_for_hops(gas_model: dict[str, Any], hops: int) -> int:
    return int(gas_model["base_gas"]) + int(gas_model["per_hop_gas"]) * hops


def effective_priority_fee(fee_model: dict[str, Any]) -> int:
    base_fee = int(fee_model["base_fee_per_gas_wei"])
    max_fee = int(fee_model["max_fee_per_gas_wei"])
    max_priority = int(fee_model["max_priority_fee_per_gas_wei"])
    if max_fee < base_fee:
        raise ContractError("max_fee_per_gas is below the snapshot base fee")
    return min(max_priority, max_fee - base_fee)


def route_net_profit(
    pools: list[dict[str, Any]], route: list[str], amount_in_wei: int,
    *, base_asset: str, gas_model: dict[str, Any],
    fee_model: dict[str, Any],
) -> dict[str, Any]:
    execution = execute_route(pools, route, amount_in_wei, base_asset)
    gas_used = gas_used_for_hops(gas_model, len(route))
    base_fee = int(fee_model["base_fee_per_gas_wei"])
    priority_fee = effective_priority_fee(fee_model)
    execution_fee = gas_used * (base_fee + priority_fee)
    coinbase_payment = int(fee_model.get("explicit_coinbase_payment_wei") or 0)
    gross_profit = execution["final_amount_wei"] - amount_in_wei
    net_profit = gross_profit - execution_fee - coinbase_payment
    return {
        **execution,
        "gas_used": gas_used,
        "effective_priority_fee_per_gas_wei": priority_fee,
        "execution_fee_wei": execution_fee,
        "explicit_coinbase_payment_wei": coinbase_payment,
        "gross_profit_wei": gross_profit,
        "net_profit_wei": net_profit,
    }


def enumerate_cycles(
    pools: list[dict[str, Any]], base_asset: str, *,
    min_hops: int = 2, max_hops: int = 4,
) -> list[list[str]]:
    """Enumerate simple pool/token cycles from and back to the base token."""
    base = base_asset.lower()
    normalized = [
        {**pool, "id": str(pool["id"]).lower(),
         "asset0": str(pool["asset0"]).lower(),
         "asset1": str(pool["asset1"]).lower()}
        for pool in pools
    ]
    found: set[tuple[str, ...]] = set()

    def visit(
        token: str, route: tuple[str, ...], seen_pools: frozenset[str],
        seen_assets: frozenset[str],
    ) -> None:
        if len(route) >= max_hops:
            return
        for pool in normalized:
            pool_id = pool["id"]
            if pool_id in seen_pools:
                continue
            if token == pool["asset0"]:
                next_asset = pool["asset1"]
            elif token == pool["asset1"]:
                next_asset = pool["asset0"]
            else:
                continue
            next_route = (*route, pool_id)
            if next_asset == base:
                if len(next_route) >= min_hops:
                    found.add(next_route)
                continue
            if next_asset in seen_assets:
                continue
            visit(
                next_asset, next_route, seen_pools | {pool_id},
                seen_assets | {next_asset},
            )

    visit(base, (), frozenset(), frozenset({base}))
    return [list(route) for route in sorted(found)]


def _best_amount(
    pools: list[dict[str, Any]], route: list[str], *, base_asset: str,
    min_amount: int, max_amount: int, gas_model: dict[str, Any],
    fee_model: dict[str, Any],
) -> dict[str, Any]:
    def score(amount: int) -> dict[str, Any]:
        return route_net_profit(
            pools, route, amount, base_asset=base_asset,
            gas_model=gas_model, fee_model=fee_model)

    grid_size = 96
    span = max_amount - min_amount
    grid = sorted({
        min_amount,
        max_amount,
        *(min_amount + span * index // grid_size
          for index in range(grid_size + 1)),
    })
    rows = [(amount, score(amount)) for amount in grid]
    best_index = max(
        range(len(rows)), key=lambda index: rows[index][1]["net_profit_wei"])
    lo = rows[max(0, best_index - 1)][0]
    hi = rows[min(len(rows) - 1, best_index + 1)][0]
    while hi - lo > 8:
        third = (hi - lo) // 3
        left = lo + third
        right = hi - third
        if score(left)["net_profit_wei"] < score(right)["net_profit_wei"]:
            lo = left + 1
        else:
            hi = right - 1
    candidates = rows + [(amount, score(amount)) for amount in range(lo, hi + 1)]
    amount, result = max(
        candidates, key=lambda row: (row[1]["net_profit_wei"], -row[0]))
    return {"amount_in_wei": amount, **result}


def solve_best_route(public: dict[str, Any]) -> dict[str, Any]:
    """Deterministic bounded oracle over every simple cycle and input size."""
    pools = public["snapshot"]["pools"]
    constraints = public["constraints"]
    cycles = enumerate_cycles(
        pools, public["base_asset"],
        min_hops=int(constraints["min_hops"]),
        max_hops=int(constraints["max_hops"]),
    )
    if not cycles:
        raise ContractError("task graph contains no valid base-token cycle")
    candidates = []
    for route in cycles:
        result = _best_amount(
            pools, route, base_asset=public["base_asset"],
            min_amount=int(constraints["min_amount_in_wei"]),
            max_amount=int(constraints["max_amount_in_wei"]),
            gas_model=public["gas_model"], fee_model=public["fee_model"],
        )
        candidates.append({"route": route, **result})
    return max(
        candidates,
        key=lambda row: (
            row["net_profit_wei"], -len(row["route"]),
            tuple(reversed(row["route"])), -row["amount_in_wei"],
        ),
    )
