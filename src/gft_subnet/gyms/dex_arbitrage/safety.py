"""Typed submission screening for DEX routes; no candidate code is executed."""
from __future__ import annotations

import re
from typing import Any

from gft_subnet.gyms.dex_arbitrage._support import ContractError

_ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")
_WEI = re.compile(r"^(0|[1-9][0-9]{0,77})$")


class UnsafeDexSubmission(Exception):
    """A route attempted to smuggle executable or unbounded content."""


def validate_submission(public: dict[str, Any], raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ContractError("route submission must be an object")
    unknown = set(raw) - {"pools", "amount_in_wei"}
    if unknown:
        raise ContractError(f"route submission has unknown fields {sorted(unknown)}")
    route = raw.get("pools")
    if not isinstance(route, list):
        raise ContractError("pools must be a list")
    max_hops = int(public["constraints"]["max_hops"])
    if len(route) > max_hops:
        raise UnsafeDexSubmission(f"route exceeds {max_hops} pools")
    min_hops = int(public["constraints"]["min_hops"])
    if len(route) < min_hops:
        raise ContractError(f"route requires at least {min_hops} pools")
    known = {
        str(pool["id"]).lower() for pool in public["snapshot"]["pools"]}
    checked_route = []
    for index, value in enumerate(route):
        if not isinstance(value, str):
            raise ContractError(f"pool #{index} must be an address string")
        if len(value) > 42:
            raise UnsafeDexSubmission(f"pool #{index} is oversized")
        if not _ADDRESS.fullmatch(value):
            raise ContractError(f"pool #{index} is not a 20-byte hex address")
        normalized = value.lower()
        if normalized not in known:
            raise ContractError(f"pool #{index} is not in the frozen snapshot")
        checked_route.append(normalized)
    if len(set(checked_route)) != len(checked_route):
        raise ContractError("a simple route cannot repeat a pool")

    amount = raw.get("amount_in_wei")
    if not isinstance(amount, str):
        raise ContractError("amount_in_wei must be a canonical decimal string")
    if len(amount) > 78:
        raise UnsafeDexSubmission("amount_in_wei is oversized")
    if not _WEI.fullmatch(amount):
        raise ContractError("amount_in_wei must be a canonical uint256 string")
    amount_value = int(amount)
    lower = int(public["constraints"]["min_amount_in_wei"])
    upper = int(public["constraints"]["max_amount_in_wei"])
    if not lower <= amount_value <= upper:
        raise ContractError(
            f"amount_in_wei must be inside [{lower}, {upper}]")
    return {"pools": checked_route, "amount_in_wei": amount_value}
