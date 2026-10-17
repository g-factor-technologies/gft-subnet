"""Load btcli wallets for testnet signing."""
from __future__ import annotations

from typing import Any


class WalletError(RuntimeError):
    pass


def load_wallet(name: str, hotkey: str = "default") -> Any:
    try:
        import bittensor as bt
    except ImportError as exc:
        raise WalletError("bittensor is not installed; uv sync --extra bittensor") from exc
    wallet_cls = getattr(bt, "Wallet", None)
    if wallet_cls is None:
        try:
            from bittensor.wallet import Wallet as wallet_cls
        except ImportError as exc:
            raise WalletError("bittensor.Wallet is missing") from exc
    try:
        return wallet_cls(name=name, hotkey=hotkey)
    except TypeError:
        return wallet_cls(name, hotkey)


def hotkey_ss58(wallet: Any) -> str:
    hotkey = getattr(wallet, "hotkey", None)
    address = getattr(hotkey, "ss58_address", None) or getattr(wallet, "hotkey_ss58", None)
    if not address:
        raise WalletError("wallet has no hotkey ss58 address")
    return str(address)


def coldkey_ss58(wallet: Any) -> str:
    cold = getattr(wallet, "coldkeypub", None) or getattr(wallet, "coldkey", None)
    address = getattr(cold, "ss58_address", None)
    if not address:
        raise WalletError("wallet has no coldkey ss58 address")
    return str(address)
