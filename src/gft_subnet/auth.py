"""Request signatures. Prefer SDK v11 `bt.http_auth`; tests use HMAC."""
from __future__ import annotations

import hashlib
import hmac
import itertools
import time
from typing import Any, Mapping, Protocol

SCHEME = "btauth/1"
MAX_SKEW_SECONDS = 30
HEADER_AUTHORIZATION = "authorization"
HEADER_HOTKEY = "x-bt-hotkey"
HEADER_TIMESTAMP = "x-bt-timestamp"
HEADER_NONCE = "x-bt-nonce"


class AuthError(ValueError):
    pass


class AuthBackend(Protocol):
    def sign(
        self,
        method: str,
        path: str,
        body: bytes,
        receiver_ss58: str | None = None,
    ) -> dict[str, str]: ...

    def verify(self, method: str, path: str, body: bytes, headers: Mapping[str, str]) -> str: ...


def _header(headers: Mapping[str, str], name: str) -> str:
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return str(value)
    return ""


def _canonical(method: str, path: str, body: bytes, timestamp: str, nonce: str, hotkey: str) -> bytes:
    digest = hashlib.sha256(body).hexdigest()
    return f"{SCHEME}\n{method.upper()}\n{path}\n{timestamp}\n{nonce}\n{hotkey}\n{digest}".encode()


class LocalAuth:
    """HMAC stand-in for `bt.http_auth`. Same header names as SDK v11 tests use."""

    def __init__(self, keys: Mapping[str, bytes], *, self_hotkey: str, now: Any = None) -> None:
        self._keys = dict(keys)
        self.self_hotkey = self_hotkey
        self._now = now or time.time
        # Validator threads share one signer; next() on a count is atomic.
        self._counter = itertools.count(1)
        self._used: set[tuple[str, str]] = set()
        if self_hotkey not in self._keys:
            raise AuthError(f"unknown self hotkey {self_hotkey}")

    def sign(
        self,
        method: str,
        path: str,
        body: bytes,
        receiver_ss58: str | None = None,
    ) -> dict[str, str]:
        del receiver_ss58
        timestamp = str(int(self._now()))
        nonce = hashlib.sha256(
            f"{self.self_hotkey}:{timestamp}:{next(self._counter)}:{hashlib.sha256(body).hexdigest()}".encode()
        ).hexdigest()[:16]
        mac = hmac.new(
            self._keys[self.self_hotkey],
            _canonical(method, path, body, timestamp, nonce, self.self_hotkey),
            hashlib.sha256,
        ).hexdigest()
        return {
            HEADER_AUTHORIZATION: f"{SCHEME} {mac}",
            HEADER_HOTKEY: self.self_hotkey,
            HEADER_TIMESTAMP: timestamp,
            HEADER_NONCE: nonce,
        }

    def verify(self, method: str, path: str, body: bytes, headers: Mapping[str, str]) -> str:
        scheme_and_mac = _header(headers, HEADER_AUTHORIZATION)
        if not scheme_and_mac.startswith(f"{SCHEME} "):
            raise AuthError("missing btauth/1 signature")
        mac = scheme_and_mac.split(" ", 1)[1]
        hotkey = _header(headers, HEADER_HOTKEY)
        timestamp = _header(headers, HEADER_TIMESTAMP)
        nonce = _header(headers, HEADER_NONCE)
        if not hotkey or not timestamp or not nonce:
            raise AuthError("missing auth headers")
        secret = self._keys.get(hotkey)
        if secret is None:
            raise AuthError("unknown hotkey")
        try:
            ts = int(timestamp)
        except ValueError as exc:
            raise AuthError("bad timestamp") from exc
        if abs(int(self._now()) - ts) > MAX_SKEW_SECONDS:
            raise AuthError("timestamp skew")
        expected = hmac.new(
            secret,
            _canonical(method, path, body, timestamp, nonce, hotkey),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, mac):
            raise AuthError("bad signature")
        replay_key = (hotkey, nonce)
        if replay_key in self._used:
            raise AuthError("replay")
        self._used.add(replay_key)
        return hotkey


class BittensorAuth:
    """SDK v11 `bt.http_auth.sign` / `verify`. Headers are X-Bittensor-*."""

    def __init__(self, wallet: Any, *, self_hotkey: str | None = None) -> None:
        try:
            import bittensor as bt
        except ImportError as exc:
            raise AuthError("bittensor is not installed") from exc
        if not hasattr(bt, "http_auth"):
            raise AuthError("bittensor.http_auth is missing; need SDK v11+")
        self._bt = bt
        self._wallet = wallet
        self.self_hotkey = self_hotkey or str(wallet.hotkey.ss58_address)

    def sign(
        self,
        method: str,
        path: str,
        body: bytes,
        receiver_ss58: str | None = None,
    ) -> dict[str, str]:
        signed = self._bt.http_auth.sign(
            self._wallet,
            method=method,
            path=path,
            body=body,
            receiver_ss58=receiver_ss58,
        )
        if isinstance(signed, dict):
            return {str(key): str(value) for key, value in signed.items()}
        headers = dict(getattr(signed, "headers", {}) or {})
        return {str(key): str(value) for key, value in headers.items()}

    def verify(self, method: str, path: str, body: bytes, headers: Mapping[str, str]) -> str:
        verifier = self._bt.http_auth.verify
        auth_error = getattr(self._bt.http_auth, "AuthError", AuthError)
        try:
            result = verifier(
                dict(headers),
                body,
                method=method,
                path=path,
                self_hotkey_ss58=self.self_hotkey,
            )
        except auth_error as exc:
            raise AuthError(str(exc)) from exc
        if isinstance(result, str):
            return result
        hotkey = getattr(result, "hotkey_ss58", None) or getattr(result, "hotkey", None)
        if not hotkey:
            raise AuthError("http_auth.verify returned no hotkey")
        return str(hotkey)
