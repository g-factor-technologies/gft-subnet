"""FastAPI helpers. Auth is `btauth/1` on every mutating path."""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from gft_subnet.auth import AuthBackend, AuthError
from gft_subnet.protocol import loads

HEALTH_PATH = "/health"


def attach_health(app: Any, *, role: str, hotkey: str, extra: Any = None) -> None:
    @app.get(HEALTH_PATH)
    def health() -> Any:
        payload = {"ok": True, "role": role, "hotkey": hotkey, "auth": "btauth/1"}
        extra_payload = extra() if callable(extra) else extra
        if extra_payload:
            payload.update(dict(extra_payload))
        return JSONResponse(payload)


async def signed_json(auth: AuthBackend, request: Any, *, path: str) -> tuple[str, dict[str, Any]]:
    body = await request.body()
    target = _request_target(request, path)
    try:
        hotkey = auth.verify(request.method, target, body, request.headers)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    try:
        payload = loads(body) if body else {}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return hotkey, payload


def _request_target(request: Any, fallback: str) -> str:
    scope = getattr(request, "scope", None) or {}
    raw = scope.get("raw_path")
    if not raw:
        return fallback
    target = raw.decode() if isinstance(raw, bytes) else str(raw)
    query = scope.get("query_string") or b""
    if query:
        target += "?" + (query.decode() if isinstance(query, bytes) else str(query))
    return target
