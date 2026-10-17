"""Validator HTTP surface: health, policy acks, chain register/serve/weights/metagraph."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from gft_subnet.auth import AuthBackend
from gft_subnet.chain import ChainClient, ChainError
from gft_subnet.http import attach_health, signed_json
from gft_subnet.ledger import Ledger
from gft_subnet.policy import PolicyError, PolicyRegistry


class ValidatorRuntime:
    def __init__(
        self,
        *,
        hotkey: str,
        auth: AuthBackend,
        chain: ChainClient,
        policy: PolicyRegistry,
        ledger: Ledger | None = None,
        allowed_callers: set[str] | None = None,
    ) -> None:
        self.hotkey = hotkey
        self.auth = auth
        self.chain = chain
        self.policy = policy
        self.ledger = ledger
        self.allowed_callers = set(allowed_callers or ()) | {hotkey}
        self.window_id: int | None = None

    def register(self) -> int:
        return self.chain.register(self.hotkey)

    def serve_axon(self, ip: str, port: int) -> Any:
        return self.chain.serve_axon(self.hotkey, ip, port)

    def set_weights(self, uids: list[int], weights: list[float]) -> int:
        return self.chain.set_weights(self.hotkey, uids, weights)

    def metagraph(self) -> Any:
        return self.chain.metagraph()

    def ack_miner(self, hotkey: str, version: int, checksum: str, gym_id: str = "") -> None:
        self.policy.ack(hotkey, version, checksum, gym_id=gym_id or None)
        if self.ledger is not None and self.window_id is not None:
            self.ledger.ack(self.window_id, hotkey, checksum)


def create_validator_app(runtime: ValidatorRuntime):
    app = FastAPI(title="gft-subnet validator")
    attach_health(app, role="validator", hotkey=runtime.hotkey)

    def _caller(hotkey: str) -> None:
        if hotkey not in runtime.allowed_callers:
            raise HTTPException(status_code=403, detail="caller is not allowed")

    def _admin(hotkey: str) -> None:
        if hotkey != runtime.hotkey:
            raise HTTPException(status_code=403, detail="validator owner required")

    @app.post("/v1/ping")
    async def ping(request: Request):
        hotkey, payload = await signed_json(runtime.auth, request, path="/v1/ping")
        _caller(hotkey)
        del payload
        return JSONResponse({"ok": True, "hotkey": runtime.hotkey, "caller": hotkey})

    @app.get("/v1/policy")
    def get_policy():
        try:
            manifest = runtime.policy.manifest()
        except PolicyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(manifest.to_dict())

    @app.post("/v1/policy/ack")
    async def ack_policy(request: Request):
        hotkey, payload = await signed_json(runtime.auth, request, path="/v1/policy/ack")
        _caller(hotkey)
        try:
            version = int(payload["version"])
            checksum = str(payload["checksum"])
            runtime.ack_miner(
                hotkey, version, checksum, gym_id=str(payload.get("gym_id") or ""),
            )
        except (PolicyError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse({"ok": True, "hotkey": hotkey, "version": version})

    @app.post("/v1/chain/register")
    async def register(request: Request):
        caller, payload = await signed_json(runtime.auth, request, path="/v1/chain/register")
        _admin(caller)
        hotkey = str(payload.get("hotkey") or runtime.hotkey)
        try:
            uid = runtime.chain.register(hotkey)
        except ChainError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"ok": True, "hotkey": hotkey, "uid": uid})

    @app.post("/v1/chain/serve-axon")
    async def serve_axon(request: Request):
        caller, payload = await signed_json(runtime.auth, request, path="/v1/chain/serve-axon")
        _admin(caller)
        hotkey = str(payload.get("hotkey") or runtime.hotkey)
        ip = str(payload["ip"])
        port = int(payload["port"])
        try:
            axon = runtime.chain.serve_axon(hotkey, ip, port)
        except (ChainError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"ok": True, "hotkey": axon.hotkey, "ip": axon.ip, "port": axon.port})

    @app.post("/v1/chain/set-weights")
    async def set_weights(request: Request):
        caller, payload = await signed_json(runtime.auth, request, path="/v1/chain/set-weights")
        _admin(caller)
        uids = [int(value) for value in payload["uids"]]
        weights = [float(value) for value in payload["weights"]]
        dummy = bool(payload.get("dummy"))
        try:
            block = runtime.set_weights(uids, weights)
        except (ChainError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"ok": True, "block": block, "uids": uids, "weights": weights, "dummy": dummy})

    @app.get("/v1/chain/metagraph")
    def metagraph():
        snapshot = runtime.metagraph()
        return JSONResponse({
            "netuid": snapshot.netuid,
            "block": snapshot.block,
            "neurons": [
                {
                    "uid": neuron.uid,
                    "hotkey": neuron.hotkey,
                    "coldkey": neuron.coldkey,
                    "validator_permit": neuron.validator_permit,
                    "axon": None if neuron.axon is None else {
                        "ip": neuron.axon.ip,
                        "port": neuron.axon.port,
                    },
                }
                for neuron in snapshot.neurons
            ],
        })

    return app
