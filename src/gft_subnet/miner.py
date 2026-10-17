"""Miner process: one policy version, one action per turn, tokens + logprobs."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from gft_subnet.attestation import MinerAttestation
from gft_subnet.auth import AuthBackend, AuthError
from gft_subnet.chain import MetagraphSnapshot
from gft_subnet.http import attach_health, signed_json
from gft_subnet.policy import PolicyError, PolicyRegistry, read_adapter_files
from gft_subnet.protocol import PolicyManifest, TurnRequest, TurnResponse, dumps


#: Upper bound on turns per batched request (one miner group per turn).
MAX_BATCH = 64


def _tokens(text: str) -> list[int]:
    return [min(255, ord(char)) for char in text[:128]]


def _response(request: TurnRequest, hotkey: str, action: dict[str, Any], *, scripted: bool) -> TurnResponse:
    text = json.dumps(action, separators=(",", ":"), sort_keys=True)
    token_ids = _tokens(text)
    return TurnResponse(
        policy_version=request.policy_version,
        episode_id=request.episode_id,
        hotkey=hotkey,
        action=action,
        text=text,
        token_ids=token_ids,
        logprobs=[0.0 for _ in token_ids],
        scripted=scripted,
    )


class ScriptedActor:
    """Labeled non-policy actor. Gold uses the gym's reference solver; never send these rows to GRPO."""

    def __init__(self, hotkey: str, *, kind: str = "gold",
                 solver: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None) -> None:
        if kind not in {"gold", "dummy"}:
            raise ValueError(f"unknown scripted kind {kind}")
        if kind == "gold" and solver is None:
            raise ValueError("a gold scripted actor needs the gym's solver")
        self.hotkey = hotkey
        self.kind = kind
        self.solver = solver
        self.scripted = True
        self._queues: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def generate(self, request: TurnRequest) -> TurnResponse:
        queue = self._queues[request.episode_id]
        if not queue:
            if self.kind == "gold":
                queue.extend(self.solver(request.observation))
            else:
                queue.extend([{"type": "inspect_task"}, {"type": "finalize"}])
        action = queue.pop(0) if queue else {"type": "finalize"}
        return _response(request, self.hotkey, action, scripted=True)


class OnPolicyStubActor:
    """On-policy-shaped tokens without the hidden solver. Used for packing / GRPO tests."""

    def __init__(self, hotkey: str) -> None:
        self.hotkey = hotkey
        self.scripted = True
        self._seen: set[str] = set()

    def generate(self, request: TurnRequest) -> TurnResponse:
        first = request.episode_id not in self._seen
        self._seen.add(request.episode_id)
        action = {"type": "inspect_task"} if first else {"type": "finalize"}
        return _response(request, self.hotkey, action, scripted=True)


class MinerRuntime:
    def __init__(
        self,
        *,
        hotkey: str,
        auth: AuthBackend,
        policy: PolicyRegistry,
        actor: Any,
        allowed_callers: set[str] | None = None,
        attestation: MinerAttestation | None = None,
    ) -> None:
        self.hotkey = hotkey
        self.auth = auth
        self.policy = policy
        self.actor = actor
        self.scripted = bool(getattr(actor, "scripted", False))
        self.allowed_callers = set(allowed_callers or ())
        self._attestation = attestation or MinerAttestation()

    def attestation(self) -> MinerAttestation:
        return self._attestation

    def ack_policy(self, version: int, checksum: str, gym_id: str = "") -> None:
        manifest = self.policy.assert_no_mix(version, gym_id=gym_id or None)
        loader = getattr(self.actor, "load_policy", None)
        if callable(loader):
            loader(manifest)
        self.policy.ack(self.hotkey, version, checksum, gym_id=manifest.gym_id)

    def load_policy(self, manifest: PolicyManifest, files: dict[str, bytes] | None = None) -> None:
        try:
            current = self.policy.manifest(gym_id=manifest.gym_id)
            same = current.version == manifest.version and current.checksum == manifest.checksum
        except PolicyError:
            same = False
        if not same:
            if files:
                self.policy.adopt(manifest, files)
            else:
                raise PolicyError("miner refused a mixed or unknown policy")
        stored = self.policy.manifest(gym_id=manifest.gym_id)
        if stored.sampling != dict(manifest.sampling or {}):
            stored = self.policy.apply_sampling(manifest.sampling, gym_id=stored.gym_id)
        self.ack_policy(stored.version, stored.checksum, gym_id=stored.gym_id)

    def generate(self, request: TurnRequest) -> TurnResponse:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: list[TurnRequest]) -> list[TurnResponse]:
        """One policy check for the batch, then one batched actor call when supported."""
        if not requests:
            return []
        if len({(request.gym_id, request.policy_version) for request in requests}) != 1:
            raise PolicyError("a batch must use one gym policy version")
        gym_id = requests[0].gym_id or self.policy.gym_id
        manifest = self.policy.assert_no_mix(requests[0].policy_version, gym_id=gym_id)
        if self.hotkey not in self.policy.acked(manifest.version, gym_id=gym_id):
            raise PolicyError(f"{self.hotkey} has not acked {manifest.slot}")
        loader = getattr(self.actor, "load_policy", None)
        if callable(loader):
            loader(manifest)
        batched = getattr(self.actor, "generate_batch", None)
        responses = (batched(requests) if callable(batched)
                     else [self.actor.generate(request) for request in requests])
        if len(responses) != len(requests):
            raise ValueError("actor returned a different number of turns")
        for response in responses:
            if response.policy_version != manifest.version:
                raise PolicyError("actor mixed policy versions")
            response.hotkey = self.hotkey
            response.scripted = self.scripted
        return responses


def create_miner_app(runtime: MinerRuntime):
    app = FastAPI(title="gft-subnet miner")
    attach_health(
        app, role="miner", hotkey=runtime.hotkey,
        extra=lambda: {"attestation": runtime.attestation().to_dict()},
    )

    @app.post("/v1/policy")
    async def load_policy(request: Request):
        hotkey, payload = await signed_json(runtime.auth, request, path="/v1/policy")
        _assert_caller(runtime, hotkey)
        try:
            manifest = PolicyManifest.from_dict(payload)
            runtime.load_policy(manifest, _adapter_files(payload))
            stored = runtime.policy.manifest(gym_id=manifest.gym_id)
        except (PolicyError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse({
            "ok": True, "hotkey": runtime.hotkey, "version": stored.version,
            "gym_id": stored.gym_id, "slot": stored.slot,
        })

    @app.post("/v1/policy/ack")
    async def ack_policy(request: Request):
        hotkey, payload = await signed_json(runtime.auth, request, path="/v1/policy/ack")
        _assert_caller(runtime, hotkey)
        try:
            runtime.ack_policy(
                int(payload["version"]), str(payload["checksum"]),
                gym_id=str(payload.get("gym_id") or ""),
            )
        except (PolicyError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse({"ok": True, "hotkey": runtime.hotkey, "version": int(payload["version"])})

    @app.post("/v1/generate")
    async def generate(request: Request):
        hotkey, payload = await signed_json(runtime.auth, request, path="/v1/generate")
        _assert_caller(runtime, hotkey)
        try:
            if payload.get("adapter_files"):
                raise ValueError("generate cannot carry adapter files")
            turn = TurnRequest.from_dict(payload)
            response = runtime.generate(turn)
        except PolicyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(response.to_dict())

    @app.post("/v1/generate_batch")
    async def generate_batch(request: Request):
        hotkey, payload = await signed_json(runtime.auth, request, path="/v1/generate_batch")
        _assert_caller(runtime, hotkey)
        try:
            if payload.get("adapter_files"):
                raise ValueError("generate cannot carry adapter files")
            raw = payload.get("requests")
            if not isinstance(raw, list) or not raw or len(raw) > MAX_BATCH:
                raise ValueError(f"requests must be a list of 1..{MAX_BATCH} turns")
            turns = [TurnRequest.from_dict(item) for item in raw]
            responses = runtime.generate_batch(turns)
        except PolicyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"responses": [response.to_dict() for response in responses]})

    @app.post("/v1/ping")
    async def ping(request: Request):
        hotkey, payload = await signed_json(runtime.auth, request, path="/v1/ping")
        _assert_caller(runtime, hotkey)
        del payload
        return JSONResponse({"ok": True, "hotkey": runtime.hotkey, "caller": hotkey})

    return app


def _assert_caller(runtime: MinerRuntime, hotkey: str) -> None:
    if hotkey not in runtime.allowed_callers:
        raise HTTPException(status_code=403, detail="caller is not allowed")


class HttpMiner:
    """httpx/stdlib client for a remote miner axon."""

    def __init__(
        self,
        base_url: str,
        *,
        hotkey: str,
        auth: AuthBackend,
        scripted: bool = False,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.hotkey = hotkey
        self.auth = auth
        self.scripted = scripted
        self.timeout = timeout

    def ack_policy(self, version: int, checksum: str, gym_id: str = "") -> None:
        payload = {"version": version, "checksum": checksum}
        if gym_id:
            payload["gym_id"] = gym_id
        self._post("/v1/policy/ack", payload)

    def load_policy(self, manifest: PolicyManifest, files: dict[str, bytes] | None = None) -> None:
        payload = manifest.to_dict()
        if files is None and manifest.adapter_dir:
            files = read_adapter_files(manifest.adapter_dir)
        if files:
            import base64

            payload["adapter_files"] = {
                name: base64.b64encode(blob).decode("ascii") for name, blob in files.items()
            }
        self._post("/v1/policy", payload)

    def generate(self, request: TurnRequest) -> TurnResponse:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: list[TurnRequest]) -> list[TurnResponse]:
        payload = self._post(
            "/v1/generate_batch", {"requests": [request.to_dict() for request in requests]})
        responses = [TurnResponse.from_dict(item) for item in payload.get("responses") or []]
        if any(response.hotkey != self.hotkey for response in responses):
            raise AuthError("miner response hotkey mismatch")
        return responses

    def attestation(self) -> MinerAttestation:
        url = self.base_url + "/health"
        try:
            import httpx
        except ImportError:
            payload = _urllib_get_json(url, timeout=self.timeout)
        else:
            try:
                response = httpx.get(url, timeout=self.timeout)
            except httpx.TimeoutException as exc:
                raise TimeoutError("miner HTTP timeout") from exc
            except httpx.TransportError as exc:
                raise OSError("miner HTTP transport failed") from exc
            if response.status_code >= 400:
                raise AuthError(f"miner /health failed: {response.status_code} {response.text}")
            payload = response.json()
        return MinerAttestation.from_dict(payload.get("attestation") if isinstance(payload, dict) else None)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = dumps(payload)
        headers = self.auth.sign("POST", path, body, receiver_ss58=self.hotkey)
        try:
            import httpx
        except ImportError:
            return _urllib_json(self.base_url + path, body, headers, timeout=self.timeout)
        try:
            response = httpx.post(
                self.base_url + path, content=body, headers=headers, timeout=self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise TimeoutError("miner HTTP timeout") from exc
        except httpx.TransportError as exc:
            raise OSError("miner HTTP transport failed") from exc
        if response.status_code >= 400:
            raise AuthError(f"miner {path} failed: {response.status_code} {response.text}")
        return response.json()


def miners_from_metagraph(
    snapshot: MetagraphSnapshot,
    *,
    auth: AuthBackend,
    skip_hotkeys: set[str] | None = None,
    timeout: float = 30.0,
    scripted: bool = False,
) -> list[HttpMiner]:
    skip = skip_hotkeys or set()
    miners: list[HttpMiner] = []
    for neuron in snapshot.miners():
        if neuron.hotkey in skip or neuron.axon is None:
            continue
        miners.append(
            HttpMiner(
                f"http://{neuron.axon.ip}:{neuron.axon.port}",
                hotkey=neuron.hotkey,
                auth=auth,
                scripted=scripted,
                timeout=timeout,
            )
        )
    return miners


def _adapter_files(payload: dict[str, Any]) -> dict[str, bytes] | None:
    raw = payload.get("adapter_files")
    if not isinstance(raw, dict) or not raw:
        return None
    import base64

    files: dict[str, bytes] = {}
    for name, blob in raw.items():
        files[str(name)] = base64.b64decode(blob) if isinstance(blob, str) else bytes(blob)
    return files


def _urllib_json(url: str, body: bytes, headers: dict[str, str], *, timeout: float) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise AuthError(f"miner HTTP {exc.code}") from exc
    return json.loads(raw)


def _urllib_get_json(url: str, *, timeout: float) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise AuthError(f"miner HTTP {exc.code}") from exc
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise AuthError("miner /health must return a JSON object")
    return payload
