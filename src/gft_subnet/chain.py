"""Chain surface used by the MVP: register, serve axon, set_weights, metagraph."""
from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class ChainError(RuntimeError):
    pass


@dataclass(frozen=True)
class AxonInfo:
    hotkey: str
    ip: str
    port: int
    protocol: str = "http"


@dataclass
class Neuron:
    uid: int
    hotkey: str
    coldkey: str
    axon: AxonInfo | None = None
    validator_permit: bool = False
    stake: float = 0.0


@dataclass
class MetagraphSnapshot:
    netuid: int
    block: int
    neurons: list[Neuron] = field(default_factory=list)

    def by_hotkey(self, hotkey: str) -> Neuron | None:
        for neuron in self.neurons:
            if neuron.hotkey == hotkey:
                return neuron
        return None

    def uid_for(self, hotkey: str) -> int | None:
        neuron = self.by_hotkey(hotkey)
        return None if neuron is None else neuron.uid

    def miners(self) -> list[Neuron]:
        return [neuron for neuron in self.neurons if neuron.axon is not None]


class ChainClient(Protocol):
    netuid: int

    def register(self, hotkey: str, *, coldkey: str | None = None) -> int: ...

    def serve_axon(self, hotkey: str, ip: str, port: int) -> AxonInfo: ...

    def set_weights(self, validator_hotkey: str, uids: list[int], weights: list[float]) -> int: ...

    def metagraph(self) -> MetagraphSnapshot: ...

    def last_weights(self, validator_hotkey: str) -> tuple[list[int], list[float]] | None: ...


class LocalChain:
    """In-memory chain for tests. Optional JSON path so CLI commands share state."""

    def __init__(self, netuid: int = 1, path: str | Path | None = None) -> None:
        self.netuid = netuid
        self.path = Path(path) if path else None
        self.block = 0
        self._neurons: dict[str, Neuron] = {}
        self._weights: dict[str, tuple[list[int], list[float]]] = {}
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.netuid = int(payload.get("netuid") or self.netuid)
        self.block = int(payload.get("block") or 0)
        self._neurons = {}
        for row in payload.get("neurons") or []:
            axon_raw = row.get("axon")
            axon = None
            if axon_raw:
                axon = AxonInfo(hotkey=row["hotkey"], ip=str(axon_raw["ip"]), port=int(axon_raw["port"]))
            neuron = Neuron(
                uid=int(row["uid"]),
                hotkey=str(row["hotkey"]),
                coldkey=str(row.get("coldkey") or row["hotkey"]),
                axon=axon,
                validator_permit=bool(row.get("validator_permit")),
                stake=float(row.get("stake") or 0.0),
            )
            self._neurons[neuron.hotkey] = neuron
        self._weights = {
            hotkey: (list(values[0]), [float(weight) for weight in values[1]])
            for hotkey, values in (payload.get("weights") or {}).items()
        }

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "netuid": self.netuid,
            "block": self.block,
            "neurons": [
                {
                    "uid": neuron.uid,
                    "hotkey": neuron.hotkey,
                    "coldkey": neuron.coldkey,
                    "validator_permit": neuron.validator_permit,
                    "stake": neuron.stake,
                    "axon": None if neuron.axon is None else {"ip": neuron.axon.ip, "port": neuron.axon.port},
                }
                for neuron in sorted(self._neurons.values(), key=lambda item: item.uid)
            ],
            "weights": {hotkey: [uids, weights] for hotkey, (uids, weights) in self._weights.items()},
        }
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def register(self, hotkey: str, *, coldkey: str | None = None) -> int:
        existing = self._neurons.get(hotkey)
        if existing is not None:
            return existing.uid
        uid = len(self._neurons)
        self._neurons[hotkey] = Neuron(
            uid=uid,
            hotkey=hotkey,
            coldkey=coldkey or hotkey,
            validator_permit=uid == 0,
        )
        self.block += 1
        self._save()
        return uid

    def serve_axon(self, hotkey: str, ip: str, port: int) -> AxonInfo:
        neuron = self._neurons.get(hotkey)
        if neuron is None:
            raise ChainError(f"hotkey {hotkey} is not registered")
        axon = AxonInfo(hotkey=hotkey, ip=ip, port=port)
        self._neurons[hotkey] = Neuron(
            uid=neuron.uid,
            hotkey=neuron.hotkey,
            coldkey=neuron.coldkey,
            axon=axon,
            validator_permit=neuron.validator_permit,
            stake=neuron.stake,
        )
        self.block += 1
        self._save()
        return axon

    def set_weights(self, validator_hotkey: str, uids: list[int], weights: list[float], *, version_key: int = 0) -> int:
        del version_key
        if len(uids) != len(weights):
            raise ChainError("uids and weights length mismatch")
        validator = self._neurons.get(validator_hotkey)
        if validator is None:
            raise ChainError(f"validator {validator_hotkey} is not registered")
        if not validator.validator_permit:
            raise ChainError(f"{validator_hotkey} has no validator permit")
        known = {neuron.uid for neuron in self._neurons.values()}
        if any(uid not in known for uid in uids):
            raise ChainError("weight uid is not in the metagraph")
        self._weights[validator_hotkey] = (list(uids), [float(value) for value in weights])
        self.block += 1
        self._save()
        return self.block

    def metagraph(self) -> MetagraphSnapshot:
        return MetagraphSnapshot(
            netuid=self.netuid,
            block=self.block,
            neurons=sorted(self._neurons.values(), key=lambda neuron: neuron.uid),
        )

    def last_weights(self, validator_hotkey: str) -> tuple[list[int], list[float]] | None:
        return self._weights.get(validator_hotkey)

    def create_subnet(self) -> int:
        self.block += 1
        self._save()
        return self.netuid

    def start_subnet(self) -> None:
        self.block += 1
        self._save()


class BittensorChain:
    """SDK v11: Subtensor.execute(ServeAxon / SetWeights / BurnedRegister)."""

    def __init__(self, *, netuid: int, network: str = "test", wallet: Any = None) -> None:
        try:
            import bittensor as bt
        except ImportError as exc:
            raise ChainError("bittensor is not installed; uv sync --extra bittensor") from exc
        self._bt = bt
        self.netuid = netuid
        self.network = network
        self.wallet = wallet
        self._client = None
        self.last_result: Any = None

    def _sdk(self) -> Any:
        if self._client is not None:
            return self._client
        bt = self._bt
        if not hasattr(bt, "Subtensor"):
            raise ChainError("bittensor.Subtensor is missing; need SDK v11+")
        try:
            self._client = bt.Subtensor(network=self.network)
        except TypeError:
            self._client = bt.Subtensor(self.network)
        return self._client

    def _execute(self, intent: Any) -> Any:
        if self.wallet is None:
            raise ChainError("wallet is required for testnet chain writes")
        client = self._sdk()
        if not hasattr(client, "execute"):
            raise ChainError("Subtensor.execute is missing; need SDK v11+")
        result = _maybe_run(client.execute(intent, self.wallet))
        self.last_result = result
        if hasattr(result, "raise_for_failure"):
            result.raise_for_failure()
        elif hasattr(result, "success") and not result.success:
            error = getattr(result, "error", None)
            raise ChainError(str(getattr(error, "code", None) or error or result))
        return result

    def register(self, hotkey: str, *, coldkey: str | None = None) -> int:
        del coldkey
        snapshot = self.metagraph()
        uid = snapshot.uid_for(hotkey)
        if uid is None:
            raise ChainError(
                f"{hotkey} is not on netuid {self.netuid}. "
                "Run: python -m gft_subnet --chain test burned-register"
            )
        return uid

    def burned_register(self) -> int:
        bt = self._bt
        if not hasattr(bt, "BurnedRegister"):
            raise ChainError("BurnedRegister is missing; use btcli tx burned-register")
        self._execute(bt.BurnedRegister(netuid=self.netuid))
        hotkey = str(self.wallet.hotkey.ss58_address)
        return self.register(hotkey)

    def create_subnet(self) -> int:
        bt = self._bt
        intent_cls = getattr(bt, "RegisterSubnet", None) or getattr(bt, "CreateSubnet", None)
        if intent_cls is None:
            raise ChainError("RegisterSubnet is missing; use btcli subnet create --network test")
        result = self._execute(intent_cls())
        netuid = (getattr(result, "data", None) or {}).get("netuid")
        if netuid is None:
            netuid = getattr(result, "netuid", None)
        if netuid is None:
            events = getattr(result, "events", None) or {}
            netuid = events.get("netuid") if isinstance(events, dict) else None
        if netuid is None:
            raise ChainError("subnet created but netuid was not returned; check btcli")
        self.netuid = int(netuid)
        return self.netuid

    def start_subnet(self) -> Any:
        bt = self._bt
        intent_cls = getattr(bt, "StartCall", None) or getattr(bt, "StartSubnet", None)
        if intent_cls is None:
            raise ChainError("StartCall is missing; use btcli tx start-call --netuid N")
        try:
            return self._execute(intent_cls(netuid=self.netuid))
        except TypeError:
            return self._execute(intent_cls())

    def serve_axon(self, hotkey: str, ip: str, port: int) -> AxonInfo:
        del hotkey
        bt = self._bt
        if not hasattr(bt, "ServeAxon"):
            raise ChainError("ServeAxon is missing; need SDK v11+")
        self._execute(bt.ServeAxon(netuid=self.netuid, ip=ip, port=port))
        return AxonInfo(hotkey=str(self.wallet.hotkey.ss58_address), ip=ip, port=port)

    def set_weights(
        self,
        validator_hotkey: str,
        uids: list[int],
        weights: list[float],
        *,
        version_key: int = 0,
    ) -> int:
        del validator_hotkey
        bt = self._bt
        if not hasattr(bt, "SetWeights"):
            raise ChainError("SetWeights is missing; need SDK v11+")
        mapping = {int(uid): float(weight) for uid, weight in zip(uids, weights, strict=True)}
        intent = _set_weights_intent(bt, self.netuid, mapping, uids, weights, version_key)
        result = self._execute(intent)
        block = getattr(result, "block", None) or getattr(self._sdk(), "block", 0)
        return int(block or 0)

    def metagraph(self) -> MetagraphSnapshot:
        client = self._sdk()
        raw = None
        if hasattr(client, "subnets") and hasattr(client.subnets, "metagraph"):
            raw = _maybe_run(client.subnets.metagraph(self.netuid))
        elif hasattr(client, "metagraph"):
            raw = _maybe_run(client.metagraph(self.netuid))
        else:
            raise ChainError("metagraph is missing on this SDK")
        snapshot = _parse_metagraph(self.netuid, raw)
        if not snapshot.block and hasattr(client, "block"):
            snapshot.block = int(client.block or 0)
        return snapshot

    def last_weights(self, validator_hotkey: str) -> tuple[list[int], list[float]] | None:
        del validator_hotkey
        client = self._sdk()
        weights = getattr(getattr(client, "weights", None), "weights", None)
        if weights is None:
            return None
        raw = _maybe_run(weights(netuid=self.netuid))
        if isinstance(raw, dict):
            uids = sorted(int(uid) for uid in raw)
            return uids, [float(raw[uid]) for uid in uids]
        return None

    def explorer_url(self) -> str | None:
        result = self.last_result
        return None if result is None else getattr(result, "explorer_url", None)


def _maybe_run(value: Any) -> Any:
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def _set_weights_intent(
    bt: Any,
    netuid: int,
    mapping: dict[int, float],
    uids: list[int],
    weights: list[float],
    version_key: int,
) -> Any:
    attempts = (
        {"netuid": netuid, "weights": mapping, "version_key": version_key},
        {"netuid": netuid, "weights": mapping, "mechid": 0, "version_key": version_key},
        {"netuid": netuid, "uids": uids, "weights": weights, "version_key": version_key},
        {"netuid": netuid, "uids": uids, "weights": weights},
    )
    last_error: TypeError | None = None
    for kwargs in attempts:
        try:
            return bt.SetWeights(**kwargs)
        except TypeError as exc:
            last_error = exc
    raise ChainError(f"SetWeights signature mismatch: {last_error}") from last_error


def _parse_metagraph(netuid: int, raw: Any) -> MetagraphSnapshot:
    block = int(getattr(raw, "block", 0) or 0)
    neurons: list[Neuron] = []
    rows = _metagraph_rows(raw)
    if rows:
        for uid, row in enumerate(rows):
            neurons.append(_neuron_from_row(uid, row))
        return MetagraphSnapshot(netuid=netuid, block=block, neurons=neurons)
    axons = getattr(raw, "axons", None)
    hotkeys = list(getattr(raw, "hotkeys", []) or [])
    coldkeys = list(getattr(raw, "coldkeys", []) or [])
    permits = list(getattr(raw, "validator_permit", []) or [])
    stakes = list(getattr(raw, "S", []) or getattr(raw, "stake", []) or [])
    for uid, hotkey in enumerate(hotkeys):
        axon_raw = axons[uid] if axons is not None and uid < len(axons) else None
        neurons.append(
            Neuron(
                uid=uid,
                hotkey=str(hotkey),
                coldkey=str(coldkeys[uid]) if uid < len(coldkeys) else str(hotkey),
                axon=_parse_axon(str(hotkey), axon_raw),
                validator_permit=bool(permits[uid]) if uid < len(permits) else False,
                stake=_stake_value(stakes[uid] if uid < len(stakes) else 0.0),
            )
        )
    return MetagraphSnapshot(netuid=netuid, block=block, neurons=neurons)


def _metagraph_rows(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    for name in ("neurons", "N"):
        rows = getattr(raw, name, None)
        if rows:
            return list(rows)
    if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes, dict)):
        try:
            return list(raw)
        except TypeError:
            return []
    return []


def _neuron_from_row(uid: int, row: Any) -> Neuron:
    if isinstance(row, dict):
        hotkey = str(row.get("hotkey") or "")
        axon = _parse_axon(hotkey, row.get("axon"))
        return Neuron(
            uid=int(row.get("uid", uid)),
            hotkey=hotkey,
            coldkey=str(row.get("coldkey") or hotkey),
            axon=axon,
            validator_permit=bool(row.get("validator_permit")),
            stake=_stake_value(row.get("stake") or 0.0),
        )
    hotkey = str(getattr(row, "hotkey", "") or "")
    axon = _parse_axon(hotkey, getattr(row, "axon", None))
    uid_value = getattr(row, "uid", uid)
    return Neuron(
        uid=int(uid_value),
        hotkey=hotkey,
        coldkey=str(getattr(row, "coldkey", "") or hotkey),
        axon=axon,
        validator_permit=bool(getattr(row, "validator_permit", False)),
        stake=_stake_value(getattr(row, "tao_stake", None) or getattr(row, "stake", 0.0)),
    )


def _parse_axon(hotkey: str, axon_raw: Any) -> AxonInfo | None:
    if axon_raw is None or axon_raw is False:
        return None
    if isinstance(axon_raw, str):
        if ":" not in axon_raw:
            return None
        ip, port = axon_raw.rsplit(":", 1)
        if not ip or not port.isdigit():
            return None
        return AxonInfo(hotkey=hotkey, ip=ip, port=int(port))
    ip = str(getattr(axon_raw, "ip", "") or (axon_raw.get("ip") if isinstance(axon_raw, dict) else "") or "")
    port = int(getattr(axon_raw, "port", 0) or (axon_raw.get("port") if isinstance(axon_raw, dict) else 0) or 0)
    if ip and port:
        return AxonInfo(hotkey=hotkey, ip=ip, port=port)
    return None


def _stake_value(value: Any) -> float:
    if value is None:
        return 0.0
    tao = getattr(value, "tao", None)
    if tao is not None:
        return float(tao)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
