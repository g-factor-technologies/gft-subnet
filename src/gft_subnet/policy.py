"""Immutable policy files and an atomic, restartable registry. ACKs are per process."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path, PurePosixPath

from gft_subnet.protocol import DEFAULT_GYM_ID, PolicyManifest


class PolicyError(RuntimeError):
    pass


def _checksum(files: dict[str, bytes]) -> str:
    hasher = hashlib.sha256()
    for name, data in sorted(files.items()):
        hasher.update(name.encode())
        hasher.update(data)
    if not files:
        hasher.update(b"empty-adapter")
    return hasher.hexdigest()


def read_adapter_files(adapter_dir: str | Path) -> dict[str, bytes]:
    root = Path(adapter_dir)
    if not root.is_dir() or root.is_symlink():
        raise PolicyError(f"invalid adapter directory: {root}")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PolicyError("adapter symlinks are forbidden")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def adapter_checksum(adapter_dir: str | Path) -> str:
    return _checksum(read_adapter_files(adapter_dir))


def write_adapter_files(
    adapter_dir: str | Path, files: dict[str, bytes], *, version: int,
    expected_checksum: str | None = None,
) -> str:
    if version < 1:
        raise PolicyError("policy version must be positive")
    # Validate the entire package before touching the filesystem.
    for name, payload in files.items():
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or ".." in path.parts
                or path.as_posix() != name or "\\" in name or "\x00" in name
                or name == "." or not isinstance(payload, bytes)):
            raise PolicyError(f"invalid adapter file: {name!r}")
    files = {**files, "VERSION": str(version).encode()}
    checksum = _checksum(files)
    if expected_checksum is not None and checksum != expected_checksum:
        raise PolicyError("checksum mismatch")
    root = Path(adapter_dir)
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise PolicyError("adapter paths must not contain symlinks")
    if root.exists():
        if adapter_checksum(root) != checksum:
            raise PolicyError("cannot overwrite an immutable policy version")
        return checksum
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".policy-", dir=root.parent) as stage:
        staging = Path(stage) / "adapter"
        staging.mkdir()
        for name, payload in files.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        staging.rename(root)
    return checksum


def write_adapter(adapter_dir: str | Path, *, version: int, payload: bytes) -> str:
    """Legacy local-test fixture; never accepted by the model actor as LoRA."""
    return write_adapter_files(adapter_dir, {"adapter.bin": payload}, version=version)


class PolicyRegistry:
    def __init__(
        self, *, base_model: str, root: str | Path, sampling: dict | None = None,
        gym_id: str = DEFAULT_GYM_ID,
    ) -> None:
        self.base_model = base_model
        self.gym_id = str(gym_id or DEFAULT_GYM_ID)
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        self.sampling = dict(sampling or {"temperature": 0.7, "max_tokens": 512})
        self._manifests: dict[str, dict[int, PolicyManifest]] = {}
        self._current: dict[str, int] = {}
        self._acks: dict[tuple[str, int], set[str]] = {}
        state = self.root / "registry.json"
        if state.exists():
            payload = json.loads(state.read_text())
            for gym_id, row in _gyms_from_state(payload).items():
                for raw in row["manifests"]:
                    manifest = PolicyManifest.from_dict(raw)
                    if manifest.base_model != base_model:
                        raise PolicyError("registry base model differs from requested model")
                    if Path(manifest.adapter_dir) != self._adapter_dir(manifest.gym_id, manifest.version):
                        raise PolicyError("registry contains an invalid adapter path")
                    if adapter_checksum(manifest.adapter_dir) != manifest.checksum:
                        raise PolicyError("stored policy checksum mismatch")
                    self._manifests.setdefault(gym_id, {})[manifest.version] = manifest
                self._current[gym_id] = int(row["current_version"])
            if self.current_version:
                self.sampling = self.manifest().sampling

    @property
    def current_version(self) -> int:
        return int(self._current.get(self.gym_id, 0))

    def _adapter_dir(self, gym_id: str, version: int) -> Path:
        return self.root / gym_id / f"v{int(version)}"

    def _ack_key(self, gym_id: str, version: int) -> tuple[str, int]:
        return (gym_id, int(version))

    def _record(self, manifest: PolicyManifest) -> PolicyManifest:
        gym_id = manifest.gym_id
        gym_manifests = {**self._manifests.get(gym_id, {}), manifest.version: manifest}
        manifests = {**self._manifests, gym_id: gym_manifests}
        current = {**self._current, gym_id: manifest.version}
        payload = {
            "base_model": self.base_model,
            "gym_id": self.gym_id,
            "current_version": current.get(self.gym_id, 0),
            "gyms": {
                name: {
                    "current_version": current[name],
                    "manifests": [item.to_dict() for item in sorted(rows.values(), key=lambda m: m.version)],
                }
                for name, rows in manifests.items()
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", dir=self.root, delete=False) as handle:
            json.dump(payload, handle)
            temp = Path(handle.name)
        temp.replace(self.root / "registry.json")
        self._manifests = manifests
        self._current = current
        self._acks[self._ack_key(gym_id, manifest.version)] = set()
        if gym_id == self.gym_id:
            self.sampling = manifest.sampling
        return manifest

    def apply_sampling(self, sampling: dict, *, gym_id: str | None = None) -> PolicyManifest:
        """Sampling is not part of the adapter checksum; still apply a new token budget."""
        from dataclasses import replace

        gym_id = gym_id or self.gym_id
        current = self.manifest(gym_id=gym_id)
        updated = replace(current, sampling=dict(sampling))
        if current.sampling == updated.sampling:
            if gym_id == self.gym_id:
                self.sampling = dict(sampling)
            return current
        return self._record(updated)

    def _publish_files(
        self, version: int, files: dict[str, bytes], *, gym_id: str | None = None,
    ) -> PolicyManifest:
        gym_id = gym_id or self.gym_id
        current = int(self._current.get(gym_id, 0))
        if version != current + 1:
            raise PolicyError(f"cannot skip from {gym_id}-v{current} to {gym_id}-v{version}")
        target = self._adapter_dir(gym_id, version)
        checksum = write_adapter_files(target, files, version=version)
        return self._record(PolicyManifest(
            version, self.base_model, None, checksum, self.sampling, str(target), gym_id,
        ))

    def publish_base(self, gym_id: str | None = None) -> PolicyManifest:
        gym_id = gym_id or self.gym_id
        version = int(self._current.get(gym_id, 0)) + 1
        return self._publish_files(version, {"base_policy.json": json.dumps(
            {"base_model": self.base_model, "gym_id": gym_id}, sort_keys=True).encode()}, gym_id=gym_id)

    def publish(self, *, version: int, adapter_bytes: bytes, gym_id: str | None = None) -> PolicyManifest:
        return self._publish_files(version, {"adapter.bin": adapter_bytes}, gym_id=gym_id)

    def publish_from_dir(
        self, *, version: int, adapter_dir: str | Path, gym_id: str | None = None,
    ) -> PolicyManifest:
        return self._publish_files(version, read_adapter_files(adapter_dir), gym_id=gym_id)

    def adopt(self, manifest: PolicyManifest, files: dict[str, bytes] | None = None) -> PolicyManifest:
        if manifest.base_model != self.base_model:
            raise PolicyError("policy base model mismatch")
        current = int(self._current.get(manifest.gym_id, 0))
        if manifest.version < current:
            raise PolicyError("policy rollback is forbidden")
        if files is None:
            raise PolicyError("policy adopt needs adapter files, not a remote filesystem path")
        target = self._adapter_dir(manifest.gym_id, manifest.version)
        checksum = write_adapter_files(target, files, version=manifest.version,
                                       expected_checksum=manifest.checksum)
        stored = PolicyManifest(
            manifest.version, manifest.base_model, manifest.revision,
            checksum, manifest.sampling, str(target), manifest.gym_id,
        )
        return self._record(stored)

    def manifest(self, version: int | None = None, *, gym_id: str | None = None) -> PolicyManifest:
        gym_id = gym_id or self.gym_id
        target = int(self._current.get(gym_id, 0) if version is None else version)
        try:
            return self._manifests[gym_id][target]
        except KeyError as exc:
            raise PolicyError(f"unknown policy {gym_id}-v{target}") from exc

    def ack(self, hotkey: str, version: int, checksum: str, gym_id: str | None = None) -> None:
        gym_id = gym_id or self.gym_id
        manifest = self.manifest(version, gym_id=gym_id)
        if checksum != manifest.checksum:
            raise PolicyError("checksum mismatch")
        if version != int(self._current.get(gym_id, 0)):
            raise PolicyError("ack is not for the current policy")
        self._acks.setdefault(self._ack_key(gym_id, version), set()).add(hotkey)

    def acked(self, version: int, gym_id: str | None = None) -> set[str]:
        gym_id = gym_id or self.gym_id
        return set(self._acks.get(self._ack_key(gym_id, version)) or ())

    def assert_no_mix(self, requested: int, *, gym_id: str | None = None) -> PolicyManifest:
        gym_id = gym_id or self.gym_id
        manifest = self.manifest(gym_id=gym_id)
        if requested != manifest.version:
            raise PolicyError(
                f"policy mix: requested {gym_id}-v{requested}, current {manifest.slot}")
        return manifest

    def can_advance(self, miners: list[str], gym_id: str | None = None) -> bool:
        gym_id = gym_id or self.gym_id
        current = int(self._current.get(gym_id, 0))
        return bool(current) and all(hotkey in self.acked(current, gym_id=gym_id) for hotkey in miners)


def _gyms_from_state(payload: dict) -> dict[str, dict]:
    if payload.get("gyms"):
        return dict(payload["gyms"])
    gym_id = str(payload.get("gym_id") or DEFAULT_GYM_ID)
    return {gym_id: {"current_version": payload["current_version"], "manifests": payload["manifests"]}}
