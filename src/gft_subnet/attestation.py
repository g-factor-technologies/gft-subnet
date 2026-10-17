"""Confidential GPU gate. Testnet may skip; prod must attest before LoRA leaves."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from gft_subnet.policy import PolicyError

MODE_NONE = "none"
MODE_CONFIDENTIAL_GPU = "confidential_gpu"


@dataclass(frozen=True)
class MinerAttestation:
    mode: str = MODE_NONE
    platform: str = ""
    quote: str = ""
    device: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MinerAttestation":
        row = dict(payload or {})
        return cls(
            mode=str(row.get("mode") or MODE_NONE),
            platform=str(row.get("platform") or ""),
            quote=str(row.get("quote") or ""),
            device=str(row.get("device") or ""),
        )


def assert_may_push_lora(
    attestation: MinerAttestation, *, require_confidential: bool,
) -> None:
    """L40S testnet skips this. Prod must present a confidential GPU quote."""
    if not require_confidential:
        return
    if attestation.mode != MODE_CONFIDENTIAL_GPU or not attestation.quote:
        raise PolicyError(
            "refusing LoRA push without confidential GPU attestation "
            f"(device={attestation.device or 'unknown'})"
        )
    verify_confidential_quote(attestation)


def verify_confidential_quote(attestation: MinerAttestation) -> None:
    del attestation
    # Hardware quote check is not wired. Fail closed so a flag cannot fake trust.
    raise PolicyError("confidential GPU quote verification is not wired yet")
