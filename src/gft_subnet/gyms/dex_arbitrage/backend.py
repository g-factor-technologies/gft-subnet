"""Reference AMM verifier plus fail-closed Foundry/revm arithmetic replay."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

from gft_subnet.gyms.dex_arbitrage._support import ContractError, sha256_digest
from gft_subnet.gyms.dex_arbitrage.math import route_net_profit
from gft_subnet.gyms.dex_arbitrage.safety import validate_submission

MODE_ENV = "SKILLHUB_DEX_ARBITRAGE_MODE"  # auto | reference | native
SOLIDITY_VERSION = "0.8.26"
# Address-space cap for the forge child, in GiB. ``RLIMIT_AS`` counts virtual
# reservations, not resident memory: forge is a Rust binary on jemalloc, which
# reserves arenas per CPU, plus a rayon pool with one stack per CPU. On a
# 6-CPU box forge needs ~1.7 GiB of address space just to start; with 16
# threads or 64 arenas (a 16-vCPU host) it aborts under the old 2 GiB cap with
# "memory allocation of N bytes failed" before compiling anything. The
# default is therefore *no* address-space limit; the container cgroup
# (--memory) and the wall-clock timeout bound the child instead. Set the env
# to a positive number of GiB to reinstate a cap (0/unlimited disables it).
FORGE_RLIMIT_AS_ENV = "SKILLHUB_FORGE_RLIMIT_AS_GIB"
FORGE_RLIMIT_NOFILE_ENV = "SKILLHUB_FORGE_RLIMIT_NOFILE"
FORGE_THREADS_ENV = "SKILLHUB_FORGE_THREADS"
FORGE_RLIMIT_NOFILE_DEFAULT = 1024
FORGE_RLIMIT_NOFILE_MIN = 1024
FORGE_THREADS_DEFAULT = 2
FORGE_TIMEOUT_SECONDS = 60
_ERROR_DETAIL_LINES = 40
_ERROR_DETAIL_BYTES = 4096
_FORGE_GAS = re.compile(
    r"\[PASS\]\s+testRouteReplay\(\)\s+\(gas:\s*(\d+)\)")
_FORGE_SUMMARY = re.compile(r"(\d+) passed;\s*(\d+) failed")
_FORGE_NO_MATCH = re.compile(r"no tests match", re.IGNORECASE)
# ``--match-test`` is applied to the ABI signature (``testRouteReplay()``),
# not the bare function name, so ``^testRouteReplay$`` selects nothing (forge
# then prints "no tests match", exits 0, and nothing runs). The optional
# ``()`` keeps the anchored pattern valid for a bare-name matcher as well.
FORGE_TEST_PATTERN = r"^testRouteReplay(\(\))?$"
# Keys of the native replay that are evidence about *this host* (forge's
# output tail carries wall-clock timings, the rlimits are per-host), not
# part of the verdict.  Stripped from the policy-visible evaluation and the
# evaluation digest, kept in the ``evaluation.json`` artifact.
_DIAGNOSTIC_KEYS = frozenset({"error_detail", "limits"})


def _tool_path(name: str, env_name: str) -> str | None:
    configured = str(os.environ.get(env_name) or "").strip()
    return configured or shutil.which(name)


def _version(path: str) -> str:
    try:
        completed = subprocess.run(
            [path, "--version"], capture_output=True, text=True,
            timeout=10, check=False, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    lines = [
        line.strip() for line in (completed.stdout + completed.stderr).splitlines()
        if line.strip()
    ]
    return " | ".join(lines)[:240] if lines else f"exit-{completed.returncode}"


@lru_cache(maxsize=1)
def native_runtime_info() -> dict[str, Any]:
    mode = str(os.environ.get(MODE_ENV) or "auto").strip().lower()
    if mode not in {"auto", "reference", "native"}:
        raise RuntimeError(f"{MODE_ENV} must be auto, reference, or native")
    forge = _tool_path("forge", "SKILLHUB_FORGE_BIN")
    solc = _tool_path("solc", "SKILLHUB_SOLC_BIN")
    versions = {
        name: _version(path)
        for name, path in (("forge", forge), ("solc", solc)) if path
    }
    solc_ok = f"Version: {SOLIDITY_VERSION}" in versions.get("solc", "")
    missing = []
    if not forge:
        missing.append("forge")
    if not solc:
        missing.append("solc")
    elif not solc_ok:
        missing.append(f"solc-{SOLIDITY_VERSION}")
    native = not missing and mode != "reference"
    if mode == "native" and not native:
        raise RuntimeError(
            f"{MODE_ENV}=native but runtime is incomplete: {', '.join(missing)}")
    return {
        "mode": mode,
        "native": native,
        "promotion_eligible": native,
        "forge_path": forge,
        "solc_path": solc,
        "versions": versions,
        "missing": missing,
        "reference_verifier": "python-exact-integer-amm",
        "native_verifier": "foundry-revm-solidity-arithmetic-replay",
        "network_required_at_runtime": False,
    }


def _env_int(env: Mapping[str, str], name: str, default: int, *,
             minimum: int) -> int:
    raw = str(env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    return value


def forge_resource_limits(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Resolve the forge child's rlimits and thread count from the environment.

    Returns a JSON-friendly dict: ``address_space_bytes`` (``None`` = no
    ``RLIMIT_AS``), ``nofile`` (never below 1024: forge opens the solc
    artifacts, cache and one pipe pair per worker), ``fsize_bytes``,
    ``threads`` (exported as ``RAYON_NUM_THREADS``; small keeps the pool's
    stack reservations bounded and the run deterministic).
    """
    env = os.environ if environ is None else environ
    raw_as = str(env.get(FORGE_RLIMIT_AS_ENV) or "").strip().lower()
    if raw_as in {"", "0", "unlimited", "none", "off"}:
        address_space = None
    else:
        try:
            gib = float(raw_as)
        except ValueError as exc:
            raise RuntimeError(
                f"{FORGE_RLIMIT_AS_ENV} must be a number of GiB or "
                f"unlimited, got {raw_as!r}") from exc
        if gib <= 0 or gib != gib:
            raise RuntimeError(
                f"{FORGE_RLIMIT_AS_ENV} must be positive, got {raw_as!r}")
        address_space = int(gib * 1024 * 1024 * 1024)
    nofile = _env_int(env, FORGE_RLIMIT_NOFILE_ENV, FORGE_RLIMIT_NOFILE_DEFAULT,
                      minimum=FORGE_RLIMIT_NOFILE_MIN)
    threads = _env_int(env, FORGE_THREADS_ENV, FORGE_THREADS_DEFAULT, minimum=1)
    return {
        "address_space_bytes": address_space,
        "address_space_gib": (None if address_space is None
                              else address_space / (1024 ** 3)),
        "nofile": nofile,
        "fsize_bytes": 64 * 1024 * 1024,
        "threads": threads,
        "timeout_seconds": FORGE_TIMEOUT_SECONDS,
    }


def _limit_resources(limits: Mapping[str, Any] | None = None) -> None:
    """``preexec_fn`` for the forge child: apply the resolved rlimits."""
    try:
        import resource

        if limits is None:
            limits = forge_resource_limits()
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits["fsize_bytes"],) * 2)
        _, hard_nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
        nofile = int(limits["nofile"])
        if hard_nofile != resource.RLIM_INFINITY:
            nofile = min(nofile, int(hard_nofile))
        resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
        address_space = limits["address_space_bytes"]
        if address_space is not None and sys.platform.startswith("linux"):
            resource.setrlimit(
                resource.RLIMIT_AS, (int(address_space), int(address_space)))
    except (ImportError, OSError, ValueError, RuntimeError):
        return


def _error_detail(exit_code: int | None, output: str) -> dict[str, Any]:
    """Tail of forge's combined stdout+stderr, bounded for the verdict."""
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    tail = "\n".join(lines[-_ERROR_DETAIL_LINES:])
    truncated = len(tail.encode("utf-8", errors="replace")) > _ERROR_DETAIL_BYTES
    if truncated:
        tail = tail.encode("utf-8", errors="replace")[-_ERROR_DETAIL_BYTES:].decode(
            "utf-8", errors="replace")
    return {
        "exit_code": exit_code,
        "output_tail": tail,
        "output_lines": len(lines),
        "truncated": truncated or len(lines) > _ERROR_DETAIL_LINES,
    }


def _solidity_source(trace: list[dict[str, Any]], expected: int) -> str:
    hops = []
    for row in trace:
        hops.append(
            "        amount = amountOut(amount, "
            f"{int(row['reserve_in_before'])}, "
            f"{int(row['reserve_out_before'])}, "
            f"{int(row['fee_bps'])});"
        )
    body = "\n".join(hops)
    return f"""// SPDX-License-Identifier: MIT
pragma solidity {SOLIDITY_VERSION};

contract DexRouteReplayTest {{
    function amountOut(
        uint256 amountIn, uint256 reserveIn, uint256 reserveOut, uint256 feeBps
    ) internal pure returns (uint256) {{
        uint256 amountWithFee = amountIn * (10000 - feeBps);
        return amountWithFee * reserveOut / (reserveIn * 10000 + amountWithFee);
    }}

    function testRouteReplay() public pure {{
        uint256 amount = {int(trace[0]['amount_in'])};
{body}
        require(amount == {int(expected)}, "python/revm amount mismatch");
    }}
}}
"""


def native_replay(reference: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    if not runtime.get("native"):
        return {
            "available": False, "ran": False, "matched": False,
            "gas_used": None, "error": "native Foundry/revm runtime unavailable",
        }
    trace = reference["trace"]
    if not trace:
        raise ContractError("cannot replay an empty route")
    with tempfile.TemporaryDirectory(prefix="skillhub-dex-") as raw_root:
        root = Path(raw_root)
        (root / "test").mkdir()
        (root / "src").mkdir()
        (root / "test" / "DexRouteReplay.t.sol").write_text(
            _solidity_source(trace, int(reference["final_amount_wei"])),
            encoding="utf-8",
        )
        solc = str(runtime["solc_path"]).replace("\\", "\\\\")
        (root / "foundry.toml").write_text(
            "[profile.default]\n"
            'src = "src"\n'
            'test = "test"\n'
            'out = "out"\n'
            'cache_path = "cache"\n'
            f'solc = "{solc}"\n'
            "offline = true\n"
            "optimizer = true\n"
            "optimizer_runs = 200\n"
            'evm_version = "cancun"\n'
            "ffi = false\n"
            "fs_permissions = []\n",
            encoding="utf-8",
        )
        path_entries = [
            str(Path(str(runtime["forge_path"])).parent),
            str(Path(str(runtime["solc_path"])).parent),
            os.environ.get("PATH", ""),
        ]
        limits = forge_resource_limits()
        env = {
            "PATH": os.pathsep.join(dict.fromkeys(path_entries)),
            "HOME": str(root),
            "TMPDIR": str(root),
            "LC_ALL": "C",
            "LANG": "C",
            "FOUNDRY_OFFLINE": "true",
            "FOUNDRY_DISABLE_NIGHTLY_WARNING": "1",
            "NO_COLOR": "1",
            "RAYON_NUM_THREADS": str(limits["threads"]),
        }
        try:
            completed = subprocess.run(
                [str(runtime["forge_path"]), "test", "--offline", "--force",
                 "--match-test", FORGE_TEST_PATTERN, "--color", "never"],
                cwd=root, capture_output=True, text=True,
                timeout=limits["timeout_seconds"],
                check=False, env=env, stdin=subprocess.DEVNULL,
                preexec_fn=(partial(_limit_resources, limits)
                            if os.name == "posix" else None),
            )
        except subprocess.TimeoutExpired as exc:
            partial_output = "\n".join(
                part.decode(errors="replace") if isinstance(part, bytes)
                else str(part or "")
                for part in (exc.stdout, exc.stderr))
            return {
                "available": True, "ran": True, "matched": False,
                "gas_used": None, "error": "native replay timed out",
                "exit_code": None,
                "error_detail": {
                    **_error_detail(None, partial_output),
                    "timeout_seconds": limits["timeout_seconds"],
                },
                "limits": limits,
            }
        verdict = parse_forge_replay(
            completed.returncode, completed.stdout + "\n" + completed.stderr)
        verdict["limits"] = limits
        return verdict


def parse_forge_replay(exit_code: int, output: str) -> dict[str, Any]:
    """Turn ``forge test`` output into the native_replay verdict.

    Fails closed: a run that selected no test (``Warning: no tests match``,
    exit 0, no summary line) or printed no ``N passed; M failed`` summary at
    all is a *failed* replay with an explicit reason, never a silent
    ``matched=False`` that looks like an arithmetic disagreement.
    """
    summaries = _FORGE_SUMMARY.findall(output)
    passed = sum(int(row[0]) for row in summaries)
    failed = sum(int(row[1]) for row in summaries)
    gas = _FORGE_GAS.findall(output)
    if _FORGE_NO_MATCH.search(output):
        error = "no_tests_matched"
    elif not summaries:
        error = "no_forge_summary"
    elif exit_code == 0 and passed >= 1 and failed == 0 and len(gas) == 1:
        error = None
    else:
        error = "Foundry/revm replay disagreed"
    verdict = {
        "available": True,
        "ran": True,
        "matched": error is None,
        "gas_used": int(gas[-1]) if gas else None,
        "exit_code": exit_code,
        "tests_passed": passed,
        "tests_failed": failed,
        "error": error,
    }
    if error is not None:
        # A failed replay used to leave nothing but ``matched=False`` behind,
        # so a forge that aborted on startup (rlimits, missing solc, panic)
        # was indistinguishable from an arithmetic disagreement in the image
        # build log. The tail of forge's own output travels with the verdict
        # into ``metadata`` / ``evaluation.json``.
        verdict["error_detail"] = _error_detail(exit_code, output)
    return verdict


def _stress_results(task: dict[str, Any], submission: dict[str, Any]
                    ) -> list[dict[str, Any]]:
    public = task["public"]
    rows = []
    for scenario in task["hidden"]["stress_snapshots"]:
        result = route_net_profit(
            scenario["pools"], submission["pools"],
            submission["amount_in_wei"], base_asset=public["base_asset"],
            gas_model=public["gas_model"], fee_model=scenario["fee_model"],
        )
        rows.append({
            "id": scenario["id"],
            "positive": result["net_profit_wei"] > 0,
            "net_profit_wei": result["net_profit_wei"],
        })
    return rows


class DexArbitrageBackend:
    def __init__(self, task: dict[str, Any]):
        self.task = task
        self.runtime = native_runtime_info()

    def evaluate(self, raw: Any) -> dict[str, Any]:
        submission = validate_submission(self.task["public"], raw)
        public = self.task["public"]
        reference = route_net_profit(
            public["snapshot"]["pools"], submission["pools"],
            submission["amount_in_wei"], base_asset=public["base_asset"],
            gas_model=public["gas_model"], fee_model=public["fee_model"],
        )
        if reference["final_asset"] != public["base_asset"].lower():
            raise ContractError("route must finish in the published base token")
        stresses = _stress_results(self.task, submission)
        oracle_profit = int(self.task["hidden"]["oracle_net_profit_wei"])
        net_profit = int(reference["net_profit_wei"])
        profit_quality = max(0.0, min(1.0, net_profit / oracle_profit))
        stress_robustness = sum(
            1.0 for row in stresses if row["positive"]) / len(stresses)
        native = native_replay(reference, self.runtime)
        gates = {
            "route_valid": True,
            "amount_bounded": True,
            "execution_success": True,
            "accounting_conserved": reference["accounting_conserved"] is True,
            "positive_net_profit": net_profit > 0,
            "stress_positive": all(row["positive"] for row in stresses),
            "native_replay": native.get("matched") is True,
        }
        quality = {
            "route_validity": 1.0,
            "execution_success": 1.0,
            "profit_quality": profit_quality,
            "stress_robustness": stress_robustness,
        }
        result = {
            "candidate_digest": sha256_digest(submission),
            "submission": submission,
            "source_safe": True,
            "gates": gates,
            "quality": quality,
            "correctness_unlocked": all(gates[name] for name in (
                "route_valid", "amount_bounded", "execution_success",
                "accounting_conserved", "positive_net_profit",
                "stress_positive",
            )),
            "verified_protocol": True,
            "evidence_bound": True,
            "measurements": {
                "amount_in_wei": submission["amount_in_wei"],
                "final_amount_wei": reference["final_amount_wei"],
                "gross_profit_wei": reference["gross_profit_wei"],
                "net_profit_wei": net_profit,
                "execution_fee_wei": reference["execution_fee_wei"],
                "gas_used": reference["gas_used"],
                "hops": len(submission["pools"]),
                "profit_regret_wei": max(0, oracle_profit - net_profit),
            },
            "stress": stresses,
            "trace": reference["trace"],
            "native_replay": native,
            "runtime": {
                "mode": self.runtime["mode"],
                "native": self.runtime["native"],
                "network_required_at_runtime": False,
            },
        }
        result["promotion_eligible"] = bool(
            result["correctness_unlocked"] and native.get("matched") is True)
        result["evaluation_digest"] = sha256_digest({
            "task_digest": self.task["digest"],
            "candidate_digest": result["candidate_digest"],
            "gates": gates,
            "quality": quality,
            "measurements": result["measurements"],
            "stress": stresses,
            # Diagnostics (forge's output tail carries wall-clock timings,
            # the rlimits are per-host) are evidence, not verdict.
            "native_replay": {
                key: value for key, value in native.items()
                if key not in _DIAGNOSTIC_KEYS
            },
        })
        return result

    def artifacts(self, evaluation: dict[str, Any] | None
                  ) -> dict[str, dict[str, Any]]:
        if not evaluation:
            return {}
        snapshot = json.dumps(
            self.task["public"]["snapshot"], indent=2, sort_keys=True)
        result = json.dumps({
            key: deepcopy(evaluation[key]) for key in (
                "candidate_digest", "submission", "gates", "quality",
                "measurements", "stress", "native_replay", "runtime",
                "evaluation_digest",
            )
        }, indent=2, sort_keys=True)
        return {
            "snapshot.json": {
                "data": snapshot.encode("utf-8"),
                "media_type": "application/json",
            },
            "evaluation.json": {
                "data": result.encode("utf-8"),
                "media_type": "application/json",
            },
        }


def public_evaluation(evaluation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not evaluation:
        return None
    measurements = {
        key: value for key, value in evaluation["measurements"].items()
        if key != "profit_regret_wei"
    }
    native_replay = {
        key: value for key, value in evaluation["native_replay"].items()
        if key not in _DIAGNOSTIC_KEYS
    }
    return {
        "candidate_digest": evaluation["candidate_digest"],
        "gates": deepcopy(evaluation["gates"]),
        "quality": deepcopy(evaluation["quality"]),
        "measurements": measurements,
        "stress": deepcopy(evaluation["stress"]),
        # Host diagnostics stay out of the policy-visible view (the solidity
        # backend strips the same keys); they survive in evaluation.json.
        "native_replay": native_replay,
        "runtime": deepcopy(evaluation["runtime"]),
    }
