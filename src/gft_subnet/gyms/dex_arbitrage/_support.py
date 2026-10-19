"""Helpers the DEX test gym needs, copied verbatim from GFT Studio.

Only what ``gyms/dex_arbitrage`` uses: hashing and errors, reward
normalizers and aggregation, key redaction, scoring and the typed simulator.
Keep in sync with GFT Studio when the gym changes there.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any
import hashlib
import json
import math
import re
import time


# --- from GFT Studio hub/models.py

class HubError(Exception):
    """Typed control-plane failure with a stable machine code."""

    code = "HUB_ERROR"
    http_status = 400
    retryable = False
    failure_class = "request"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class ContractError(HubError):
    code = "CONTRACT_INVALID"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")


def sha256_digest(value: Any) -> str:
    raw = value if isinstance(value, (bytes, bytearray)) else canonical_json(value)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def require_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{name} must be finite")
    return number



# --- from GFT Studio hub/rewards/normalizers.py

def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def higher_is_better(value: float, low: float, high: float) -> float:
    if high == low:
        raise ContractError("higher_is_better range cannot be empty")
    return clip01((value - low) / (high - low))


def lower_is_better(value: float, low: float, high: float) -> float:
    if high == low:
        raise ContractError("lower_is_better range cannot be empty")
    return clip01((high - value) / (high - low))


def target_tolerance(value: float, target: float, tolerance: float) -> float:
    if tolerance <= 0:
        raise ContractError("tolerance must be positive")
    return clip01(1.0 - abs(value - target) / tolerance)


def cost_or_latency(value: float, scale: float) -> float:
    if scale <= 0:
        raise ContractError("cost/latency scale must be positive")
    return math.exp(-max(value, 0.0) / scale)


def budget_score(value: float, soft: float, hard: float) -> float:
    if hard <= soft:
        raise ContractError("hard budget must exceed soft budget")
    if value <= soft:
        return 1.0
    if value >= hard:
        return 0.0
    return (hard - value) / (hard - soft)


def apply_normalizer(kind: str, value: float, params: dict) -> float:
    value = require_finite("component", value)
    if kind == "higher_is_better":
        return higher_is_better(value, params["low"], params["high"])
    if kind == "lower_is_better":
        return lower_is_better(value, params["low"], params["high"])
    if kind == "target_tolerance":
        return target_tolerance(value, params["target"], params["tolerance"])
    if kind == "cost_or_latency":
        return cost_or_latency(value, params["scale"])
    if kind == "unit_interval":
        return clip01(value)
    if kind == "identity":
        return value
    raise ContractError(f"unknown normalizer {kind!r}")



# --- from GFT Studio hub/rewards/runtime.py

def aggregate_reward(components: dict[str, float], profile: dict[str, Any],
                     *, gates: dict[str, bool], penalties: dict[str, float] | None = None
                     ) -> dict[str, Any]:
    """Apply frozen gates, weights, penalties, and clip to one native scalar."""
    declared = profile["components"]
    normalized = {}
    weighted = []
    weight_sum = 0.0
    for name, spec in declared.items():
        if name not in components:
            raise ContractError(f"missing reward component {name}")
        raw = require_finite(name, components[name])
        low, high = spec.get("range", [0.0, 1.0])
        if raw < low or raw > high:
            raise ContractError(f"{name}={raw} outside [{low}, {high}]")
        if spec.get("weight", 0) > 0 and "normalizer" not in spec:
            raise ContractError(f"positive-weight component {name} needs a frozen normalizer")
        normalizer = spec.get("normalizer") or {"kind": "unit_interval"}
        value = apply_normalizer(normalizer["kind"], raw, normalizer)
        if not math.isfinite(value):
            raise ContractError(f"{name} normalized to a non-finite value")
        normalized[name] = value
        weight = float(spec.get("weight", 0.0))
        if weight < 0:
            raise ContractError(f"{name} weight cannot be negative")
        if weight > 0:
            weighted.append(weight * value)
            weight_sum += weight
    if weight_sum <= 0:
        raise ContractError("reward profile has no positive weights")
    gate = 1.0
    failed_gates = []
    for name, passed in gates.items():
        if not passed:
            gate = 0.0
            failed_gates.append(name)
    base = sum(weighted) / weight_sum
    penalty_total = sum(float(value) for value in (penalties or {}).values())
    minimum = float(profile.get("minimum", 0.0))
    maximum = float(profile.get("maximum", 1.0))
    native = max(minimum, min(maximum, gate * base - penalty_total))
    return {
        "native": native,
        "gate": gate,
        "failed_gates": failed_gates,
        "base": base,
        "normalized": normalized,
        "raw": {name: require_finite(name, value) for name, value in components.items()},
        "penalties": dict(penalties or {}),
        "profile_digest": profile.get("digest") or sha256_digest(profile),
        "range": [minimum, maximum],
    }



# --- from GFT Studio hub/security.py

# Whole-word tokens only. A key is hidden when one of its `_`/camelCase
# tokens equals one of these (singular or plural), never by substring, so
# ``expansion_angle_deg`` (pan), ``occupancy_total`` (pan) and
# ``progress_label`` stay visible while ``hidden_expected`` and
# ``api_tokens`` are removed.
HIDDEN_KEY_TOKENS = frozenset({
    "hidden", "evidence", "expected", "secret", "password", "token",
    "grader", "ssn", "pan",
})


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


_KEY_TOKEN_RE = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")


def key_tokens(key: Any) -> frozenset[str]:
    """Split ``snake_case``/``camelCase``/``kebab-case`` keys into lowercase words."""
    return frozenset(
        token.lower() for token in _KEY_TOKEN_RE.findall(str(key))
    )


def is_hidden_key(
    key: Any,
    extra_keys: frozenset[str] | set[str] = frozenset(),
    *,
    tokens: frozenset[str] = HIDDEN_KEY_TOKENS,
) -> bool:
    """True when the key names hidden material (exact token match)."""
    if str(key) in extra_keys:
        return True
    for token in key_tokens(key):
        if token in tokens:
            return True
        if token.endswith("s") and token[:-1] in tokens:
            return True
    return False


def redact(value: Any, *, extra_keys: frozenset[str] | set[str] = frozenset()) -> Any:
    """Drop hidden keys (exact token match, plus ``extra_keys``) and mask emails."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if is_hidden_key(key, extra_keys):
                continue
            cleaned[str(key)] = redact(item, extra_keys=extra_keys)
        return cleaned
    if isinstance(value, list):
        return [redact(item, extra_keys=extra_keys) for item in value]
    if isinstance(value, str):
        return EMAIL_RE.sub("[redacted-email]", value)
    return value



# --- from GFT Studio hub/verticals/common.py

def public_family(task: dict[str, Any]) -> str:
    """Family label the *policy* may see.

    Verticals whose family names encode the answer ("single_phase_clean",
    "cubic_basic", "strain_dominated") set ``task["public_family"]`` to an
    opaque label; the real family stays available server-side for
    contamination checks and reporting.
    """
    return str(task.get("public_family") or task["family"])


def public_task_id(task: dict[str, Any]) -> str:
    """Episode-visible task id; opaque when the real id embeds the family."""
    return str(task.get("public_task_id") or task["id"])


def public_task_view(task: dict[str, Any]) -> dict[str, Any]:
    view = deepcopy(task["public"])
    view["task_id"] = public_task_id(task)
    view["family"] = public_family(task)
    view["vertical"] = task.get("vertical") or view.get("vertical") or "shopify"
    view["split_declared"] = "episode"
    return view


def make_reward_profile(profile_id: str, components: dict[str, float],
                        gates: list[str], grader: str) -> dict[str, Any]:
    profile = {
        "id": profile_id,
        "components": {
            name: {
                "unit": "fraction",
                "direction": "higher",
                "range": [0.0, 1.0],
                "weight": weight,
                "normalizer": {"kind": "unit_interval"},
            }
            for name, weight in components.items()
        },
        "gates": gates,
        "minimum": 0.0,
        "maximum": 1.0,
    }
    profile["digest"] = sha256_digest({
        "components": profile["components"],
        "gates": profile["gates"],
        "minimum": 0.0,
        "maximum": 1.0,
    })
    profile["grader_digest"] = sha256_digest(grader)
    return profile


def profile_with_digest(template: dict[str, Any]) -> dict[str, Any]:
    profile = dict(template)
    profile["digest"] = sha256_digest({
        "components": profile["components"],
        "gates": profile["gates"],
        "minimum": profile["minimum"],
        "maximum": profile["maximum"],
    })
    return profile


def efficiency_score(calls: int, cost_micros: int, budgets: dict[str, Any]) -> float:
    """Calls and metered cost only.

    Episode wall-clock latency depends on host load and on the policy's think
    time between steps, so the same trajectory would score differently from
    run to run.  It stays in step metadata as telemetry, never in the reward.
    """
    return (
        2 / 3 * budget_score(calls, budgets["calls_soft"], budgets["calls_hard"])
        + 1 / 3 * budget_score(
            cost_micros, budgets["cost_soft_micros"], budgets["cost_hard_micros"])
    )


def build_terminal_evidence(
        task: dict[str, Any], receipts: list[dict[str, Any]],
        artifacts: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Bind a terminal score to concrete artifact bytes and action receipts."""
    manifest = {}
    for name, payload in sorted(artifacts.items()):
        data = payload.get("data")
        if not isinstance(data, (bytes, bytearray)) or not data:
            return None
        manifest[name] = {
            "digest": sha256_digest(data),
            "media_type": str(payload.get("media_type") or "application/octet-stream"),
            "size": len(data),
        }
    if not manifest:
        return None
    core = {
        "evidence_id": task["hidden"].get("evidence_id"),
        "artifact_manifest_digest": sha256_digest(manifest),
        "receipt_digest": sha256_digest(receipts),
    }
    return {**core, "binding_digest": sha256_digest(core)}


def evidence_is_bound(task: dict[str, Any], evidence: dict[str, Any] | None) -> bool:
    if not evidence:
        return False
    core = {
        "evidence_id": evidence.get("evidence_id"),
        "artifact_manifest_digest": evidence.get("artifact_manifest_digest"),
        "receipt_digest": evidence.get("receipt_digest"),
    }
    return bool(
        core["evidence_id"] == task["hidden"].get("evidence_id")
        and core["artifact_manifest_digest"]
        and core["receipt_digest"]
        and evidence.get("binding_digest") == sha256_digest(core)
    )


#: Ceiling for the reward of an episode that did not fully solve the task.
#: A solved episode always scores at least this much, so "any success beats
#: any failure" holds regardless of how the quality components come out.
PARTIAL_CREDIT_CAP = 0.25


def finish_score(components: dict[str, float], gates: dict[str, bool],
                 profile: dict[str, Any], *, task_success: float,
                 evidence_id: str,
                 evidence: dict[str, Any] | None,
                 effective_work: bool,
                 zero_reward_gates: set[str],
                 promotion_only_gates: set[str] | None = None,
                 partial_credit_cap: float = PARTIAL_CREDIT_CAP) -> dict[str, Any]:
    """Return the training reward plus a strict, separate promotion gate.

    Reward contract (fail-closed):

    * Any failed gate in ``zero_reward_gates`` (or no effective work) → 0.
    * Any other failed gate, or ``task_success < 1`` → the episode is *not
      complete*.  It earns at most ``partial_credit_cap * progress`` — pure
      progress shaping.  Quality, safety and efficiency components do **not**
      pay for an unsolved task, so "do nothing safely" or "resubmit the
      baseline" cannot out-score a real attempt.
    * Complete episode → ``cap + (1 - cap) * base`` where ``base`` is the
      weighted quality score.  This is always ≥ the best possible incomplete
      reward.
    * ``promotion_only_gates`` (e.g. an optional native sign-off replay when
      the reference implementation *is* the ground truth) never change the
      training reward; they only decide ``hard_gated_success``.  A vertical
      may put a gate here only when the reference result is exact without it.

    ``hard_gated_success`` / promotion still require every gate to pass.
    """
    # clip01 alone is not a NaN guard: max(0.0, min(1.0, nan)) is 1.0 in
    # Python, so a non-finite success signal would score as *perfect*.
    # Fail closed on the contract instead.
    cap = clip01(require_finite("partial_credit_cap", partial_credit_cap))
    progress = clip01(require_finite("task_success", task_success))
    checked_gates = dict(gates)
    checked_gates["effective_work"] = effective_work
    checked_gates["task_complete"] = progress + 1e-9 >= 1.0
    for name in ("evidence_bound", "evidence_present"):
        if name in checked_gates:
            checked_gates[name] = bool(checked_gates[name]) and evidence_is_bound(
                {"hidden": {"evidence_id": evidence_id}}, evidence)
    unknown_gates = set(zero_reward_gates) - set(checked_gates)
    if unknown_gates:
        # A typo in a vertical's ZERO_REWARD_GATES used to be skipped
        # silently, which turns a fail-closed safety gate into a no-op.
        raise ContractError(
            "unknown zero-reward gates not among the checked gates "
            f"{sorted(unknown_gates)}; available: {sorted(checked_gates)}")
    reward_gates = {
        name: checked_gates[name]
        for name in zero_reward_gates
    }
    reward_gates["effective_work"] = effective_work
    scored = aggregate_reward(components, profile, gates=reward_gates)
    reward_gate = scored["gate"]
    penalty_total = sum(float(value) for value in scored["penalties"].values())
    minimum = float(profile.get("minimum", 0.0))
    maximum = float(profile.get("maximum", 1.0))
    failed = [name for name, passed in checked_gates.items() if not passed]
    promotion_only = set(promotion_only_gates or ())
    complete = not any(name not in promotion_only for name in failed)
    if complete:
        earned = cap + (1.0 - cap) * scored["base"]
    else:
        earned = cap * progress
    scored["native"] = max(
        minimum, min(maximum, reward_gate * earned - penalty_total))
    scored["failed_gates"] = failed
    # Unmet gates that block only promotion (e.g. an operator hardware
    # receipt): a training-complete gold run may still miss these.
    scored["pending_promotion_gates"] = [
        name for name in failed if name in promotion_only]
    scored["gate"] = 0.0 if failed else 1.0
    scored["reward_gate"] = reward_gate
    scored["task_progress"] = progress
    scored["complete"] = complete
    scored["partial_credit_cap"] = cap
    scored["hard_gated_success"] = not failed
    scored["evidence_digest"] = evidence.get("binding_digest") if evidence else None
    scored["profile_digest"] = profile.get("digest") or sha256_digest(profile)
    scored["grader_digest"] = profile.get("grader_digest")
    return scored


def default_budgets() -> dict[str, int]:
    return {
        "calls_soft": 10,
        "calls_hard": 20,
        "latency_soft_ms": 8_000,
        "latency_hard_ms": 60_000,
        "cost_soft_micros": 80_000,
        "cost_hard_micros": 250_000,
    }


class TypedSimulator:
    """Shared OpenEnv loop. Verticals implement handle() and score()."""

    score_each_step = False
    max_calls: int | None = None

    def __init__(self, task: dict[str, Any], reward_profile: dict[str, Any]):
        self.task = task
        self.reward_profile = reward_profile
        self.state = deepcopy(task["snapshot"])
        self.initial = deepcopy(task["snapshot"])
        self.receipts: list[dict[str, Any]] = []
        self.calls = 0
        self.started = time.perf_counter()
        self.done = False
        #: True when the episode ended because a budget ran out (calls,
        #: simulated time, physics steps), not because the task finished or
        #: the policy misbehaved.  Mirrors Gymnasium's ``truncated`` flag.
        self.truncated = False
        self.cost_micros = 0
        self._keys: set[str] = set()
        self.last_observation: dict[str, Any] = {}
        self.last_reward = 0.0
        self.last_components: dict[str, Any] = {}

    def task_view(self) -> dict[str, Any]:
        """Policy-visible task body carried by every observation.

        Defaults to the full ``public`` block.  Verticals whose public data
        holds large arrays override this with a compact rendering (summary,
        preview, data handles) and expose the exact arrays through read-only
        inspect actions instead.
        """
        return public_task_view(self.task)

    def reset_observation(self) -> dict[str, Any]:
        observation = self.task_view()
        observation["ok"] = True
        observation["done"] = False
        observation["result"] = self.public_summary()
        self.last_observation = observation
        return redact(observation)

    def step(self, action: dict[str, Any], *, cost_micros: int = 0) -> dict[str, Any]:
        if self.done:
            raise ContractError("episode is already terminal")
        if not isinstance(action, dict) or "type" not in action:
            raise ContractError("action must be a JSON object with type")
        self.calls += 1
        self.cost_micros += cost_micros
        key = action.get("idempotency_key") or ""
        if key and key in self._keys:
            result, error = {"replayed": True}, None
        else:
            try:
                result, error = self.handle(action)
            except Exception:
                # A rejected action must not change replayable episode state;
                # the gateway only journals successful steps.
                self.calls -= 1
                self.cost_micros -= cost_micros
                raise
            if key and error is None:
                self._keys.add(key)
        if self.max_calls is not None and self.calls >= self.max_calls and not self.done:
            self.done = True
            self.truncated = True
        observation = self.task_view()
        observation["ok"] = error is None
        observation["error"] = error
        observation["result"] = redact(result)
        observation["done"] = self.done
        observation["truncated"] = self.truncated
        reward = 0.0
        components: dict[str, Any] = {}
        if self.done or self.score_each_step:
            components = self.score()
            reward = components["native"]
        if self.done:
            observation["completion"] = {
                "hard_gated_success": components["hard_gated_success"],
                "promotion_eligible": components.get(
                    "promotion_eligible", components["hard_gated_success"]
                ),
                "training_reward_only": components.get(
                    "training_reward_only", False
                ),
                "evidence_ref": components["evidence_digest"],
            }
        self.last_observation = observation
        self.last_reward = reward
        self.last_components = components
        return {
            "observation": observation,
            "reward": reward,
            "done": self.done,
            "metadata": {
                "base_reward": components.get("base", 0.0),
                "task_progress": components.get("task_progress", 0.0),
                "reward_gate": components.get("reward_gate", 0.0),
                "reward_components": {
                    name: value
                    for name, value in (components.get("normalized") or {}).items()
                },
                "raw": components.get("raw") or {},
                "failed_gates": components.get("failed_gates") or [],
                "pending_promotion_gates": components.get("pending_promotion_gates") or [],
                "calls": self.calls,
                "latency_ms": self.latency_ms,
                "cost_micros": self.cost_micros,
                "evidence_digest": components.get("evidence_digest"),
                "profile_digest": components.get("profile_digest"),
                "grader_digest": components.get("grader_digest"),
                "verification_level": components.get("verification_level"),
                "promotion_eligible": components.get(
                    "promotion_eligible", components.get("hard_gated_success", False)
                ),
                "training_reward_only": components.get("training_reward_only", False),
                "truncated": self.truncated,
                # Vertical-specific, policy-invisible metadata (namespaced).
                **self.step_metadata(),
            },
        }

    @property
    def latency_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000

    def public_summary(self) -> dict[str, Any]:
        return {"resources": sorted(self.state.keys())}

    def step_metadata(self) -> dict[str, Any]:
        """Extra step metadata under a vertical namespace.

        Metadata never reaches the policy (only observation, reward and done
        do), so hidden scoring artifacts may live here; never in
        :meth:`task_view` or the step result.
        """
        return {}

    def public_state(self) -> dict[str, Any]:
        visible = deepcopy(self.state)
        visible.pop("hidden", None)
        visible.pop("event_log", None)
        return redact(visible)

    def handle(self, action: dict[str, Any]) -> tuple[Any, str | None]:
        raise NotImplementedError

    def score(self) -> dict[str, Any]:
        raise NotImplementedError

    def artifact_payloads(self) -> dict[str, dict[str, Any]]:
        """Terminal binary artifacts keyed by safe file name."""
        return {
            "terminal_state.json": {
                "media_type": "application/json",
                "data": json.dumps(
                    self.public_state(), sort_keys=True, separators=(",", ":")
                ).encode("utf-8"),
            },
        }

    def terminal_evidence(self) -> dict[str, Any] | None:
        return build_terminal_evidence(
            self.task, self.receipts, self.artifact_payloads())

    def close(self) -> None:
        """Release episode-local resources."""

    def mark_unsafe(self, operation: str, reason: str) -> tuple[Any, str]:
        """End the episode for misconduct: a safety bypass or unsafe command.

        Scorers treat the ``rejected_unsafe`` receipt as a zero-reward gate.
        Do not use this for budget exhaustion or schema mistakes; those are
        :meth:`truncate` and a non-terminal ``(result, error)`` respectively.
        """
        self.receipts.append({
            "operation": operation, "rejected_unsafe": True, "effective": False,
        })
        self.done = True
        return {}, reason

    def truncate(self, operation: str, reason: str) -> tuple[Any, str]:
        """End the episode because a budget ran out, not for misconduct.

        The protocol stays valid, so verified progress keeps its capped
        partial reward; full reward still needs the explicit closing step.
        """
        self.receipts.append({
            "operation": operation, "truncated": True, "effective": False,
        })
        self.done = True
        self.truncated = True
        return {"truncated": True}, reason
