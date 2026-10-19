"""DEX arbitrage test gym: reference grading, rewards, safety (ported from GFT Studio)."""
from __future__ import annotations

import json
import re
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest

from gft_subnet.gyms.dex_arbitrage._support import ContractError
from gft_subnet.gyms.dex_arbitrage import backend as backend_module
from gft_subnet.gyms.dex_arbitrage.backend import (
    DexArbitrageBackend,
    public_evaluation,
)
from gft_subnet.gyms.dex_arbitrage.environment import (
    DexArbitrageGym,
    solve_from_observation,
    solve_task,
)
from gft_subnet.gyms.dex_arbitrage.math import amount_out, route_net_profit
from gft_subnet.gyms.dex_arbitrage.safety import (
    UnsafeDexSubmission,
    validate_submission,
)
from gft_subnet.gyms.dex_arbitrage.tasks import (
    FAMILIES,
    build_family_task,
)


@pytest.fixture(autouse=True)


def reference_runtime(monkeypatch):
    monkeypatch.setenv("SKILLHUB_DEX_ARBITRAGE_MODE", "reference")
    backend_module.native_runtime_info.cache_clear()
    yield
    backend_module.native_runtime_info.cache_clear()


def _gold(task):
    return {
        "pools": deepcopy(task["hidden"]["oracle_route"]),
        "amount_in_wei": str(task["hidden"]["oracle_amount_in_wei"]),
    }


@pytest.mark.parametrize("family", FAMILIES)


def test_gold_route_is_profitable_and_passes_reference_gates(family):
    task = build_family_task(family, 7, "benchmark")
    evaluation = DexArbitrageBackend(task).evaluate(_gold(task))

    assert evaluation["correctness_unlocked"] is True
    assert evaluation["measurements"]["net_profit_wei"] > 0
    assert evaluation["quality"]["profit_quality"] == pytest.approx(1.0)
    assert all(
        evaluation["gates"][gate]
        for gate in evaluation["gates"] if gate != "native_replay"
    )
    assert evaluation["gates"]["native_replay"] is False
    assert evaluation["promotion_eligible"] is False


@pytest.mark.parametrize("family", FAMILIES)


def test_gold_episode_gets_full_training_reward_but_not_promotion(family):
    task = build_family_task(family, 11, "train")
    gym = DexArbitrageGym(task)
    gym.reset_observation()
    for action in solve_task(task):
        result = gym.step(action)
        assert result["observation"]["error"] is None

    assert result["done"] is True
    assert result["reward"] == pytest.approx(1.0)
    assert result["metadata"]["verification_level"] == \
        "reference_exact_integer_amm"
    assert result["metadata"]["promotion_eligible"] is False
    assert result["metadata"]["training_reward_only"] is True
    assert result["metadata"]["failed_gates"] == ["native_replay"]


def test_foundry_replay_is_a_separate_promotion_gate(monkeypatch):
    monkeypatch.setattr(backend_module, "native_replay", lambda *_: {
        "available": True,
        "ran": True,
        "matched": True,
        "gas_used": 12345,
        "error": None,
    })
    task = build_family_task("triangular_cycle", 3, "qualification")
    gym = DexArbitrageGym(task)
    gym.reset_observation()
    for action in solve_task(task):
        result = gym.step(action)

    assert result["observation"]["completion"]["hard_gated_success"] is True
    assert result["metadata"]["promotion_eligible"] is True
    assert result["metadata"]["verification_level"] == \
        "foundry_revm_arithmetic_replay"


@pytest.mark.parametrize("output,code,matched", [
    ("[PASS] testRouteReplay() (gas: 234)\nSuite result: ok. 1 passed; 0 failed", 0, True),
    ("Compiler run successful!\nWarning: no tests match the provided pattern", 0, False),
    ("[PASS] testSomethingElse() (gas: 234)\nSuite result: ok. 1 passed; 0 failed", 0, False),
    ("[FAIL] testRouteReplay() (gas: 234)\nSuite result: FAILED. 0 passed; 1 failed", 1, False),
    ("[PASS] testRouteReplay() (gas: 234)\nSuite result: ok. 1 passed; 0 failed", 1, False),
])


def test_native_replay_requires_the_named_test(monkeypatch, output, code, matched):
    task = build_family_task("triangular_cycle", 17, "benchmark")
    candidate = _gold(task)
    reference = route_net_profit(
        task["public"]["snapshot"]["pools"], candidate["pools"],
        int(candidate["amount_in_wei"]), base_asset=task["public"]["base_asset"],
        gas_model=task["public"]["gas_model"], fee_model=task["public"]["fee_model"],
    )

    def fake_run(args, **kwargs):
        selector = args[args.index("--match-test") + 1]
        assert re.fullmatch(selector, "testRouteReplay()")
        assert re.fullmatch(selector, "testRouteReplay")
        assert not re.fullmatch(selector, "testRouteReplayOther()")
        return subprocess.CompletedProcess(args, code, output, "")

    monkeypatch.setattr(backend_module.subprocess, "run", fake_run)
    result = backend_module.native_replay(reference, {
        "native": True, "forge_path": "/test/forge", "solc_path": "/test/solc",
    })
    assert result["matched"] is matched
    assert result["ran"] is True


_FORGE_PASS_NEW = """Compiling 1 files with Solc 0.8.26
Compiler run successful!

Ran 1 test for test/DexRouteReplay.t.sol:DexRouteReplayTest
[PASS] testRouteReplay() (gas: 1444)
Suite result: ok. 1 passed; 0 failed; 0 skipped; finished in 33.22ms (9.50ms CPU time)

Ran 1 test suite in 105.95ms (33.22ms CPU time): 1 tests passed, 0 failed, 0 skipped (1 total tests)
"""


_FORGE_PASS_OLD = """Compiling 1 files with 0.8.26
Compiler run successful!

Running 1 test for test/DexRouteReplay.t.sol:DexRouteReplayTest
[PASS] testRouteReplay() (gas: 1444)
Test result: ok. 1 passed; 0 failed; 0 skipped; finished in 1.02ms

Ran 1 test suites: 1 tests passed, 0 failed, 0 skipped (1 total tests)
"""


_FORGE_FAIL = """Ran 1 test for test/DexRouteReplay.t.sol:DexRouteReplayTest
[FAIL: revert: python/revm amount mismatch] testRouteReplay() (gas: 1500)
Suite result: FAILED. 0 passed; 1 failed; 0 skipped; finished in 1.00ms
"""


_FORGE_NO_MATCH = """Compiling 1 files with Solc 0.8.26
Compiler run successful!
Warning: no tests match the provided pattern:
\tmatch-test: `^testRouteReplay$`

Did you mean `testRouteReplay`?
"""


def test_forge_match_test_pattern_matches_signature_and_bare_name():
    # forge >= 1.x filters on the ABI signature ``testRouteReplay()``; older
    # matchers used the bare name. The pattern must select exactly this test
    # under both conventions and nothing that merely shares the prefix.
    pattern = re.compile(backend_module.FORGE_TEST_PATTERN)
    assert pattern.search("testRouteReplay()")
    assert pattern.search("testRouteReplay")
    assert not pattern.search("testRouteReplayFuzz()")
    assert not pattern.search("testRouteReplay(uint256)")


@pytest.mark.parametrize("output", [_FORGE_PASS_NEW, _FORGE_PASS_OLD])


def test_parse_forge_replay_accepts_old_and_new_pass_output(output):
    verdict = backend_module.parse_forge_replay(0, output)
    assert verdict["matched"] is True
    assert verdict["error"] is None
    assert verdict["gas_used"] == 1444
    assert (verdict["tests_passed"], verdict["tests_failed"]) == (1, 0)


def test_parse_forge_replay_fails_closed_when_no_tests_match():
    # forge exits 0 and prints no summary when the filter selects nothing:
    # that must be an explicit failure, not a silent "disagreed".
    verdict = backend_module.parse_forge_replay(0, _FORGE_NO_MATCH)
    assert verdict["matched"] is False
    assert verdict["error"] == "no_tests_matched"
    assert (verdict["tests_passed"], verdict["tests_failed"]) == (0, 0)

    verdict = backend_module.parse_forge_replay(0, "Compiler run successful!")
    assert verdict["matched"] is False
    assert verdict["error"] == "no_forge_summary"


def test_parse_forge_replay_reports_arithmetic_disagreement():
    verdict = backend_module.parse_forge_replay(1, _FORGE_FAIL)
    assert verdict["matched"] is False
    assert verdict["error"] == "Foundry/revm replay disagreed"
    assert verdict["tests_failed"] == 1
    assert verdict["gas_used"] is None


def test_native_replay_gate_fails_when_forge_selects_no_tests(monkeypatch):
    monkeypatch.setattr(
        backend_module, "native_replay",
        lambda *_: backend_module.parse_forge_replay(0, _FORGE_NO_MATCH))
    task = build_family_task("triangular_cycle", 3, "qualification")
    evaluation = DexArbitrageBackend(task).evaluate(_gold(task))
    assert evaluation["gates"]["native_replay"] is False
    assert evaluation["promotion_eligible"] is False
    assert evaluation["native_replay"]["error"] == "no_tests_matched"


_FORGE_ABORT = """Compiling 1 files with Solc 0.8.26
memory allocation of 7 bytes failed
memory allocation of 7 bytes failed
"""


def test_failed_replay_carries_forge_output_tail_and_exit_code():
    # Run 1788386531: forge aborted under RLIMIT_AS before compiling and the
    # smoke only printed ``AssertionError``. The verdict now carries the
    # exit code and the tail of forge's own output.
    verdict = backend_module.parse_forge_replay(134, _FORGE_ABORT)
    assert verdict["matched"] is False
    assert verdict["error"] == "no_forge_summary"
    detail = verdict["error_detail"]
    assert detail["exit_code"] == 134
    assert "memory allocation of 7 bytes failed" in detail["output_tail"]
    assert detail["output_lines"] == 3
    assert detail["truncated"] is False
    # A passing replay stays lean.
    assert "error_detail" not in backend_module.parse_forge_replay(
        0, _FORGE_PASS_NEW)


def test_error_detail_is_bounded_to_the_last_lines_and_bytes():
    noise = "\n".join(f"line {index} " + "x" * 200 for index in range(400))
    detail = backend_module._error_detail(1, noise + "\nlast line")
    assert detail["truncated"] is True
    assert detail["output_lines"] == 401
    assert detail["output_tail"].endswith("last line")
    assert len(detail["output_tail"].encode()) <= backend_module._ERROR_DETAIL_BYTES
    assert "line 0 " not in detail["output_tail"]


def test_forge_resource_limits_default_to_no_address_space_cap():
    limits = backend_module.forge_resource_limits({})
    assert limits["address_space_bytes"] is None
    assert limits["address_space_gib"] is None
    assert limits["nofile"] == 1024
    assert limits["threads"] == 2
    assert limits["timeout_seconds"] == 60
    for value in ("0", "unlimited", "none", "off", " "):
        assert backend_module.forge_resource_limits(
            {backend_module.FORGE_RLIMIT_AS_ENV: value}
        )["address_space_bytes"] is None


def test_forge_resource_limits_honour_env_knobs():
    limits = backend_module.forge_resource_limits({
        backend_module.FORGE_RLIMIT_AS_ENV: "8",
        backend_module.FORGE_RLIMIT_NOFILE_ENV: "4096",
        backend_module.FORGE_THREADS_ENV: "1",
    })
    assert limits["address_space_bytes"] == 8 * 1024 ** 3
    assert limits["address_space_gib"] == 8.0
    assert limits["nofile"] == 4096
    assert limits["threads"] == 1
    assert backend_module.forge_resource_limits(
        {backend_module.FORGE_RLIMIT_AS_ENV: "1.5"}
    )["address_space_bytes"] == int(1.5 * 1024 ** 3)


@pytest.mark.parametrize("env, message", [
    ({backend_module.FORGE_RLIMIT_AS_ENV: "two"}, "number of GiB"),
    ({backend_module.FORGE_RLIMIT_AS_ENV: "-1"}, "positive"),
    ({backend_module.FORGE_RLIMIT_NOFILE_ENV: "128"}, ">= 1024"),
    ({backend_module.FORGE_THREADS_ENV: "0"}, ">= 1"),
    ({backend_module.FORGE_THREADS_ENV: "many"}, "integer"),
])


def test_forge_resource_limits_reject_bad_values(env, message):
    with pytest.raises(RuntimeError, match=message):
        backend_module.forge_resource_limits(env)


def test_native_replay_runs_forge_without_an_address_space_cap(monkeypatch, tmp_path):
    # The forge child must see RAYON_NUM_THREADS and the resolved limits;
    # the preexec hook is what applied the fatal 2 GiB RLIMIT_AS before.
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs["env"]
        captured["timeout"] = kwargs["timeout"]
        captured["preexec"] = kwargs["preexec_fn"]
        return SimpleNamespace(returncode=0, stdout=_FORGE_PASS_NEW, stderr="")

    monkeypatch.setattr(backend_module.subprocess, "run", fake_run)
    monkeypatch.delenv(backend_module.FORGE_RLIMIT_AS_ENV, raising=False)
    forge = tmp_path / "forge"
    solc = tmp_path / "solc"
    forge.write_text("")
    solc.write_text("")
    runtime = {"native": True, "forge_path": str(forge), "solc_path": str(solc)}
    task = build_family_task("two_pool_cycle", 1, "qualification")
    reference = route_net_profit(
        task["public"]["snapshot"]["pools"], task["hidden"]["oracle_route"],
        int(task["hidden"]["oracle_amount_in_wei"]),
        base_asset=task["public"]["base_asset"],
        gas_model=task["public"]["gas_model"],
        fee_model=task["public"]["fee_model"])

    verdict = backend_module.native_replay(reference, runtime)

    assert verdict["matched"] is True
    assert verdict["limits"]["address_space_bytes"] is None
    assert verdict["limits"]["nofile"] == 1024
    assert captured["env"]["RAYON_NUM_THREADS"] == "2"
    assert captured["timeout"] == 60
    assert captured["args"][:2] == [str(forge), "test"]
    assert "--match-test" in captured["args"]
    preexec = captured["preexec"]
    assert preexec is not None
    # The bound preexec hook carries the resolved limits (no RLIMIT_AS).
    assert preexec.args[0]["address_space_bytes"] is None


def test_evaluation_digest_ignores_forge_diagnostics(monkeypatch):
    task = build_family_task("triangular_cycle", 3, "qualification")
    gold = _gold(task)

    def replay(detail):
        verdict = backend_module.parse_forge_replay(134, _FORGE_ABORT)
        verdict["error_detail"]["output_tail"] = detail
        verdict["limits"] = {"address_space_bytes": None}
        return verdict

    monkeypatch.setattr(backend_module, "native_replay", lambda *_: replay("a"))
    first = DexArbitrageBackend(task).evaluate(gold)
    monkeypatch.setattr(backend_module, "native_replay", lambda *_: replay("b"))
    second = DexArbitrageBackend(task).evaluate(gold)

    assert first["evaluation_digest"] == second["evaluation_digest"]
    assert first["native_replay"]["error_detail"]["output_tail"] == "a"
    # Host diagnostics (forge's output tail) are evidence for evaluation.json,
    # not feedback the policy may see; the solidity backend strips the same
    # keys from its public view.
    public = public_evaluation(first)
    assert "error_detail" not in public["native_replay"]
    assert "limits" not in public["native_replay"]
    assert public["native_replay"]["matched"] is False
    artifacts = DexArbitrageBackend(task).artifacts(first)
    artifact_evaluation = json.loads(artifacts["evaluation.json"]["data"])
    assert artifact_evaluation["native_replay"]["error_detail"][
        "exit_code"] == 134
    assert artifact_evaluation["native_replay"]["error_detail"]["output_tail"] == "a"


@pytest.mark.parametrize("family", FAMILIES)


def test_public_solver_matches_hidden_oracle(family):
    task = build_family_task(family, 17, "benchmark")
    observation = DexArbitrageGym(task).reset_observation()
    action = solve_from_observation(observation)[0]

    assert action["pools"] == task["hidden"]["oracle_route"]
    assert int(action["amount_in_wei"]) == task["hidden"][
        "oracle_amount_in_wei"]


def test_reverse_cycle_loses_money_after_fees():
    task = build_family_task("two_pool_cycle", 5, "train")
    gold = _gold(task)
    result = route_net_profit(
        task["public"]["snapshot"]["pools"],
        list(reversed(gold["pools"])),
        int(gold["amount_in_wei"]),
        base_asset=task["public"]["base_asset"],
        gas_model=task["public"]["gas_model"],
        fee_model=task["public"]["fee_model"],
    )
    assert result["net_profit_wei"] < 0


def test_losing_route_gets_capped_partial_credit_and_no_promotion():
    """Valid, executable and unprofitable: ``positive_net_profit`` is *the
    task*, so the reversed cycle is not complete.  It keeps a bounded shaping
    signal (``<= 0.25 * progress``) but can never reach the success floor or
    promote."""
    task = build_family_task("two_pool_cycle", 5, "train")
    gold = _gold(task)
    worker = DexArbitrageGym(task)
    try:
        worker.reset_observation()
        worker.step({
            "type": "submit_route",
            "pools": list(reversed(gold["pools"])),
            "amount_in_wei": gold["amount_in_wei"],
        })
        result = worker.step({"type": "finalize"})
        evaluation = worker.state["best_evaluation"]
        assert evaluation["gates"]["route_valid"] is True
        assert evaluation["gates"]["execution_success"] is True
        assert evaluation["gates"]["positive_net_profit"] is False
        assert 0.0 < result["reward"] <= 0.25
        assert result["metadata"]["task_progress"] < 1.0
        assert "positive_net_profit" in result["metadata"]["failed_gates"]
        assert result["observation"]["completion"]["hard_gated_success"] is False
        assert result["metadata"]["promotion_eligible"] is False
    finally:
        worker.close()

    gold_worker = DexArbitrageGym(task)
    try:
        gold_worker.reset_observation()
        gold_worker.step({"type": "submit_route", **gold})
        gold_result = gold_worker.step({"type": "finalize"})
        assert gold_result["reward"] == pytest.approx(1.0)
        assert gold_result["reward"] > result["reward"]
    finally:
        gold_worker.close()


def test_unfinalized_submission_keeps_capped_step_credit():
    """``finalized`` is a soft gate (as in sql): the trainer scores an
    unfinished episode by its best step, so a submitted route must already be
    worth ``0.25 * progress`` and finalize must still pay strictly more.
    With ``finalized`` in the zero set, Qwen3-8B's never-finalizing rollouts
    all scored 0 and GRPO had no signal."""
    task = build_family_task("two_pool_cycle", 5, "train")
    gold = _gold(task)
    worker = DexArbitrageGym(task)
    try:
        worker.reset_observation()
        assert worker.step({"type": "inspect_task"})["reward"] == 0.0
        submitted = worker.step({"type": "submit_route", **gold})
        assert submitted["done"] is False
        assert submitted["reward"] == pytest.approx(0.25)
        assert "finalized" in submitted["metadata"]["failed_gates"]
        finalized = worker.step({"type": "finalize"})
        assert finalized["reward"] == pytest.approx(1.0)
    finally:
        worker.close()

    losing = DexArbitrageGym(task)
    try:
        losing.reset_observation()
        step = losing.step({
            "type": "submit_route",
            "pools": list(reversed(gold["pools"])),
            "amount_in_wei": gold["amount_in_wei"],
        })
        assert 0.0 < step["reward"] < 0.25
    finally:
        losing.close()


def test_finalize_only_pays_nothing():
    task = build_family_task("two_pool_cycle", 5, "train")
    worker = DexArbitrageGym(task)
    try:
        worker.reset_observation()
        result = worker.step({"type": "finalize"})
        assert result["reward"] == 0.0
        assert "effective_work" in result["metadata"]["failed_gates"]
    finally:
        worker.close()


def test_suboptimal_size_scores_below_oracle():
    task = build_family_task("size_sensitive_cycle", 9, "train")
    gold = _gold(task)
    minimum = int(task["public"]["constraints"]["min_amount_in_wei"])
    smaller = max(minimum, int(gold["amount_in_wei"]) // 2)
    evaluation = DexArbitrageBackend(task).evaluate({
        "pools": gold["pools"],
        "amount_in_wei": str(smaller),
    })
    assert 0 < evaluation["quality"]["profit_quality"] < 1


def test_exact_integer_amount_out_formula():
    actual = amount_out(10**18, 1_000 * 10**18, 2_000_000 * 10**18, 30)
    amount_with_fee = 10**18 * 9970
    expected = amount_with_fee * (2_000_000 * 10**18) // (
        1_000 * 10**18 * 10_000 + amount_with_fee)
    assert actual == expected


def test_submission_rejects_unknown_repeated_and_unbounded_inputs():
    task = build_family_task("two_pool_cycle", 1, "train")
    public = task["public"]
    gold = _gold(task)

    with pytest.raises(ContractError, match="not in the frozen snapshot"):
        validate_submission(public, {
            **gold, "pools": [gold["pools"][0], "0x" + "f" * 40],
        })
    with pytest.raises(ContractError, match="repeat"):
        validate_submission(public, {
            **gold, "pools": [gold["pools"][0], gold["pools"][0]],
        })
    with pytest.raises(ContractError, match="inside"):
        validate_submission(public, {
            **gold,
            "amount_in_wei": str(
                int(public["constraints"]["max_amount_in_wei"]) + 1),
        })
    with pytest.raises(UnsafeDexSubmission, match="oversized"):
        validate_submission(public, {
            **gold, "amount_in_wei": "1" * 79,
        })


def test_public_observation_and_artifacts_do_not_leak_oracle():
    task = build_family_task("competing_cycles", 2, "benchmark")
    gym = DexArbitrageGym(task)
    observation = json.dumps(gym.reset_observation(), sort_keys=True)
    assert "oracle_route" not in observation
    assert task["hidden"]["evidence_id"] not in observation

    evaluation = DexArbitrageBackend(task).evaluate(_gold(task))
    artifacts = json.dumps({
        name: payload["data"].decode("utf-8")
        for name, payload in DexArbitrageBackend(task).artifacts(evaluation).items()
    }, sort_keys=True)
    assert "oracle_route" not in artifacts
    assert task["hidden"]["evidence_id"] not in artifacts
