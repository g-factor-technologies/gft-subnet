"""Subnet window on the DEX test gym: signed HTTP, local chain, gym loop, GRPO packing, miner weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gft_subnet.gyms.dex_arbitrage import DexArbitrageTestGym
from gft_subnet.gyms.dex_arbitrage import backend as backend_module
from gft_subnet.auth import LocalAuth
from gft_subnet.chain import ChainError, LocalChain
from gft_subnet.coordinator import ack_miners, run_window, run_windows
from gft_subnet.gym_loop import run_episode
from gft_subnet.ledger import Ledger
from gft_subnet.miner import (
    MinerRuntime,
    OnPolicyStubActor,
    ScriptedActor,
    create_miner_app,
)
from gft_subnet.policy import PolicyError, PolicyRegistry
from gft_subnet.protocol import (
    Episode,
    EpisodeTurn,
    TurnRequest,
    dumps,
)
from gft_subnet.scoring import miner_credits, scores_to_weights
from gft_subnet.validator import ValidatorRuntime, create_validator_app
from gft_subnet.training import (
    group_relative_advantages,
    install_bittensor_rollout,
    pack_episodes,
    run_one_grpo_step,
)


GYM = DexArbitrageTestGym()
VALIDATOR = "validator"
MINER_A = "miner-a"
MINER_B = "miner-b"


@pytest.fixture(autouse=True)
def reference_runtime(monkeypatch):
    monkeypatch.setenv("SKILLHUB_DEX_ARBITRAGE_MODE", "reference")
    backend_module.native_runtime_info.cache_clear()
    yield
    backend_module.native_runtime_info.cache_clear()


def _secrets() -> dict[str, bytes]:
    return {
        hotkey: hashlib.sha256(hotkey.encode()).digest()
        for hotkey in (VALIDATOR, MINER_A, MINER_B)
    }


def _auth(hotkey: str) -> LocalAuth:
    return LocalAuth(_secrets(), self_hotkey=hotkey)


def _policy(tmp_path) -> PolicyRegistry:
    registry = PolicyRegistry(base_model="stub", root=tmp_path / "policy")
    registry.publish(version=1, adapter_bytes=b"adapter-v1")
    return registry


def _signed_post(client: TestClient, auth: LocalAuth, path: str, payload: dict):
    body = dumps(payload)
    headers = auth.sign("POST", path, body)
    return client.post(path, content=body, headers=headers)


def _miner_runtime(tmp_path, hotkey: str, actor) -> MinerRuntime:
    return MinerRuntime(
        hotkey=hotkey,
        auth=_auth(hotkey),
        policy=_policy(tmp_path / hotkey),
        actor=actor,
        allowed_callers={VALIDATOR, hotkey},
    )


def _served_chain(chain=None) -> LocalChain:
    chain = chain or LocalChain()
    for hotkey in (VALIDATOR, MINER_A, MINER_B):
        chain.register(hotkey)
    chain.serve_axon(MINER_A, "127.0.0.1", 8091)
    chain.serve_axon(MINER_B, "127.0.0.1", 8092)
    return chain


def test_local_chain_register_serve_dummy_weights_and_metagraph():
    chain = LocalChain(netuid=7)
    assert chain.register(VALIDATOR) == 0
    assert chain.register(MINER_A) == 1
    assert chain.register(MINER_B) == 2
    chain.serve_axon(MINER_A, "127.0.0.1", 8091)
    chain.serve_axon(MINER_B, "127.0.0.1", 8092)
    block = chain.set_weights(VALIDATOR, [1, 2], [0.5, 0.5])
    snapshot = chain.metagraph()
    assert snapshot.netuid == 7
    assert snapshot.block == block
    assert snapshot.uid_for(MINER_A) == 1
    assert [neuron.hotkey for neuron in snapshot.miners()] == [MINER_A, MINER_B]
    assert chain.last_weights(VALIDATOR) == ([1, 2], [0.5, 0.5])


def test_local_chain_persists_between_processes(tmp_path):
    path = tmp_path / "chain.json"
    first = LocalChain(netuid=3, path=path)
    first.register(VALIDATOR)
    first.register(MINER_A)
    first.serve_axon(MINER_A, "127.0.0.1", 8091)
    first.set_weights(VALIDATOR, [1], [1.0])
    second = LocalChain(path=path)
    assert second.netuid == 3
    snapshot = second.metagraph()
    assert snapshot.uid_for(MINER_A) == 1
    assert snapshot.miners()[0].axon.port == 8091
    assert second.last_weights(VALIDATOR) == ([1], [1.0])


def test_set_weights_requires_permit():
    chain = LocalChain()
    chain.register(VALIDATOR)
    chain.register(MINER_A)
    with pytest.raises(ChainError, match="permit"):
        chain.set_weights(MINER_A, [0], [1.0])


def test_cli_register_serve_dummy_weights_and_metagraph(tmp_path, capsys):
    from gft_subnet.__main__ import main

    state = str(tmp_path / "chain.json")
    common = ["--chain", "local", "--state", state]
    assert main([*common, "--hotkey", VALIDATOR, "register"]) == 0
    assert main([*common, "--hotkey", MINER_A, "register"]) == 0
    assert main([*common, "--hotkey", MINER_A, "serve-axon", "--ip", "127.0.0.1", "--port", "8091"]) == 0
    assert main([*common, "--hotkey", VALIDATOR, "set-weights", "--dummy"]) == 0
    assert main([*common, "metagraph"]) == 0
    dumped = capsys.readouterr().out.strip().splitlines()[-1]
    meta = json.loads(dumped)
    assert meta["neurons"][1]["axon"]["port"] == 8091


def test_http_auth_on_miner_and_validator(tmp_path):
    miner = _miner_runtime(tmp_path, MINER_A, ScriptedActor(MINER_A, kind="dummy"))
    miner.policy = _policy(tmp_path / "shared")
    validator_chain = LocalChain()
    validator_chain.register(VALIDATOR)
    validator = ValidatorRuntime(
        hotkey=VALIDATOR,
        auth=_auth(VALIDATOR),
        chain=validator_chain,
        policy=miner.policy,
        allowed_callers={VALIDATOR, MINER_A},
    )
    miner_client = TestClient(create_miner_app(miner))
    validator_client = TestClient(create_validator_app(validator))

    assert miner_client.get("/health").json()["role"] == "miner"
    assert validator_client.get("/health").json()["role"] == "validator"
    assert miner_client.post("/v1/ping", json={}).status_code == 401
    assert validator_client.post("/v1/ping", json={}).status_code == 401

    ping_miner = _signed_post(miner_client, _auth(VALIDATOR), "/v1/ping", {})
    ping_validator = _signed_post(validator_client, _auth(VALIDATOR), "/v1/ping", {})
    assert ping_miner.status_code == 200
    assert ping_validator.status_code == 200
    replay = miner_client.post("/v1/ping", content=dumps({}), headers=dict(ping_miner.request.headers))
    assert replay.status_code == 401

    validator_chain.register(MINER_A)
    served = _signed_post(
        validator_client,
        _auth(VALIDATOR),
        "/v1/chain/serve-axon",
        {"hotkey": MINER_A, "ip": "127.0.0.1", "port": 8091},
    )
    assert served.status_code == 200
    dummy = _signed_post(
        validator_client,
        _auth(VALIDATOR),
        "/v1/chain/set-weights",
        {"uids": [1], "weights": [1.0], "dummy": True},
    )
    assert dummy.status_code == 200
    assert dummy.json()["dummy"] is True
    meta = validator_client.get("/v1/chain/metagraph").json()
    assert meta["neurons"][1]["axon"]["port"] == 8091


def test_gym_loop_gold_beats_dummy(tmp_path):
    chain = LocalChain()
    chain.register(VALIDATOR)
    chain.register(MINER_A)
    chain.register(MINER_B)
    policy = _policy(tmp_path)
    gold = MinerRuntime(
        hotkey=MINER_A, auth=_auth(MINER_A), policy=policy,
        actor=ScriptedActor(MINER_A, kind="gold", solver=GYM.scripted_actions),
    )
    dummy = MinerRuntime(
        hotkey=MINER_B, auth=_auth(MINER_B), policy=policy,
        actor=ScriptedActor(MINER_B, kind="dummy"),
    )
    ack_miners(policy, [gold, dummy])
    gold_episode = run_episode(gym=GYM, 
        miner=gold, family="two_pool_cycle", task_seed=1, generation_seed=0, policy=policy,
    )
    dummy_episode = run_episode(gym=GYM, 
        miner=dummy, family="two_pool_cycle", task_seed=1, generation_seed=0, policy=policy,
    )
    assert gold_episode.complete
    assert dummy_episode.complete
    assert gold_episode.scripted and dummy_episode.scripted
    assert gold_episode.reward == pytest.approx(1.0)
    assert dummy_episode.reward < gold_episode.reward
    assert gold_episode.turns[0].action["type"] == "submit_route"
    assert dummy_episode.turns[0].action["type"] == "inspect_task"


def test_policy_mix_is_rejected(tmp_path):
    policy = _policy(tmp_path)
    actor = ScriptedActor(MINER_A, kind="dummy")

    class Mixed:
        scripted = True

        def generate(self, request: TurnRequest):
            response = actor.generate(request)
            response.policy_version = request.policy_version + 1
            return response

    miner = MinerRuntime(hotkey=MINER_A, auth=_auth(MINER_A), policy=policy, actor=Mixed())
    ack_miners(policy, [miner])
    with pytest.raises(PolicyError, match="mixed policy"):
        run_episode(gym=GYM, 
            miner=miner, family="two_pool_cycle", task_seed=1, generation_seed=0, policy=policy,
        )


def test_v_plus_1_requires_acks(tmp_path):
    policy = _policy(tmp_path)
    miner = MinerRuntime(
        hotkey=MINER_A, auth=_auth(MINER_A), policy=policy,
        actor=OnPolicyStubActor(MINER_A),
    )
    ack_miners(policy, [miner])
    policy.publish(version=2, adapter_bytes=b"adapter-v2")
    with pytest.raises(PolicyError, match="has not acked"):
        run_episode(gym=GYM, 
            miner=miner, family="two_pool_cycle", task_seed=1, generation_seed=0, policy=policy,
        )
    miner.ack_policy(2, policy.manifest().checksum)
    episode = run_episode(gym=GYM, 
        miner=miner, family="two_pool_cycle", task_seed=1, generation_seed=0, policy=policy,
    )
    assert episode.policy_version == 2


def _on_policy_episode(hotkey: str, reward: float, policy_version: int = 1) -> Episode:
    action_text = '{"type":"finalize"}'
    tokens = [ord(char) for char in action_text]
    return Episode(
        episode_id=f"{hotkey}-{reward}",
        hotkey=hotkey,
        family="two_pool_cycle",
        task_seed=1,
        generation_seed=0,
        policy_version=policy_version,
        turns=[
            EpisodeTurn(
                turn=0,
                action={"type": "finalize"},
                observation={"done": True},
                token_ids=tokens,
                logprobs=[-0.2] * len(tokens),
                env_token_ids=[10, 11, 12],
                prompt_ids=[5, 6, 7],
            )
        ],
        reward=reward,
        complete=True,
        prompt="route the swap",
        scripted=False,
    )


def test_scripted_rows_are_not_packed_for_grpo():
    scripted = _on_policy_episode(MINER_A, 1.0)
    scripted.scripted = True
    packed = pack_episodes([scripted, _on_policy_episode(MINER_B, 0.1)])
    assert packed["n"] == 1
    assert packed["hotkeys"] == [MINER_B]


def test_pack_episodes_sets_env_mask_on_gym_tokens():
    episode = _on_policy_episode(MINER_A, 0.4)
    first = episode.turns[0]
    episode.turns.append(EpisodeTurn(
        turn=1, action={"type": "finalize"}, observation={},
        token_ids=[20], logprobs=[-0.3], env_token_ids=[],
        prompt_ids=first.prompt_ids + first.token_ids + [10, 11, 12],
    ))
    packed = pack_episodes([episode])
    mask = packed["env_mask"][0]
    completion = packed["completion_ids"][0]
    assert packed["tool_mask"][0] == mask
    assert mask[-4:] == [0, 0, 0, 1]
    assert all(value == 1 for value in mask[:-4])
    assert len(mask) == len(completion)


def test_one_grpo_step_and_rollout_func(tmp_path):
    episodes = [_on_policy_episode(MINER_A, 1.0), _on_policy_episode(MINER_B, 0.0)]
    before = PolicyRegistry(base_model="stub", root=tmp_path / "policy")
    manifest = before.publish(version=1, adapter_bytes=b"adapter-v1")
    trainer = SimpleNamespace(environment_factory=object(), rollout_func=None)
    with pytest.raises(PolicyError, match="real PEFT"):
        run_one_grpo_step(episodes, adapter_dir=manifest.adapter_dir, trainer=trainer)
    install_bittensor_rollout(trainer, episodes, group_size=2)
    groups = pack_episodes(episodes, group_size=2)["groups"]
    rollout = trainer.rollout_func(groups, trainer)
    assert len(rollout["completion_ids"]) == 2
    assert "env_mask" in rollout
    assert group_relative_advantages([1.0, 0.0], group_size=2) == [0.5, -0.5]
    installed = install_bittensor_rollout(SimpleNamespace(environment_factory=1), episodes)
    assert callable(installed.rollout_func)


def test_behavior_logprobs_compare_bf16_to_float():
    torch = pytest.importorskip("torch")
    from gft_subnet.verify import behavior_logprobs_match

    got = torch.tensor([0.25, -1.5, 0.0], dtype=torch.bfloat16)
    recorded = torch.tensor([0.25, -1.5, 9.0], dtype=torch.float32)
    mask = torch.tensor([True, True, False])
    assert behavior_logprobs_match(got, recorded, mask)
    recorded_bad = recorded.clone()
    recorded_bad[0] = 9.0
    assert not behavior_logprobs_match(got, recorded_bad, mask)
    # A loose per-token max still rejects a consistent shift (other adapter/temperature).
    shifted = got.float() + 0.5
    assert behavior_logprobs_match(got, shifted, mask, atol=1.0, mean_atol=1.0)
    assert not behavior_logprobs_match(got, shifted, mask, atol=1.0, mean_atol=0.1)


def test_window_weights_follow_accepted_slots_not_reward(tmp_path):
    chain = _served_chain()
    policy = _policy(tmp_path)
    gold = MinerRuntime(hotkey=MINER_A, auth=_auth(MINER_A), policy=policy,
                        actor=ScriptedActor(MINER_A, kind="gold", solver=GYM.scripted_actions))
    dummy = MinerRuntime(hotkey=MINER_B, auth=_auth(MINER_B), policy=policy,
                         actor=ScriptedActor(MINER_B, kind="dummy"))
    result = run_window(gym=GYM, 
        chain=chain, validator_hotkey=VALIDATOR, miners=[gold, dummy], policy=policy,
        ledger=Ledger(tmp_path / "ledger.sqlite"), families=["two_pool_cycle"],
        group_size=4, tasks_per_window=1, train=False,
    )
    # Gold earns higher reward, but both did the same verified work: equal pay.
    assert result.extras["dealt"] == {MINER_A: 2, MINER_B: 2}
    assert result.extras["credits"] == {MINER_A: 2.0, MINER_B: 2.0}
    assert (result.uids, result.weights) == ([1, 2], [0.5, 0.5])
    assert chain.last_weights(VALIDATOR) == (result.uids, result.weights)
    assert scores_to_weights({MINER_A: 0.0, MINER_B: 0.0}, chain.metagraph()) == ([], [])


def test_window_skips_chain_weight_rate_limit(tmp_path):
    class Limited(LocalChain):
        def set_weights(self, validator_hotkey, uids, weights, **kwargs):
            del validator_hotkey, uids, weights, kwargs
            raise RuntimeError(
                "Weights on netuid 572 were last set 33 blocks ago; "
                "the rate limit is 100 blocks. Retry in ~67 blocks"
            )

    chain = Limited()
    chain.register(VALIDATOR)
    chain.register(MINER_A)
    chain.register(MINER_B)
    chain.serve_axon(MINER_A, "127.0.0.1", 8091)
    chain.serve_axon(MINER_B, "127.0.0.1", 8092)
    policy = _policy(tmp_path)
    result = run_window(gym=GYM, 
        chain=chain,
        validator_hotkey=VALIDATOR,
        miners=[
            MinerRuntime(hotkey=MINER_A, auth=_auth(MINER_A), policy=policy,
                         actor=ScriptedActor(MINER_A, kind="gold", solver=GYM.scripted_actions)),
            MinerRuntime(hotkey=MINER_B, auth=_auth(MINER_B), policy=policy,
                         actor=ScriptedActor(MINER_B, kind="dummy")),
        ],
        policy=policy,
        ledger=Ledger(tmp_path / "ledger.sqlite"),
        families=["two_pool_cycle"],
        # One task, 2 rollouts each: the group always mixes gold and dummy.
        group_size=4,
        tasks_per_window=1,
        dummy_weights=False,
        train=False,
    )
    assert result.extras["weights_skipped"] == "rate_limit"
    assert result.uids == []
    assert chain.last_weights(VALIDATOR) is None


def test_window_skips_cuda_oom_and_keeps_loop_alive(tmp_path, monkeypatch):
    from gft_subnet.coordinator import _is_cuda_oom

    assert _is_cuda_oom(RuntimeError(
        "CUDA out of memory. Tried to allocate 2.90 GiB. GPU 0 has a total capacity of 44.39 GiB"))
    assert not _is_cuda_oom(RuntimeError("cpu ram exhausted"))

    def boom(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.90 GiB")

    result, _, _ = _training_window(tmp_path, monkeypatch, boom)
    assert result.extras["training_skipped"] == "oom"
    assert result.next_policy is None


def test_validator_chooses_single_use_tasks_in_family_rotation(tmp_path):
    from gft_subnet.coordinator import choose_window_tasks

    assert choose_window_tasks(
        families=["a", "b", "c"], count=2, window_id=1, used_tasks={("c", 4)},
    ) == [("c", 5), ("a", 1)]
    chain = _served_chain()
    policy = _policy(tmp_path)
    miners = [MinerRuntime(hotkey=hotkey, auth=_auth(hotkey), policy=policy,
                           actor=OnPolicyStubActor(hotkey)) for hotkey in (MINER_A, MINER_B)]
    results = run_windows(gym=GYM, 
        steps=2, chain=chain, validator_hotkey=VALIDATOR, miners=miners, policy=policy,
        ledger=Ledger(tmp_path / "ledger.sqlite"), families=["two_pool_cycle", "triangular_cycle"],
        group_size=2, tasks_per_window=2, train=False,
    )
    tasks = [tuple(task) for result in results for task in result.extras["tasks"]]
    assert len(tasks) == 4 == len(set(tasks))
    for result in results:
        assert {(ep.family, ep.task_seed) for ep in result.episodes} == {
            tuple(task) for task in result.extras["tasks"]}
    assert results[-1].trainer is None


def test_rollouts_are_dealt_in_proportion_to_throughput():
    import random

    from gft_subnet.coordinator import assign_rollouts
    from gft_subnet.gym_loop import RolloutSpec

    specs = [RolloutSpec("two_pool_cycle", 1, index, f"k{index}") for index in range(16)]
    miners = [SimpleNamespace(hotkey=name) for name in ("fast", "slow", "new")]
    assignment = assign_rollouts(specs, miners, {"fast": 300.0, "slow": 100.0}, random.Random(0))
    counts = {hotkey: len(dealt) for hotkey, dealt in assignment.items()}
    # "new" is unmeasured and counts as the average (200 tok/s).
    assert counts == {"fast": 8, "slow": 3, "new": 5}
    assert sorted(spec.index for dealt in assignment.values() for spec in dealt) == list(range(16))
    few = assign_rollouts(specs[:2], miners, {"fast": 1000.0, "slow": 1.0, "new": 1.0},
                          random.Random(0))
    assert {hotkey: len(dealt) for hotkey, dealt in few.items()} == {"fast": 2}


def test_config_loads_yaml_families_and_sampling(tmp_path):
    from gft_subnet.__main__ import _apply_config, _parser

    args = _parser().parse_args(["run-window"])
    _apply_config(args, {
        "families": ["size_sensitive_cycle", "two_pool_cycle"],
        "seeds": [1],
        "sampling": {"temperature": 1.2, "max_tokens": 256},
        "train_steps": 50,
    })
    assert args.families[0] == "size_sensitive_cycle"
    assert args.sampling["temperature"] == 1.2
    assert args.steps == 50


def test_task_groups_mix_miners_with_unique_forced_sampling_keys(tmp_path):
    chain = _served_chain()
    policy = _policy(tmp_path)
    miners = [MinerRuntime(hotkey=hotkey, auth=_auth(hotkey), policy=policy,
                           actor=OnPolicyStubActor(hotkey)) for hotkey in (MINER_A, MINER_B)]
    result = run_window(gym=GYM, 
        chain=chain, validator_hotkey=VALIDATOR, miners=miners, policy=policy,
        ledger=Ledger(tmp_path / "ledger.sqlite"), families=["two_pool_cycle"],
        group_size=4, tasks_per_window=2, train=False,
    )
    assert len(result.episodes) == 8 and result.extras["missing_rollouts"] == 0
    assert sum(result.extras["dealt"].values()) == 8
    for family, seed in result.extras["tasks"]:
        group = [ep for ep in result.episodes if (ep.family, ep.task_seed) == (family, seed)]
        assert sorted(ep.generation_seed for ep in group) == [0, 1, 2, 3]
    assert len({ep.sampling_key for ep in result.episodes}) == 8
    # Identical stub work: equal verified token credits and equal weights, even with zero advantage.
    assert result.extras["credits"][MINER_A] == result.extras["credits"][MINER_B] > 0
    assert result.uids == [1, 2]
    assert result.weights == pytest.approx([0.5, 0.5])
    assert chain.last_weights(VALIDATOR) == (result.uids, result.weights)


def test_turn_timeout_marks_infra_error(tmp_path):
    import time

    policy = _policy(tmp_path)
    inner = OnPolicyStubActor(MINER_A)

    class Slow:
        hotkey = MINER_A
        scripted = False

        def generate(self, request: TurnRequest):
            time.sleep(0.3)
            return inner.generate(request)

    miner = MinerRuntime(hotkey=MINER_A, auth=_auth(MINER_A), policy=policy, actor=Slow())
    ack_miners(policy, [miner])
    started = time.monotonic()
    episode = run_episode(gym=GYM, 
        miner=miner, family="two_pool_cycle", task_seed=1, generation_seed=0, policy=policy,
        turn_timeout_seconds=0.05,
    )
    assert time.monotonic() - started < 0.2
    assert episode.complete is False
    assert episode.infra_error is not None
    assert "Timeout" in episode.infra_error


def test_policy_adopt_and_http_push(tmp_path):
    from gft_subnet.policy import read_adapter_files

    src = PolicyRegistry(base_model="stub", root=tmp_path / "src")
    manifest = src.publish(version=1, adapter_bytes=b"adapter-v1")
    dest = PolicyRegistry(base_model="stub", root=tmp_path / "dest")
    adopted = dest.adopt(manifest, read_adapter_files(manifest.adapter_dir))
    assert adopted.version == 1
    assert adopted.checksum == manifest.checksum

    miner = MinerRuntime(
        hotkey=MINER_A,
        auth=_auth(MINER_A),
        policy=PolicyRegistry(base_model="stub", root=tmp_path / "miner-policy"),
        actor=OnPolicyStubActor(MINER_A),
        allowed_callers={VALIDATOR, MINER_A},
    )
    client = TestClient(create_miner_app(miner))
    import base64

    payload = manifest.to_dict()
    payload["adapter_files"] = {
        name: base64.b64encode(blob).decode("ascii")
        for name, blob in read_adapter_files(manifest.adapter_dir).items()
    }
    response = _signed_post(client, _auth(VALIDATOR), "/v1/policy", payload)
    assert response.status_code == 200
    assert miner.policy.current_version == 1
    assert MINER_A in miner.policy.acked(1)
    assert response.json()["slot"] == "dex_arbitrage-v1"


def test_gym_lora_slots_and_generate_does_not_ship_files(tmp_path):
    from gft_subnet.policy import read_adapter_files
    from gft_subnet.protocol import TurnResponse, policy_slot

    class Recorder:
        scripted = False

        def __init__(self):
            self.worn: list[str] = []

        def load_policy(self, manifest):
            if self.worn[-1:] != [manifest.slot]:
                self.worn.append(manifest.slot)

        def generate(self, request):
            assert "adapter_files" not in request.to_dict()
            text = json.dumps({"type": "inspect_task"})
            return TurnResponse(
                policy_version=request.policy_version, episode_id=request.episode_id,
                hotkey=MINER_A, action={}, text=text, token_ids=[1], logprobs=[0.0],
                prompt_ids=[0], scripted=False,
            )

    src = PolicyRegistry(base_model="stub", root=tmp_path / "src")
    dex = src.publish(version=1, adapter_bytes=b"dex-v1")
    sql = src.publish(version=1, adapter_bytes=b"sql-v1", gym_id="sql")
    assert dex.slot == policy_slot("dex_arbitrage", 1)
    assert sql.slot == "sql-v1"

    actor = Recorder()
    miner = MinerRuntime(
        hotkey=MINER_A, auth=_auth(MINER_A),
        policy=PolicyRegistry(base_model="stub", root=tmp_path / "miner"),
        actor=actor, allowed_callers={VALIDATOR, MINER_A},
    )
    client = TestClient(create_miner_app(miner))
    import base64
    validator = _auth(VALIDATOR)

    def push(manifest):
        payload = manifest.to_dict()
        payload["adapter_files"] = {
            name: base64.b64encode(blob).decode("ascii")
            for name, blob in read_adapter_files(manifest.adapter_dir).items()
        }
        posted = _signed_post(client, validator, "/v1/policy", payload)
        assert posted.status_code == 200
        assert posted.json()["slot"] == manifest.slot

    push(dex)
    push(sql)
    assert actor.worn == ["dex_arbitrage-v1", "sql-v1"]
    assert client.get("/health").json()["attestation"]["mode"] == "none"
    sql_turn = {
        "policy_version": 1, "episode_id": "ep-2", "turn": 0, "family": "two_pool_cycle",
        "task_seed": 1, "generation_seed": 0, "observation": {}, "messages": [],
        "gym_id": "sql",
    }
    dex_turn = {**sql_turn, "gym_id": "dex_arbitrage", "episode_id": "ep-1"}
    assert _signed_post(client, validator, "/v1/generate", sql_turn).status_code == 200
    assert actor.worn == ["dex_arbitrage-v1", "sql-v1"]
    assert _signed_post(client, validator, "/v1/generate", dex_turn).status_code == 200
    assert actor.worn == ["dex_arbitrage-v1", "sql-v1", "dex_arbitrage-v1"]
    assert _signed_post(client, validator, "/v1/generate", dex_turn).status_code == 200
    assert actor.worn == ["dex_arbitrage-v1", "sql-v1", "dex_arbitrage-v1"]
    blocked = _signed_post(
        client, validator, "/v1/generate", {**dex_turn, "adapter_files": {"x": "y"}},
    )
    assert blocked.status_code == 400
    unknown = _signed_post(
        client, validator, "/v1/generate", {**dex_turn, "gym_id": "lean"},
    )
    assert unknown.status_code == 409


def test_lora_push_skips_attestation_until_confidential_is_required(tmp_path):
    from gft_subnet.attestation import (
        MODE_CONFIDENTIAL_GPU, MinerAttestation,
    )
    from gft_subnet.coordinator import _push_policy

    policy = _policy(tmp_path)
    l40s = MinerRuntime(
        hotkey=MINER_A, auth=_auth(MINER_A), policy=_policy(tmp_path / "l40s"),
        actor=OnPolicyStubActor(MINER_A),
        attestation=MinerAttestation(device="L40S"),
    )
    _push_policy([l40s], policy.manifest(), require_confidential=False)
    with pytest.raises(PolicyError, match="confidential GPU"):
        _push_policy([l40s], policy.manifest(), require_confidential=True)
    h100 = MinerRuntime(
        hotkey=MINER_B, auth=_auth(MINER_B), policy=_policy(tmp_path / "h100"),
        actor=OnPolicyStubActor(MINER_B),
        attestation=MinerAttestation(
            mode=MODE_CONFIDENTIAL_GPU, platform="nvidia-cc", quote="placeholder",
            device="H100",
        ),
    )
    with pytest.raises(PolicyError, match="not wired"):
        _push_policy([h100], policy.manifest(), require_confidential=True)


def test_miners_from_metagraph_skips_validator():
    from gft_subnet.miner import miners_from_metagraph

    chain = LocalChain(netuid=4)
    chain.register(VALIDATOR)
    chain.register(MINER_A)
    chain.register(MINER_B)
    chain.serve_axon(MINER_A, "10.0.0.1", 8091)
    chain.serve_axon(MINER_B, "10.0.0.2", 8092)
    miners = miners_from_metagraph(
        chain.metagraph(), auth=_auth(VALIDATOR), skip_hotkeys={VALIDATOR},
    )
    assert [miner.hotkey for miner in miners] == [MINER_A, MINER_B]
    assert miners[0].base_url == "http://10.0.0.1:8091"


def test_fake_trainer_cannot_publish_an_adapter(tmp_path):
    episodes = [_on_policy_episode(MINER_A, 1.0), _on_policy_episode(MINER_B, 0.0)]
    before = PolicyRegistry(base_model="stub", root=tmp_path / "policy")
    manifest = before.publish(version=1, adapter_bytes=b"adapter-v1")

    def save_model(path):
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "adapter_config.json").write_text("{}", encoding="utf-8")
        (root / "adapter_model.safetensors").write_bytes(b"lora")

    trainer = SimpleNamespace(environment_factory=object(), train=lambda: None, save_model=save_model)
    with pytest.raises(PolicyError, match="real PEFT"):
        run_one_grpo_step(episodes, adapter_dir=manifest.adapter_dir, trainer=trainer)


def test_bittensor_auth_wraps_sdk_http_auth(monkeypatch):
    import sys

    from gft_subnet.auth import AuthError, BittensorAuth

    class FakeAuthError(Exception):
        pass

    class FakeHttpAuth:
        AuthError = FakeAuthError

        @staticmethod
        def sign(wallet, method, path, body, receiver_ss58=None):
            del method, path, body
            return {
                "X-Bittensor-Hotkey": wallet.hotkey.ss58_address,
                "X-Bittensor-Receiver": receiver_ss58 or "",
            }

        @staticmethod
        def verify(headers, body, method, path, self_hotkey_ss58):
            del body, method, path
            if headers.get("X-Bittensor-Receiver") not in {"", None, self_hotkey_ss58}:
                raise FakeAuthError("wrong receiver")
            return SimpleNamespace(hotkey_ss58=headers["X-Bittensor-Hotkey"])

    monkeypatch.setitem(sys.modules, "bittensor", SimpleNamespace(http_auth=FakeHttpAuth))
    sender = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Sender"))
    auth = BittensorAuth(sender, self_hotkey="5Sender")
    headers = auth.sign("POST", "/v1/generate", b"{}", receiver_ss58="5Miner")
    assert headers["X-Bittensor-Receiver"] == "5Miner"
    miner_auth = BittensorAuth(
        SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Miner")),
        self_hotkey="5Miner",
    )
    assert miner_auth.verify("POST", "/v1/generate", b"{}", headers) == "5Sender"
    with pytest.raises(AuthError, match="wrong receiver"):
        miner_auth.verify(
            "POST", "/v1/generate", b"{}", {**headers, "X-Bittensor-Receiver": "5Other"},
        )


def test_bittensor_chain_execute_intents(monkeypatch):
    import sys

    from gft_subnet.chain import BittensorChain

    captured: list[Any] = []

    class FakeResult:
        success = True
        block = 42
        netuid = 9
        explorer_url = "https://example.test/tx"

        def raise_for_failure(self):
            return None

    class FakeSubnets:
        def metagraph(self, netuid):
            del netuid
            return [
                SimpleNamespace(
                    uid=0, hotkey="5Owner", coldkey="5OwnerCold",
                    axon=None, validator_permit=True, tao_stake=1.0,
                ),
                SimpleNamespace(
                    uid=1, hotkey="5Miner", coldkey="5MinerCold",
                    axon="203.0.113.7:8091", validator_permit=False, tao_stake=0.0,
                ),
            ]

    class FakeSubtensor:
        def __init__(self, network="finney"):
            self.network = network
            self.block = 42
            self.subnets = FakeSubnets()

        def execute(self, intent, wallet):
            captured.append((intent.op, intent, wallet))
            return FakeResult()

    def _intent(name):
        def factory(**kwargs):
            return SimpleNamespace(op=name, **kwargs)
        return factory

    fake_bt = SimpleNamespace(
        Subtensor=FakeSubtensor,
        BurnedRegister=_intent("BurnedRegister"),
        ServeAxon=_intent("ServeAxon"),
        SetWeights=_intent("SetWeights"),
        RegisterSubnet=_intent("RegisterSubnet"),
        StartCall=_intent("StartCall"),
    )
    monkeypatch.setitem(sys.modules, "bittensor", fake_bt)
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Owner"))
    chain = BittensorChain(netuid=1, network="test", wallet=wallet)
    assert chain.create_subnet() == 9
    assert chain.netuid == 9
    chain.start_subnet()
    assert chain.burned_register() == 0
    axon = chain.serve_axon("5Miner", "203.0.113.7", 8091)
    assert axon.port == 8091
    block = chain.set_weights("5Owner", [1], [1.0], version_key=3)
    assert block == 42
    snapshot = chain.metagraph()
    assert snapshot.miners()[0].axon.ip == "203.0.113.7"
    assert snapshot.miners()[0].axon.port == 8091
    ops = [row[0] for row in captured]
    assert ops[0] == "RegisterSubnet"
    assert ops[1] == "StartCall"
    assert ops[2] == "BurnedRegister"
    assert ops[3] == "ServeAxon"
    assert ops[4] == "SetWeights"
    weights_intent = captured[4][1]
    assert weights_intent.weights == {1: 1.0}
    assert weights_intent.version_key == 3
    assert chain.explorer_url() == "https://example.test/tx"


def test_load_wallet_uses_name_kwargs(monkeypatch):
    import sys

    from gft_subnet.wallets import coldkey_ss58, hotkey_ss58, load_wallet

    seen: dict[str, str] = {}

    class FakeWallet:
        def __init__(self, *, name, hotkey="default"):
            seen["name"] = name
            seen["hotkey"] = hotkey
            self.hotkey = SimpleNamespace(ss58_address="5Hot")
            self.coldkeypub = SimpleNamespace(ss58_address="5Cold")

    monkeypatch.setitem(sys.modules, "bittensor", SimpleNamespace(Wallet=FakeWallet))
    wallet = load_wallet("owner", "default")
    assert seen == {"name": "owner", "hotkey": "default"}
    assert hotkey_ss58(wallet) == "5Hot"
    assert coldkey_ss58(wallet) == "5Cold"


def test_cli_wallets_and_local_burned_register(tmp_path, capsys, monkeypatch):
    from gft_subnet.__main__ import main

    def fake_load(name, hotkey="default"):
        return SimpleNamespace(
            hotkey=SimpleNamespace(ss58_address=f"hot-{name}"),
            coldkeypub=SimpleNamespace(ss58_address=f"cold-{name}"),
        )

    monkeypatch.setattr("gft_subnet.wallets.load_wallet", fake_load)
    assert main(["--chain", "test", "wallets"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["wallet"] for row in payload["wallets"]] == ["owner", "miner-a", "miner-b"]
    assert "mnemonic" in payload["discord"].lower()

    state = str(tmp_path / "chain.json")
    assert main(["--chain", "local", "--state", state, "--hotkey", MINER_A, "burned-register"]) == 0
    assert main(["--chain", "local", "--state", state, "create-subnet"]) == 0
    assert main(["--chain", "local", "--state", state, "start-subnet"]) == 0
    dumped = capsys.readouterr().out.strip().splitlines()
    assert json.loads(dumped[0])["uid"] == 0
    assert json.loads(dumped[1])["netuid"] == 1


def test_actor_parses_json_and_rejects_stub_adapter(tmp_path):
    from gft_subnet.actor import (
        HuggingFacePolicyActor, chat_template_kwargs,
    )
    from gft_subnet.gym_loop import parse_json_action

    assert chat_template_kwargs("Qwen/Qwen3-8B") == {"enable_thinking": False}
    assert chat_template_kwargs("Qwen/Qwen2.5-0.5B-Instruct") == {}
    assert parse_json_action('say {"type":"inspect_task"} please')["type"] == "inspect_task"
    assert GYM.normalize_action(parse_json_action(
        '{"type":"submit_route","pools":["0xabc"],"amount_in_wei":100}',
    )) == {"type": "submit_route", "pools": ["0xabc"], "amount_in_wei": "100"}
    assert parse_json_action("not json")["type"] == "invalid"
    assert parse_json_action(
        '<think>plan</think>{"type":"inspect_task"}',
    )["type"] == "inspect_task"
    from gft_subnet.gyms.dex_arbitrage import JSON_ACTION_REPLY
    assert "one line" in JSON_ACTION_REPLY
    assert "min_amount_in_wei" in JSON_ACTION_REPLY
    actor = HuggingFacePolicyActor(hotkey=MINER_A, model_id="stub")
    actor._model = SimpleNamespace()
    adapter = tmp_path / "v1"
    adapter.mkdir()
    (adapter / "adapter.bin").write_bytes(b"v1")
    with pytest.raises(PolicyError, match="real PEFT"):
        actor.load_adapter(str(adapter))
    assert actor._loaded_checksum is None


@pytest.mark.parametrize("route,payload", [
    ("/v1/chain/register", {}),
    ("/v1/chain/serve-axon", {"ip": "127.0.0.1", "port": 9000}),
    ("/v1/chain/set-weights", {"uids": [1], "weights": [1.0]}),
])
def test_validator_chain_admin_is_not_delegated_to_miners(tmp_path, route, payload):
    chain = LocalChain()
    chain.register(VALIDATOR)
    runtime = ValidatorRuntime(hotkey=VALIDATOR, auth=_auth(VALIDATOR), chain=chain,
                               policy=_policy(tmp_path), allowed_callers={MINER_A})
    client = TestClient(create_validator_app(runtime))
    assert _signed_post(client, _auth(MINER_A), route, payload).status_code == 403
    assert chain.last_weights(VALIDATOR) is None


@pytest.mark.parametrize("route,payload", [
    ("/v1/policy", {"version": 2}),
    ("/v1/policy/ack", {"version": 1, "checksum": "bad"}),
    ("/v1/generate", {}),
    ("/v1/generate_batch", {"requests": []}),
])
def test_miner_http_denies_unspecified_callers(tmp_path, route, payload):
    runtime = MinerRuntime(hotkey=MINER_A, auth=_auth(MINER_A), policy=_policy(tmp_path),
                           actor=ScriptedActor(MINER_A, solver=GYM.scripted_actions))
    client = TestClient(create_miner_app(runtime))
    assert _signed_post(client, _auth(VALIDATOR), route, payload).status_code == 403


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../../escape", "a//b", ".", "a\\b"])
def test_adapter_paths_rejected_before_any_write(tmp_path, name):
    from gft_subnet.policy import write_adapter_files

    root = tmp_path / "policy" / "v1"
    with pytest.raises(PolicyError, match="invalid adapter file"):
        write_adapter_files(root, {"good": b"ok", name: b"bad"}, version=1)
    assert not root.parent.exists()


def test_checksum_failure_and_symlinks_do_not_mutate_policy(tmp_path):
    from gft_subnet.policy import adapter_checksum, write_adapter_files

    policy = _policy(tmp_path)
    manifest = policy.manifest()
    with pytest.raises(PolicyError, match="checksum"):
        write_adapter_files(manifest.adapter_dir, {"bad": b"bad"}, version=1,
                            expected_checksum=manifest.checksum)
    assert adapter_checksum(manifest.adapter_dir) == manifest.checksum
    link = tmp_path / "link"
    link.symlink_to(Path(manifest.adapter_dir), target_is_directory=True)
    with pytest.raises(PolicyError, match="symlink"):
        write_adapter_files(link / "v2", {"bad": b"bad"}, version=2)
    assert not (Path(manifest.adapter_dir) / "v2").exists()


def test_registry_resume_repushes_current_policy_and_rejects_rollback(tmp_path):
    from gft_subnet.coordinator import _push_policy
    from gft_subnet.policy import read_adapter_files

    original = _policy(tmp_path / "validator")
    v1 = original.manifest()
    miner_policy = PolicyRegistry(base_model="stub", root=tmp_path / "miner")
    miner_policy.adopt(v1, read_adapter_files(v1.adapter_dir))
    original.publish(version=2, adapter_bytes=b"v2")
    resumed = PolicyRegistry(base_model="stub", root=original.root)
    assert resumed.current_version == 2
    assert resumed.acked(2) == set()
    miner = MinerRuntime(hotkey=MINER_A, auth=_auth(MINER_A), policy=miner_policy,
                         actor=ScriptedActor(MINER_A, solver=GYM.scripted_actions))
    _push_policy([miner], resumed.manifest())
    ack_miners(resumed, [miner])
    assert miner.policy.manifest().checksum == resumed.manifest().checksum
    with pytest.raises(PolicyError, match="rollback"):
        miner.policy.adopt(v1, read_adapter_files(v1.adapter_dir))


def test_apply_sampling_survives_registry_reload(tmp_path):
    policy = _policy(tmp_path)
    policy.apply_sampling({"temperature": 1.2, "max_tokens": 256})
    reloaded = PolicyRegistry(base_model="stub", root=policy.root)
    assert reloaded.manifest().sampling == {"temperature": 1.2, "max_tokens": 256}


def test_config_preserves_explicit_cli_values_even_when_equal_to_defaults():
    from gft_subnet.__main__ import _apply_config, _parser

    args = _parser().parse_args([
        "--netuid", "99", "--network", "test", "run-window", "--group-size", "2",
        "--version-key", "0", "--family", "two_pool_cycle",
    ])
    _apply_config(args, {"netuid": 1, "network": "finney", "group_size": 4,
                         "version_key": 7, "families": ["four_hop_cycle"]})
    assert (args.netuid, args.network, args.group_size, args.version_key) == (99, "test", 2, 0)
    assert args.families == ["two_pool_cycle"]


def test_sdk_real_result_exposes_new_subnet_in_data():
    bt = pytest.importorskip("bittensor")
    from gft_subnet.chain import BittensorChain

    chain = BittensorChain(netuid=1, wallet=object())
    chain._execute = lambda intent: bt.ExtrinsicResult(success=True, data={"netuid": 99})
    assert chain.create_subnet() == 99


def test_prompts_include_entire_public_snapshot_and_feedback():
    from gft_subnet.gym_loop import _messages

    observations = [GYM.make("two_pool_cycle", seed, "train") for seed in (1, 2)]
    initial = [gym.reset_observation() for gym in observations]
    prompts = [_messages(GYM, obs) for obs in initial]
    assert prompts[0] != prompts[1]
    for obs, prompt in zip(initial, prompts, strict=True):
        content = prompt[-1]["content"]
        assert json.dumps(obs, sort_keys=True, separators=(",", ":")) in content
        assert "inspect_task" in content
        assert "single JSON object" in content
    after = observations[0].step({"type": "inspect_evaluation"})["observation"]
    assert _messages(GYM, after) != prompts[0]


def test_groups_drop_infrastructure_failures_but_keep_model_failures():
    episodes = [_on_policy_episode(str(i), reward) for i, reward in enumerate([1., 0., 1., 0.])]
    for ep in episodes[2:]:
        ep.task_seed = 2
    episodes[0].complete = False  # model stopped before finalize; still a training example
    episodes[2].infra_error = "OSError: verifier unavailable"
    packed = pack_episodes(episodes, group_size=2)
    assert packed["episode_ids"] == [ep.episode_id for ep in episodes[:2]]
    episodes[1].infra_error = "TimeoutError: transport"
    assert pack_episodes(episodes, group_size=2)["n"] == 0


def test_packing_rejects_rewritten_context_and_mixed_policy():
    episode = _on_policy_episode(MINER_A, 1.)
    episode.turns.append(EpisodeTurn(1, {}, {}, [20], [-.2], [], prompt_ids=[999]))
    with pytest.raises(ValueError, match="changed"):
        pack_episodes([episode])
    with pytest.raises(ValueError, match="mixed policy"):
        pack_episodes([_on_policy_episode(MINER_A, 1., 1), _on_policy_episode(MINER_B, 0., 2)])


def test_rollout_rewards_follow_selected_task_groups():
    from gft_subnet.training import rollout_func_from_episodes

    episodes = [_on_policy_episode(str(i), reward) for i, reward in enumerate([1., .2, .4, .1])]
    for ep in episodes[2:]:
        ep.task_seed = 2
    packed = pack_episodes(episodes, group_size=2)
    rollout = rollout_func_from_episodes(episodes, group_size=2)
    result = rollout(packed["groups"][2:] + packed["groups"][:2], None)
    assert result["verified_reward"] == [.4, .1, 1., .2]
    with pytest.raises(ValueError, match="partial"):
        rollout(packed["groups"][:1], None)


def test_training_errors_are_not_replaced_with_a_fake_adapter(tmp_path, monkeypatch):
    from gft_subnet.coordinator import _make_trainer

    def fail(**kwargs):
        raise RuntimeError("out of memory")
    monkeypatch.setattr("gft_subnet.training.make_one_step_trainer", fail)
    with pytest.raises(RuntimeError, match="out of memory"):
        _make_trainer(model_id="model", output_dir=str(tmp_path), episodes=[], group_size=2,
                      adapter_dir=str(tmp_path), temperature=1.)
    assert list(tmp_path.iterdir()) == []


def test_testnet_server_cannot_fall_back_to_local_hmac():
    from gft_subnet.__main__ import main

    with pytest.raises(SystemExit):
        main(["--chain", "test", "serve-miner", "--validator-hotkey", "validator"])


def test_rollouts_of_a_timed_out_miner_are_redealt(tmp_path):
    from threading import Event

    release = Event()
    policy = _policy(tmp_path)

    class BlockedActor:
        scripted = True

        def generate(self, request):
            release.wait(2)
            return OnPolicyStubActor(MINER_A).generate(request)

    slow = MinerRuntime(hotkey=MINER_A, auth=_auth(MINER_A), policy=policy, actor=BlockedActor())
    fast = MinerRuntime(hotkey=MINER_B, auth=_auth(MINER_B), policy=policy,
                        actor=OnPolicyStubActor(MINER_B))
    try:
        result = run_window(gym=GYM, 
            chain=_served_chain(), validator_hotkey=VALIDATOR, miners=[slow, fast],
            policy=policy, ledger=Ledger(tmp_path / "ledger.sqlite"), families=["two_pool_cycle"],
            group_size=2, tasks_per_window=1, train=False, turn_timeout_seconds=.05,
        )
        # The slow miner's rollout went to the fast miner; forced sampling makes it equivalent.
        assert result.extras["failed_miners"] == [MINER_A]
        assert result.extras["missing_rollouts"] == 0
        assert {ep.hotkey for ep in result.episodes} == {MINER_B}
        assert result.scores[MINER_A] == 0.0
    finally:
        release.set()


def test_miner_protocol_violation_fails_that_miner_not_the_window(tmp_path):
    policy = _policy(tmp_path)

    class WrongVersion:
        scripted = True

        def generate(self, request):
            response = OnPolicyStubActor(MINER_A).generate(request)
            response.policy_version += 1
            return response

    bad = MinerRuntime(hotkey=MINER_A, auth=_auth(MINER_A), policy=policy, actor=WrongVersion())
    good = MinerRuntime(hotkey=MINER_B, auth=_auth(MINER_B), policy=policy,
                        actor=OnPolicyStubActor(MINER_B))
    result = run_window(gym=GYM, 
        chain=_served_chain(), validator_hotkey=VALIDATOR, miners=[bad, good], policy=policy,
        ledger=Ledger(tmp_path / "ledger.sqlite"), families=["two_pool_cycle"],
        group_size=2, tasks_per_window=1, train=False,
    )
    assert result.extras["failed_miners"] == [MINER_A]
    assert result.extras["missing_rollouts"] == 0


def _training_window(tmp_path, monkeypatch, grpo_step, *, trainer=None, ack=None, tag=True):
    from gft_subnet import coordinator

    rewards = {MINER_A: 0.8, MINER_B: 0.1}

    def fake_collect(*, assignment, **kwargs):
        del kwargs
        pairs = [
            (spec, Episode(
                f"ep-{hotkey}-{spec.index}", hotkey, spec.family, spec.task_seed, spec.index, 1,
                [EpisodeTurn(0, {"type": "finalize"}, {}, [3, 4], [0.0, 0.0], [],
                             prompt_ids=[1, 2])],
                rewards[hotkey], True, "p", False, sampling_key=spec.sampling_key,
            ))
            for hotkey, specs in assignment.items() for spec in specs
        ]
        return pairs, {}

    made = []

    def fake_make(**kwargs):
        made.append(kwargs)
        return SimpleNamespace(model=kwargs["model"], optimizer=kwargs["optimizer"])

    monkeypatch.setattr(coordinator, "collect_rollouts", fake_collect)
    monkeypatch.setattr(coordinator, "_make_trainer", fake_make)
    monkeypatch.setattr(coordinator, "_load_policy_model", lambda model_id, adapter_dir: "loaded-model")
    monkeypatch.setattr(coordinator, "verify_episode_traces", lambda model, episodes, **kw: {})
    monkeypatch.setattr("gft_subnet.training.run_one_grpo_step", grpo_step)
    if ack is not None:
        monkeypatch.setattr(coordinator, "ack_miners", ack)
    policy = _policy(tmp_path)
    if trainer is not None and tag:
        trainer.policy_checksum = policy.manifest().checksum
    result = run_window(gym=GYM, 
        chain=_served_chain(), validator_hotkey=VALIDATOR,
        miners=[MinerRuntime(hotkey=hotkey, auth=_auth(hotkey), policy=policy,
                             actor=OnPolicyStubActor(hotkey)) for hotkey in (MINER_A, MINER_B)],
        policy=policy, ledger=Ledger(tmp_path / "ledger.sqlite"), families=["two_pool_cycle"],
        group_size=2, tasks_per_window=1, train=True, model_id="stub", trainer=trainer,
    )
    return result, made, policy


def test_trainer_model_is_dropped_when_weights_may_differ_from_published_policy(
        tmp_path, monkeypatch):
    def oom_mid_step(episodes, *, trainer, **kwargs):
        del episodes, kwargs
        trainer.policy_checksum = None
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.90 GiB")

    previous = SimpleNamespace(model="live-model", optimizer=SimpleNamespace(optimizer="adam"))
    result, made, _ = _training_window(tmp_path, monkeypatch, oom_mid_step, trainer=previous)
    assert made[0]["model"] == "live-model" and made[0]["optimizer"] == "adam"
    assert result.extras["training_skipped"] == "oom"
    assert result.trainer is None


def test_trainer_model_is_kept_when_validation_fails_before_any_update(tmp_path, monkeypatch):
    def reject_logprobs(episodes, **kwargs):
        del episodes, kwargs
        raise PolicyError("behavior logprobs do not match the current training policy")

    result, _, policy = _training_window(tmp_path, monkeypatch, reject_logprobs)
    assert result.extras["training_skipped"] == "policy_error"
    assert result.trainer.policy_checksum == policy.manifest().checksum


def test_trainer_with_stale_policy_is_not_reused(tmp_path, monkeypatch):
    def reject(episodes, **kwargs):
        del episodes, kwargs
        raise PolicyError("stop")

    stale = SimpleNamespace(model="stale", optimizer=None, policy_checksum="other")
    _, made, _ = _training_window(tmp_path, monkeypatch, reject, trainer=stale, tag=False)
    assert made[0]["model"] == "loaded-model" and made[0]["optimizer"] is None


def test_push_failure_after_publish_is_not_reported_as_skipped_training(tmp_path, monkeypatch):
    from gft_subnet import coordinator

    def publishable(episodes, *, adapter_dir, **kwargs):
        del episodes, kwargs
        target = Path(adapter_dir).parent / "trained"
        target.mkdir()
        (target / "adapter.bin").write_bytes(b"adapter-v2")
        return {"packed": {"n": 2}, "adapter_dir": str(target)}

    calls = []
    real_ack = coordinator.ack_miners

    def flaky_ack(policy, miners):
        calls.append(policy.current_version)
        if len(calls) > 1:
            raise PolicyError("miner-b did not ack")
        return real_ack(policy, miners)

    result, _, policy = _training_window(tmp_path, monkeypatch, publishable, ack=flaky_ack)
    assert result.next_policy.version == 2 == policy.current_version
    assert result.extras["training_skipped"] is None
    assert result.extras["push_error"] == "miner-b did not ack"
    assert result.trainer.policy_checksum == policy.manifest().checksum


def test_follow_up_turns_send_only_changed_observation_fields():
    from gft_subnet.gym_loop import _messages

    gym = GYM.make("two_pool_cycle", 1, "train")
    first = gym.reset_observation()
    after = gym.step({"type": "inspect_evaluation"})["observation"]
    follow_up = _messages(GYM, after, first)
    assert len(follow_up) == 1 and follow_up[0]["role"] == "user"
    content = follow_up[0]["content"]
    assert "single JSON object" in content
    assert "Action JSON Schema" not in content
    assert json.dumps(first["snapshot"], sort_keys=True, separators=(",", ":")) not in content
    assert len(content) < len(_messages(GYM, after)[-1]["content"])


class _WordTokenizer:
    """Deterministic chat tokenizer for validator trace checks."""

    words = ["<pad>", "user", "assistant", "hello", '{"type":"finalize"}', "noise", "system"]

    def __len__(self):
        return len(self.words)

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        ids = []
        for message in messages:
            ids += [self.words.index(message["role"]), 3]
        return ids + [2]

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(self.words[i] for i in ids)


def test_validator_rebuilds_context_and_decodes_text_from_tokens():
    from gft_subnet.actor import TokenTraceVerifier, turn_prompt_ids
    from gft_subnet.protocol import TurnResponse

    tokenizer = _WordTokenizer()
    verifier = TokenTraceVerifier(tokenizer, model_id="stub")
    request = TurnRequest(1, "ep", 1, "two_pool_cycle", 1, 0, {},
                          [{"role": "user", "content": "x"}], prefix_token_ids=[1, 3, 2, 5])
    prompt = turn_prompt_ids(tokenizer, request, model_id="stub")
    assert prompt == [1, 3, 2, 5, 1, 3, 2]

    def response(prompt_ids, token_ids, text):
        return TurnResponse(1, "ep", MINER_A, {}, text, token_ids, [0.0] * len(token_ids),
                            prompt_ids=prompt_ids, scripted=False)

    # The reward-bearing text comes from the trained tokens, not the miner's text field.
    assert verifier.verify(request, response(prompt, [4], "anything")) == '{"type":"finalize"}'
    with pytest.raises(ValueError, match="validator-built context"):
        verifier.verify(request, response([1, 3, 2], [4], ""))
    with pytest.raises(ValueError, match="vocabulary"):
        verifier.verify(request, response(prompt, [99], ""))


def test_untrusted_trace_is_an_infra_error_not_a_scored_episode(tmp_path):
    from gft_subnet.actor import TokenTraceVerifier

    policy = _policy(tmp_path)
    miner = _miner_runtime(tmp_path, MINER_A, OnPolicyStubActor(MINER_A))
    miner.policy = policy
    ack_miners(policy, [miner])
    episode = run_episode(gym=GYM, 
        miner=miner, family="two_pool_cycle", task_seed=1, generation_seed=0, policy=policy,
        verifier=TokenTraceVerifier(_WordTokenizer(), model_id="stub"),
    )
    assert episode.infra_error and "validator-built context" in episode.infra_error
    assert miner_credits([episode]) == {}


def test_forced_pick_is_the_only_legal_token():
    torch = pytest.importorskip("torch")
    from gft_subnet.sampling import (
        forced_pick,
        forced_uniform,
        forced_violation,
        sampling_key,
    )

    key = sampling_key("window", "two_pool_cycle", 5, "checksum", 0)
    assert key != sampling_key("window", "two_pool_cycle", 5, "checksum", 1)
    assert forced_uniform(key, 0, 3) == forced_uniform(key, 0, 3) != forced_uniform(key, 0, 4)
    torch.manual_seed(0)
    logits = torch.randn(6, 50)
    uniforms = torch.tensor([forced_uniform(key, 0, t) for t in range(6)], dtype=torch.float64)
    picks = torch.tensor([forced_pick(logits[t], float(uniforms[t]), 0.7)[0] for t in range(6)])
    distance, logp = forced_violation(logits, picks, uniforms, 0.7)
    assert torch.all(distance == 0)
    expected = (logits / 0.7).log_softmax(-1).gather(1, picks[:, None]).squeeze(1)
    assert torch.allclose(logp, expected, atol=1e-5)
    other = (picks + 1) % 50
    assert torch.all(forced_violation(logits, other, uniforms, 0.7)[0] > 0)


def test_credit_is_one_per_accepted_slot_regardless_of_reward_or_group():
    from gft_subnet.scoring import classify_trace_failures

    def ep(hotkey, reward, seed=1, error=None):
        episode = _on_policy_episode(hotkey, reward)
        episode.episode_id = f"{hotkey}-{reward}-{seed}-{error}"
        episode.task_seed, episode.infra_error = seed, error
        return episode

    episodes = [
        ep(MINER_A, 1.0), ep(MINER_B, 0.0),                   # reward does not matter
        ep(MINER_A, 0.0, seed=2),                             # incomplete group still pays
        ep(MINER_B, 0.0, seed=2, error="TraceError: tokens continue after end-of-sequence"),
        ep("warned", 0.0, seed=3, error="TraceWarning: drift"),
        ep("slow", 0.0, seed=4, error="TimeoutError: timed out"),
    ]
    assert miner_credits(episodes) == {MINER_A: 2.0, MINER_B: 1.0}
    assert classify_trace_failures(episodes) == ({MINER_B}, {"warned"})


def test_moving_credit_decays_without_work_and_resets_on_trace_error():
    from gft_subnet.scoring import credit_scores

    # Both start at 100; A then idles for 10 windows while B keeps working.
    history = [(1, {MINER_A: 100.0, MINER_B: 100.0}, set(), set())]
    history += [(w, {MINER_B: 100.0}, set(), set()) for w in range(2, 12)]
    scores = credit_scores(history, alpha=0.2)
    assert scores[MINER_A] < 0.15 * scores[MINER_B]
    # A TraceError resets moving credit; a later clean window starts from zero.
    history += [(12, {MINER_B: 100.0}, {MINER_B}, set()), (13, {MINER_B: 100.0}, set(), set())]
    assert credit_scores(history, alpha=0.2)[MINER_B] == pytest.approx(20.0)


def test_quarantine_after_trace_error_or_repeated_warnings(tmp_path):
    from gft_subnet.scoring import (
        ERROR_QUARANTINE_WINDOWS,
        WARNING_HORIZON,
        quarantined_miners,
    )

    ledger = Ledger(tmp_path / "credits.sqlite")
    windows = [ledger.open_window(1, "c") for _ in range(3)]
    ledger.record_credits(windows[0], {MINER_A: 1.0}, {MINER_B}, {"flaky"})
    ledger.record_credits(windows[1], {MINER_A: 1.0}, set(), {"flaky", "once"})
    history = ledger.credit_history()
    assert [row[0] for row in history] == windows[:2]
    assert quarantined_miners(history, window_id=windows[2]) == {MINER_B, "flaky"}
    # Warnings age out of the horizon; a TraceError benches for longer.
    later = windows[1] + WARNING_HORIZON + 1
    assert quarantined_miners(history, window_id=later) == {MINER_B}
    assert quarantined_miners(history, window_id=windows[0] + ERROR_QUARANTINE_WINDOWS + 1) == set()


def test_quarantined_miner_gets_no_slots(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    ledger.record_credits(ledger.open_window(1, "c"), {}, {MINER_A}, set())
    policy = _policy(tmp_path)
    result = run_window(gym=GYM, 
        chain=_served_chain(), validator_hotkey=VALIDATOR,
        miners=[MinerRuntime(hotkey=hotkey, auth=_auth(hotkey), policy=policy,
                             actor=OnPolicyStubActor(hotkey)) for hotkey in (MINER_A, MINER_B)],
        policy=policy, ledger=ledger, families=["two_pool_cycle"],
        group_size=2, tasks_per_window=1, train=False,
    )
    assert result.extras["quarantined_miners"] == [MINER_A]
    assert result.extras["dealt"] == {MINER_B: 2}


def test_failed_miner_loses_measured_throughput():
    from gft_subnet.coordinator import (
        THROUGHPUT_EMA_ALPHA,
        _update_throughput,
    )

    throughput = {MINER_A: 100.0, MINER_B: 100.0}
    _update_throughput(throughput, {MINER_B: 100.0}, {MINER_A})
    assert throughput == {MINER_A: pytest.approx(100.0 * (1 - THROUGHPUT_EMA_ALPHA)),
                          MINER_B: pytest.approx(100.0)}


def test_completion_contract():
    from gft_subnet.verify import completion_violation

    def check(tokens):
        return completion_violation(tokens, eos_token_id=1, max_tokens=4)

    assert check([5, 6, 1]) is None                 # stopped at EOS
    assert check([5, 6, 7, 8]) is None              # hit the token limit
    assert "after end-of-sequence" in check([5, 1, 6])
    assert "limit is 4" in check([5, 6, 7, 8, 9])
    assert "without end-of-sequence" in check([5, 6])
    assert "no generated tokens" in check([])


def test_miner_http_generates_a_batch_in_request_order(tmp_path):
    from gft_subnet.miner import MAX_BATCH

    policy = _policy(tmp_path)
    runtime = MinerRuntime(hotkey=MINER_A, auth=_auth(MINER_A), policy=policy,
                           actor=OnPolicyStubActor(MINER_A), allowed_callers={VALIDATOR})
    ack_miners(policy, [runtime])
    client = TestClient(create_miner_app(runtime))
    validator = _auth(VALIDATOR)
    turns = [TurnRequest(1, f"ep-{i}", 0, "two_pool_cycle", 1, i, {}, []).to_dict() for i in range(3)]
    response = _signed_post(client, validator, "/v1/generate_batch", {"requests": turns})
    assert response.status_code == 200
    assert [item["episode_id"] for item in response.json()["responses"]] == ["ep-0", "ep-1", "ep-2"]
    too_many = {"requests": turns[:1] * (MAX_BATCH + 1)}
    assert _signed_post(client, validator, "/v1/generate_batch", too_many).status_code == 400


def test_group_runs_in_lockstep_with_one_call_per_turn(tmp_path):
    from gft_subnet.gym_loop import RolloutSpec, run_rollouts

    policy = _policy(tmp_path)
    batches = []

    class Finalizer:
        """Rollout 0 finalizes at once; the others inspect first, then finalize."""

        scripted = True

        def __init__(self):
            self.stub = OnPolicyStubActor(MINER_A)

        def generate_batch(self, requests):
            batches.append([request.generation_seed for request in requests])
            out = []
            for r in requests:
                action = ({"type": "finalize"} if r.generation_seed == 0 or r.turn > 0
                          else {"type": "inspect_task"})
                response = self.stub.generate(r)
                response.action, response.text = action, json.dumps(action)
                out.append(response)
            return out

    miner = MinerRuntime(hotkey=MINER_A, auth=_auth(MINER_A), policy=policy, actor=Finalizer())
    ack_miners(policy, [miner])
    specs = [RolloutSpec("two_pool_cycle", 3, 0, "k0"), RolloutSpec("two_pool_cycle", 3, 1, "k1"),
             RolloutSpec("triangular_cycle", 4, 2, "k2")]
    episodes = run_rollouts(gym=GYM, miner=miner, rollouts=specs, policy=policy)
    assert batches == [[0, 1, 2], [1, 2]]
    assert [len(ep.turns) for ep in episodes] == [1, 2, 2]
    assert all(ep.infra_error is None and ep.complete for ep in episodes)
    assert [(ep.family, ep.sampling_key) for ep in episodes] == [
        ("two_pool_cycle", "k0"), ("two_pool_cycle", "k1"), ("triangular_cycle", "k2")]


def test_forced_sampling_tolerance_ignores_boundary_noise_but_bounds_swaps():
    torch = pytest.importorskip("torch")
    from gft_subnet.verify import FORCED_SAMPLING_TOLERANCE, forced_sampling_ok

    cuda = FORCED_SAMPLING_TOLERANCE["cuda"]
    honest = torch.zeros(1000)
    honest[:30] = 0.005          # boundary noise: many tiny off-pick tokens
    honest[30:33] = 0.18         # a few larger shifts, as seen on L40S
    assert forced_sampling_ok(honest, cuda)[0]
    swapped = honest.clone()
    swapped[100:115] = 0.2       # 15 deliberate near-pick swaps
    assert not forced_sampling_ok(swapped, cuda)[0]
    noisy = torch.zeros(1000)
    noisy[:60] = 0.005           # too many off-pick tokens even if all tiny
    assert not forced_sampling_ok(noisy, cuda)[0]
    far = torch.zeros(40)
    far[0] = 0.5                 # one token far from its forced pick
    assert not forced_sampling_ok(far, cuda)[0]
    short = torch.zeros(40)
    short[:3] = 0.05             # a short rollout may still carry a few shifts
    assert forced_sampling_ok(short, cuda)[0]
    assert not forced_sampling_ok(short, FORCED_SAMPLING_TOLERANCE["cpu"])[0]
