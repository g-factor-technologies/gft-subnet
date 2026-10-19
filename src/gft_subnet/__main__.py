"""CLI for local smoke and Bittensor testnet.

python -m gft_subnet --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gft_subnet.auth import BittensorAuth, LocalAuth
from gft_subnet.chain import BittensorChain, LocalChain
from gft_subnet.coordinator import run_window, run_windows
from gft_subnet.gym import load_gym
from gft_subnet.ledger import Ledger
from gft_subnet.miner import (
    MinerRuntime,
    OnPolicyStubActor,
    ScriptedActor,
    create_miner_app,
    miners_from_metagraph,
)
from gft_subnet.policy import PolicyRegistry
from gft_subnet.validator import ValidatorRuntime, create_validator_app


def _keys(hotkeys: list[str]) -> dict[str, bytes]:
    import hashlib

    return {hotkey: hashlib.sha256(hotkey.encode()).digest() for hotkey in hotkeys}


def _load_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    import yaml

    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def _apply_config(args: argparse.Namespace, config: dict[str, Any]) -> None:
    # None means omitted; an explicitly supplied default value still wins over YAML.
    defaults = {
        "netuid": ("netuid", 1), "network": ("network", "test"),
        "group_size": ("group_size", 8), "turn_timeout": ("turn_timeout_seconds", 30.0),
        "tasks_per_window": ("tasks_per_window", 2),
        "model": ("base_model", None), "version_key": ("version_key", 0),
        "device": ("device", "cpu"), "gym": ("gym", "dex_arbitrage"),
        "confidential_compute": ("confidential_compute", False),
    }
    for name, (key, fallback) in defaults.items():
        if hasattr(args, name) and getattr(args, name) is None:
            setattr(args, name, config.get(key, fallback))
    args.sampling = dict(config.get("sampling") or {"temperature": 0.7, "max_tokens": 512})
    args.gym_plugin = load_gym(args.gym)
    if args.gym_id is None:
        args.gym_id = config.get("gym_id") or args.gym_plugin.gym_id
    if args.cmd == "run-window":
        args.families = [args.family] if args.family is not None else list(
            config.get("families") or args.gym_plugin.families)
        if getattr(args, "steps", None) is None:
            args.steps = int(config.get("train_steps") or 1)


def _identity(args: argparse.Namespace) -> tuple[Any, str]:
    if args.wallet:
        from gft_subnet.wallets import hotkey_ss58, load_wallet

        wallet = load_wallet(args.wallet, args.wallet_hotkey)
        return wallet, hotkey_ss58(wallet)
    return None, args.hotkey


def _chain(args: argparse.Namespace, wallet: Any = None):
    if args.chain == "local":
        return LocalChain(netuid=args.netuid, path=args.state)
    return BittensorChain(netuid=args.netuid, network=args.network, wallet=wallet)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gft_subnet")
    parser.add_argument("--config", default="")
    parser.add_argument("--chain", choices=("local", "test"), default="local")
    parser.add_argument("--network", default=None)
    parser.add_argument("--netuid", type=int, default=None)
    parser.add_argument("--hotkey", default="validator")
    parser.add_argument("--gym", default=None,
                        help="'dex_arbitrage' (built-in test gym) or 'package.module:GymClass'")
    parser.add_argument("--gym-id", default=None, dest="gym_id")
    parser.add_argument(
        "--confidential-compute", default=None, action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--wallet", default="")
    parser.add_argument("--wallet-hotkey", default="default")
    parser.add_argument("--state", default="outputs/subnet/chain.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("wallets")
    sub.add_parser("register")
    sub.add_parser("burned-register")
    sub.add_parser("create-subnet")
    sub.add_parser("start-subnet")
    axon = sub.add_parser("serve-axon")
    axon.add_argument("--ip", default="127.0.0.1")
    axon.add_argument("--port", type=int, required=True)
    weights = sub.add_parser("set-weights")
    weights.add_argument("--dummy", action="store_true")
    weights.add_argument("--uids", default="")
    weights.add_argument("--weights", default="")
    weights.add_argument("--version-key", type=int, default=None)
    sub.add_parser("metagraph")

    miner = sub.add_parser("serve-miner")
    miner.add_argument("--port", type=int, default=8091)
    miner.add_argument("--kind", choices=("gold", "dummy", "on_policy", "model"), default="dummy")
    miner.add_argument("--host", default="127.0.0.1")
    miner.add_argument("--model", default=None)
    miner.add_argument("--device", default=None)
    miner.add_argument("--validator-hotkey", action="append", default=[])
    miner.add_argument("--work-dir", default="outputs/subnet")
    miner.add_argument("--announce", action="store_true")
    miner.add_argument("--ip", default="127.0.0.1")
    miner.add_argument("--max-batch", type=int, default=8,
                       help="rollouts per GPU generate call; lower it if a group OOMs")

    validator = sub.add_parser("serve-validator")
    validator.add_argument("--port", type=int, default=8090)
    validator.add_argument("--host", default="127.0.0.1")
    validator.add_argument("--work-dir", default="outputs/subnet")
    validator.add_argument("--announce", action="store_true")
    validator.add_argument("--ip", default="127.0.0.1")
    validator.add_argument("--miner-hotkey", action="append", default=[])

    window = sub.add_parser("run-window")
    window.add_argument("--work-dir", default="outputs/subnet")
    window.add_argument("--family", default=None)
    window.add_argument("--group-size", type=int, default=None)
    window.add_argument("--tasks-per-window", type=int, default=None)
    window.add_argument("--dummy-weights", action="store_true")
    window.add_argument("--train", action="store_true")
    window.add_argument("--discover-miners", action="store_true")
    window.add_argument("--model", default=None)
    window.add_argument("--turn-timeout", type=float, default=None)
    window.add_argument("--version-key", type=int, default=None)
    window.add_argument("--miner-hotkey", action="append", default=[])
    window.add_argument("--steps", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    config = _load_yaml(args.config or None)
    _apply_config(args, config)
    if args.chain == "test" and args.network != "test":
        parser.error("this MVP only supports the Bittensor test network")
    if args.chain == "test" and args.cmd not in {"wallets", "metagraph"} and not args.wallet:
        parser.error("testnet commands require --wallet; local HMAC auth is not allowed")
    if args.cmd == "wallets":
        return _print_wallets(args)
    wallet, hotkey = _identity(args)
    args.hotkey = hotkey
    chain = _chain(args, wallet)
    if args.cmd == "register":
        uid = chain.register(hotkey)
        print(json.dumps({"hotkey": hotkey, "uid": uid}))
        return 0
    if args.cmd == "burned-register":
        if args.chain == "local":
            uid = chain.register(hotkey)
        else:
            uid = chain.burned_register()
        print(json.dumps({
            "hotkey": hotkey, "uid": uid,
            "explorer": getattr(chain, "explorer_url", lambda: None)(),
        }))
        return 0
    if args.cmd == "create-subnet":
        netuid = chain.create_subnet()
        print(json.dumps({"netuid": netuid, "explorer": getattr(chain, "explorer_url", lambda: None)()}))
        return 0
    if args.cmd == "start-subnet":
        chain.start_subnet()
        print(json.dumps({
            "ok": True, "netuid": args.netuid,
            "explorer": getattr(chain, "explorer_url", lambda: None)(),
        }))
        return 0
    if args.cmd == "serve-axon":
        axon = chain.serve_axon(hotkey, args.ip, args.port)
        print(json.dumps({
            "hotkey": axon.hotkey, "ip": axon.ip, "port": axon.port,
            "explorer": getattr(chain, "explorer_url", lambda: None)(),
        }))
        return 0
    if args.cmd == "set-weights":
        snapshot = chain.metagraph()
        if args.dummy:
            miners = [neuron for neuron in snapshot.neurons if neuron.axon is not None]
            uids = [neuron.uid for neuron in miners]
            weights = [1.0 / len(uids) for _ in uids] if uids else []
        else:
            uids = [int(part) for part in args.uids.split(",") if part]
            weights = [float(part) for part in args.weights.split(",") if part]
        block = chain.set_weights(hotkey, uids, weights, version_key=args.version_key)
        print(json.dumps({
            "block": block, "uids": uids, "weights": weights, "dummy": bool(args.dummy),
            "explorer": getattr(chain, "explorer_url", lambda: None)(),
        }))
        return 0
    if args.cmd == "metagraph":
        snapshot = chain.metagraph()
        print(json.dumps({
            "netuid": snapshot.netuid,
            "block": snapshot.block,
            "neurons": [
                {
                    "uid": neuron.uid,
                    "hotkey": neuron.hotkey,
                    "validator_permit": neuron.validator_permit,
                    "axon": None if neuron.axon is None else {"ip": neuron.axon.ip, "port": neuron.axon.port},
                }
                for neuron in snapshot.neurons
            ],
        }))
        return 0
    if args.cmd == "serve-miner":
        return _serve_miner(args, chain, wallet, hotkey)
    if args.cmd == "serve-validator":
        return _serve_validator(args, chain, wallet, hotkey)
    if args.cmd == "run-window":
        return _run_window(args, chain, wallet, hotkey)
    raise ValueError(f"unknown command {args.cmd}")


def _print_wallets(args: argparse.Namespace) -> int:
    from gft_subnet.wallets import coldkey_ss58, hotkey_ss58, load_wallet

    names = [args.wallet] if args.wallet else ["owner", "miner-a", "miner-b"]
    rows = []
    for name in names:
        wallet = load_wallet(name, args.wallet_hotkey)
        rows.append({
            "wallet": name,
            "hotkey_name": args.wallet_hotkey,
            "coldkey": coldkey_ss58(wallet),
            "hotkey": hotkey_ss58(wallet),
        })
    print(json.dumps({
        "network": args.network,
        "wallets": rows,
        "discord": "Post the owner coldkey in Requests for Testnet TAO. Never post the mnemonic.",
    }, indent=2))
    return 0


def _auth(args: argparse.Namespace, wallet: Any, hotkey: str, peers: list[str]):
    if wallet is not None:
        return BittensorAuth(wallet, self_hotkey=hotkey)
    return LocalAuth(_keys([hotkey, *peers]), self_hotkey=hotkey)


def _maybe_announce(args: argparse.Namespace, chain: Any, hotkey: str, port: int) -> None:
    if not getattr(args, "announce", False):
        return
    axon = chain.serve_axon(hotkey, args.ip, port)
    print(json.dumps({
        "announced": True, "hotkey": axon.hotkey, "ip": axon.ip, "port": axon.port,
        "explorer": getattr(chain, "explorer_url", lambda: None)(),
    }), flush=True)


def _serve_miner(args: argparse.Namespace, chain: Any, wallet: Any, hotkey: str) -> int:
    import uvicorn

    from gft_subnet.actor import HuggingFacePolicyActor

    work = Path(args.work_dir)
    allowed = set(args.validator_hotkey)
    if args.chain == "test" and not allowed:
        raise SystemExit("serve-miner requires --validator-hotkey on testnet")
    if args.chain == "local" and not allowed:
        allowed = {"validator"}
    auth = _auth(args, wallet, hotkey, list(allowed))
    model_id = args.model or "Qwen/Qwen2.5-0.5B-Instruct"
    policy = PolicyRegistry(
        base_model=model_id, root=work / "policy", sampling=args.sampling,
        gym_id=args.gym_id,
    )
    if policy.current_version == 0:
        policy.publish_base()
    if args.kind == "model":
        actor = HuggingFacePolicyActor(
            hotkey=hotkey,
            model_id=model_id,
            adapter_dir=policy.manifest().adapter_dir,
            device=args.device,
            max_batch=args.max_batch,
        )
    elif args.kind == "on_policy":
        actor = OnPolicyStubActor(hotkey)
    else:
        actor = ScriptedActor(hotkey, kind=args.kind, solver=args.gym_plugin.scripted_actions)
    runtime = MinerRuntime(
        hotkey=hotkey, auth=auth, policy=policy, actor=actor, allowed_callers=allowed,
    )
    manifest = policy.manifest()
    runtime.ack_policy(manifest.version, manifest.checksum)
    if args.kind == "model":
        actor.load_policy(manifest)
    _maybe_announce(args, chain, hotkey, args.port)
    uvicorn.run(create_miner_app(runtime), host=args.host, port=args.port)
    return 0


def _serve_validator(args: argparse.Namespace, chain: Any, wallet: Any, hotkey: str) -> int:
    import uvicorn

    work = Path(args.work_dir)
    auth = _auth(args, wallet, hotkey, ["miner-a", "miner-b"])
    policy = PolicyRegistry(
        base_model="stub", root=work / "policy", gym_id=args.gym_id,
    )
    if policy.current_version == 0:
        policy.publish(version=1, adapter_bytes=b"v1")
    ledger = Ledger(work / "ledger.sqlite")
    runtime = ValidatorRuntime(hotkey=hotkey, auth=auth, chain=chain, policy=policy,
                               ledger=ledger, allowed_callers=set(args.miner_hotkey))
    _maybe_announce(args, chain, hotkey, args.port)
    uvicorn.run(create_validator_app(runtime), host=args.host, port=args.port)
    return 0


def _run_window(args: argparse.Namespace, chain: Any, wallet: Any, hotkey: str) -> int:
    work = Path(args.work_dir)
    model_id = args.model or None
    policy = PolicyRegistry(
        base_model=model_id or "stub", root=work / "policy", sampling=args.sampling,
        gym_id=args.gym_id,
    )
    if policy.current_version == 0:
        policy.publish_base()
    else:
        policy.apply_sampling(args.sampling)
    ledger = Ledger(work / "ledger.sqlite")
    auth = _auth(args, wallet, hotkey, ["miner-a", "miner-b"])
    verifier = None
    if args.discover_miners or args.chain == "test":
        if args.train and not args.miner_hotkey:
            raise SystemExit("training requires --miner-hotkey for each controlled miner")
        miners = miners_from_metagraph(
            chain.metagraph(),
            auth=auth,
            skip_hotkeys={hotkey},
            timeout=args.turn_timeout,
        )
        if args.miner_hotkey:
            miners = [miner for miner in miners if miner.hotkey in set(args.miner_hotkey)]
        if len(miners) < 2:
            raise SystemExit("need two miners with serve-axon in the metagraph")
        validator = hotkey
        if model_id:
            from gft_subnet.actor import TokenTraceVerifier

            verifier = TokenTraceVerifier.from_pretrained(model_id)
    else:
        validator = "validator"
        for name in [validator, "miner-a", "miner-b"]:
            chain.register(name)
        chain.serve_axon("miner-a", "127.0.0.1", 8091)
        chain.serve_axon("miner-b", "127.0.0.1", 8092)
        keys = _keys([validator, "miner-a", "miner-b"])
        miners = [
            MinerRuntime(
                hotkey="miner-a",
                auth=LocalAuth(keys, self_hotkey="miner-a"),
                policy=policy,
                actor=ScriptedActor("miner-a", kind="gold",
                                    solver=args.gym_plugin.scripted_actions),
            ),
            MinerRuntime(
                hotkey="miner-b",
                auth=LocalAuth(keys, self_hotkey="miner-b"),
                policy=policy,
                actor=ScriptedActor("miner-b", kind="dummy"),
            ),
        ]
    kwargs = {
        "gym": args.gym_plugin,
        "chain": chain,
        "validator_hotkey": validator,
        "miners": miners,
        "policy": policy,
        "ledger": ledger,
        "families": list(args.families),
        "group_size": args.group_size,
        "tasks_per_window": args.tasks_per_window,
        "dummy_weights": args.dummy_weights,
        "train": args.train,
        "turn_timeout_seconds": args.turn_timeout,
        "version_key": args.version_key,
        "model_id": model_id if args.train else None,
        "trainer_output_dir": str(work / "trainer"),
        "require_confidential": bool(args.confidential_compute),
        "verifier": verifier,
    }
    steps = int(getattr(args, "steps", 1) or 1)
    if steps == 1:
        print(json.dumps(_window_json(run_window(**kwargs))))
        return 0
    results = run_windows(steps=steps, log_path=work / "windows.jsonl", **kwargs)
    print(json.dumps({
        "steps": steps,
        "trained": sum(1 for item in results if item.next_policy is not None),
        "skipped": sum(1 for item in results if item.extras.get("training_skipped")),
        "final_policy_version": results[-1].policy_version,
        "final_next_version": (
            None if results[-1].next_policy is None else results[-1].next_policy.version
        ),
        "windows": [_window_json(item) for item in results],
    }))
    return 0


def _window_json(result: Any) -> dict[str, Any]:
    return {
        "window_id": result.window_id,
        "policy_version": result.policy_version,
        "scores": result.scores,
        "uids": result.uids,
        "weights": result.weights,
        "block": result.block,
        "dummy_weights": result.dummy_weights,
        "next_version": None if result.next_policy is None else result.next_policy.version,
        "explorer": result.extras.get("explorer_url"),
        "weights_skipped": result.extras.get("weights_skipped"),
        "training_skipped": result.extras.get("training_skipped"),
        **{key: result.extras.get(key) for key in (
            "tasks", "dealt", "failed_miners", "missing_rollouts", "credits", "trace")},
    }


if __name__ == "__main__":
    raise SystemExit(main())
