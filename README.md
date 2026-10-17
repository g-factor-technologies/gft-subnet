# gft-subnet

Verified on-policy RL on Bittensor. Miners generate rollouts from the current LoRA policy; every sampled token is fixed by public randomness (forced sampling), the validator replays every trace, pays one credit per accepted slot, and takes one central GRPO step.

Protocol: [docs/protocol.md](docs/protocol.md).

## Where to look

| Topic | Code | Tests |
|---|---|---|
| Validator scoring (slot credit, decay, quarantine, weights) | [scoring.py](src/gft_subnet/scoring.py) | `tests/test_subnet.py::test_credit_*`, `test_quarantine_*`, `test_window_weights_follow_accepted_slots_not_reward` |
| Trace replay and tolerances | [verify.py](src/gft_subnet/verify.py): `verify_episode_traces`, `FORCED_SAMPLING_TOLERANCE`, `LOGPROB_TOLERANCE`, `completion_violation` | `tests/test_verification_cpu.py::test_adaptive_attacks_against_cuda_tolerances`, `tests/test_subnet.py::test_completion_contract` |
| Forced sampling | [sampling.py](src/gft_subnet/sampling.py) | `test_batched_forced_sampling_matches_single_rows_and_verifies` |
| Window: tasks, dealing, re-dealing, training | [coordinator.py](src/gft_subnet/coordinator.py) | `tests/test_subnet.py` |
| Validator-side gym loop | [gym_loop.py](src/gft_subnet/gym_loop.py), [gym.py](src/gft_subnet/gym.py) | |
| Miner | [miner.py](src/gft_subnet/miner.py), [actor.py](src/gft_subnet/actor.py) | |
| GRPO step | [training.py](src/gft_subnet/training.py) | `test_two_real_grpo_steps_reload_lora_and_preserve_multiturn_logprobs` |

## Gyms

The validator runs the gym; miners never receive gym code. Scoring and replay do not depend on the gym. A gym subclasses `gft_subnet.gym.Gym` and is selected with `--gym package.module:ClassName`. The package ships one reference gym, `guess_number`; production gyms are private plug-ins.

## Quick start

```sh
uv venv && uv pip install -e ".[train,dev]"
pytest
python -m gft_subnet --state outputs/local/chain.json run-window --work-dir outputs/local --group-size 4 --dummy-weights
```

The local window uses a local chain and two scripted miners. Testnet: install `.[chain]` and see [configs/testnet.example.yaml](configs/testnet.example.yaml).

Licensed under [Apache-2.0](LICENSE). © 2026 G-Factor Technologies.
