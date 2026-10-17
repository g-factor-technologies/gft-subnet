"""Real, offline optimizer/adapter qualification; controlled rewards, not a capability eval."""
import copy
from pathlib import Path

import pytest


def test_two_real_grpo_steps_reload_lora_and_preserve_multiturn_logprobs(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    pytest.importorskip("peft")

    from gft_subnet.actor import HuggingFacePolicyActor
    from gft_subnet.policy import PolicyError, PolicyRegistry
    from gft_subnet.protocol import Episode, EpisodeTurn, TurnRequest
    from gft_subnet.sampling import sampling_key
    from gft_subnet.training import (
        _adapter_fingerprint, make_one_step_trainer, pack_episodes, run_one_grpo_step,
    )
    from gft_subnet.verify import verify_episode_traces

    def _live_model(trainer):
        accelerator = getattr(trainer, "accelerator", None)
        if accelerator is not None:
            return accelerator.unwrap_model(trainer.model)
        return trainer.model

    # No Hub download, wallet, RPC, GPU allocation, or pre-existing model is needed.
    model_dir, vocab = _tiny_model(tmp_path)
    policy = PolicyRegistry(base_model=str(model_dir), root=tmp_path / "policy",
                            sampling={"temperature": 0.7, "max_tokens": 4})
    policy.publish_base()
    actor = HuggingFacePolicyActor(hotkey="miner", model_id=str(model_dir))
    previous_fingerprint = None
    live_model = None
    live_optimizer = None

    for version in (1, 2):
        manifest = policy.manifest()
        actor.load_policy(manifest)
        episodes = []
        for seed in (0, 1):
            turns, prefix = [], []
            key = sampling_key(f"window-{version}", "two_pool_cycle", 1, manifest.checksum, seed)
            for turn in (0, 1):
                request = TurnRequest(version, f"ep-{seed}", turn, "two_pool_cycle", 1, seed,
                                      {"step": turn}, [{"role": "user", "content": "observe"}],
                                      prefix_token_ids=prefix, sampling_key=key)
                response = actor.generate(request)
                # Forced sampling: the same public key always yields the same tokens.
                assert actor.generate(request).token_ids == response.token_ids
                turns.append(EpisodeTurn(
                    turn, response.action, {}, response.token_ids, response.logprobs, [],
                    prompt_ids=response.prompt_ids,
                ))
                prefix = response.prompt_ids + response.token_ids
            episodes.append(Episode(f"ep-{seed}", "miner", "two_pool_cycle", 1, seed,
                                    version, turns, float(seed == 0), True, sampling_key=key))
        packed = pack_episodes(episodes, group_size=2)
        trainer = make_one_step_trainer(
            model_id=str(model_dir), output_dir=tmp_path / f"trainer-{version}",
            episodes=episodes, group_size=2, adapter_dir=manifest.adapter_dir, use_cpu=True,
            temperature=manifest.sampling["temperature"], model=live_model,
            optimizer=live_optimizer,
        )
        assert trainer.model.peft_config
        stats = verify_episode_traces(trainer.model, episodes, temperature=manifest.sampling["temperature"])
        assert stats["rejected"] == 0 and stats["checked"] == 2
        assert all(episode.infra_error is None for episode in episodes)
        forged = copy.deepcopy(episodes[0])
        last = forged.turns[-1]
        last.token_ids[-1] = (last.token_ids[-1] + 1) % len(vocab)
        verify_episode_traces(trainer.model, [forged], temperature=manifest.sampling["temperature"])
        assert forged.infra_error and "forced samples" in forged.infra_error
        assert all("lora_" in name for name, p in trainer.model.named_parameters() if p.requires_grad)
        if previous_fingerprint is not None:
            assert _adapter_fingerprint(trainer.model) == previous_fingerprint
        # Recompute the probability of every generated token under the exact packed context.
        trainer.model.eval()
        for row in range(packed["n"]):
            prompt, completion = packed["prompt_ids"][row], packed["completion_ids"][row]
            ids = torch.tensor([prompt + completion])
            with torch.no_grad():
                logits = trainer.model(input_ids=ids).logits[0, len(prompt)-1:-1].float()
                logps = (logits / manifest.sampling["temperature"]).log_softmax(-1)
                logps = logps.gather(1, torch.tensor(completion)[:, None]).squeeze(1)
            mask = torch.tensor(packed["env_mask"][row], dtype=torch.bool)
            assert (~mask).any()  # feedback is present but is not trained as model output
            assert torch.allclose(logps[mask], torch.tensor(packed["logprobs"][row])[mask], atol=1e-5)
        if version == 1:
            episodes[0].turns[0].logprobs[0] += 1.0
            with pytest.raises(PolicyError, match="behavior logprobs"):
                run_one_grpo_step(episodes, adapter_dir=manifest.adapter_dir,
                                  group_size=2, trainer=trainer)
            episodes[0].turns[0].logprobs[0] -= 1.0
        trained = run_one_grpo_step(episodes, adapter_dir=manifest.adapter_dir,
                                    group_size=2, trainer=trainer)
        assert trainer.state.global_step == 1
        live_model = _live_model(trainer)
        live_optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
        # Adam moments carry over: the second window is Adam step 2, not a fresh sign step.
        steps = {int(state["step"]) for state in live_optimizer.state.values()}
        assert steps == {version}
        previous_fingerprint = _adapter_fingerprint(live_model)
        saved = Path(trained["adapter_dir"])
        assert (saved / "adapter_config.json").is_file()
        assert (saved / "adapter_model.safetensors").is_file()
        assert not (saved / "model.safetensors").exists()
        next_manifest = policy.publish_from_dir(version=version + 1, adapter_dir=saved)
        actor.load_policy(next_manifest)
        assert actor._loaded_checksum == next_manifest.checksum
        assert _adapter_fingerprint(actor._model) == previous_fingerprint
        policy = PolicyRegistry(base_model=str(model_dir), root=policy.root)
        assert policy.current_version == version + 1


def test_chunked_completion_logps_match_dense_softmax(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    from gft_subnet.training import chunked_completion_logps

    torch.manual_seed(3)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=0, eos_token_id=1, bos_token_id=None,
    )).eval()
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    mask = torch.ones_like(ids)
    keep = 4
    temperature = 0.7
    with torch.no_grad():
        dense = (model(input_ids=ids).logits[:, :-1][:, -keep:] / temperature).log_softmax(-1)
        dense = dense.gather(2, ids[:, -keep:].unsqueeze(-1)).squeeze(-1)
        chunked, entropy, aux = chunked_completion_logps(
            model, ids, mask, keep, temperature=temperature, chunk_size=2)
    assert entropy is None and aux is None
    assert torch.allclose(chunked, dense, atol=1e-5)


def _tiny_model(tmp_path):
    """Tiny local Llama + word tokenizer with a chat template; returns (dir, vocab)."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.set_num_threads(1)
    model_dir = tmp_path / "tiny-llama"
    vocab = {word: index for index, word in enumerate(
        ["<pad>", "<eos>", "<unk>", "user", "assistant", "observe", "step"]
        + [f"token{i}" for i in range(25)])}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                        eos_token="<eos>", unk_token="<unk>")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}"
        "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    )
    torch.manual_seed(7)
    LlamaForCausalLM(LlamaConfig(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=2048,
        pad_token_id=0, eos_token_id=1, bos_token_id=None,
    )).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return model_dir, vocab


def test_batched_forced_sampling_matches_single_rows_and_verifies(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from gft_subnet.actor import HuggingFacePolicyActor
    from gft_subnet.policy import PolicyRegistry
    from gft_subnet.protocol import Episode, EpisodeTurn, TurnRequest
    from gft_subnet.sampling import sampling_key
    from gft_subnet.training import load_policy_model
    from gft_subnet.verify import verify_episode_traces

    model_dir, _ = _tiny_model(tmp_path)
    policy = PolicyRegistry(base_model=str(model_dir), root=tmp_path / "policy",
                            sampling={"temperature": 0.9, "max_tokens": 6})
    manifest = policy.publish_base()
    actor = HuggingFacePolicyActor(hotkey="miner", model_id=str(model_dir), max_batch=3)
    actor.load_policy(manifest)
    keys = [sampling_key("w", "two_pool_cycle", 1, manifest.checksum, i) for i in range(4)]
    first = [TurnRequest(1, f"ep-{i}", 0, "two_pool_cycle", 1, i, {},
                         [{"role": "user", "content": "observe " * (i + 1)}], sampling_key=keys[i])
             for i in range(4)]
    batched = actor.generate_batch(first)  # 4 rows split into GPU calls of 3 + 1
    single = [actor.generate(request) for request in first]
    assert [r.token_ids for r in batched] == [r.token_ids for r in single]
    for got, want in zip(batched, single, strict=True):
        assert torch.allclose(torch.tensor(got.logprobs), torch.tensor(want.logprobs), atol=1e-5)
    # Second turn: rows now carry different-length prefixes (left padding in one call).
    second = [TurnRequest(1, f"ep-{i}", 1, "two_pool_cycle", 1, i, {},
                          [{"role": "user", "content": "step"}],
                          prefix_token_ids=r.prompt_ids + r.token_ids, sampling_key=keys[i])
              for i, r in enumerate(batched)]
    replies = actor.generate_batch(second)
    episodes = [
        Episode(f"ep-{i}", "miner", "two_pool_cycle", 1, i, 1, [
            EpisodeTurn(0, {}, {}, a.token_ids, a.logprobs, [], prompt_ids=a.prompt_ids),
            EpisodeTurn(1, {}, {}, b.token_ids, b.logprobs, [], prompt_ids=b.prompt_ids),
        ], 0.0, True, sampling_key=keys[i])
        for i, (a, b) in enumerate(zip(batched, replies, strict=True))
    ]
    model = load_policy_model(model_id=str(model_dir), adapter_dir=manifest.adapter_dir, use_cpu=True)
    stats = verify_episode_traces(model, episodes, temperature=0.9)
    assert stats["checked"] == 4 and stats["rejected"] == 0


def test_adaptive_attacks_against_cuda_tolerances(tmp_path):
    """A miner that regenerates after every deliberate swap, checked with the CUDA tolerances.

    Stretching, cutting or misplacing the end of a turn is rejected outright.
    Steering a few tokens inside the flip budget still passes: that residual
    lever is bounded by ``FORCED_SAMPLING_TOLERANCE`` and documented here.
    """
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM

    from gft_subnet.protocol import Episode, EpisodeTurn
    from gft_subnet.sampling import forced_pick, forced_uniform, sampling_key
    from gft_subnet.verify import (
        FORCED_SAMPLING_TOLERANCE,
        LOGPROB_TOLERANCE,
        verify_episode_traces,
    )

    model_dir, _ = _tiny_model(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(model_dir).eval()
    temperature, eos, limit, prompt = 0.9, 1, 12, [3, 5, 6, 4]
    cuda = FORCED_SAMPLING_TOLERANCE["cuda"]
    keys = [sampling_key("window", "two_pool_cycle", 1, "checksum", i) for i in range(400)]

    def distribution(ids):
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
        probs = (logits.float() / temperature).softmax(-1)
        return logits, probs, probs.cumsum(-1)

    def reachable(probs, cdf, uniform, pick):
        """Tokens other than the forced pick whose CDF interval is within max_distance."""
        target = uniform * float(cdf[-1])
        hi, lo = cdf, cdf - probs
        distance = (lo - target).clamp(min=0) + (target - hi).clamp(min=0)
        mask = (distance <= cuda["max_distance"]) & (distance > 0)
        mask[pick] = False
        mask[eos] = False
        return [int(t) for t in torch.argsort(probs, descending=True) if mask[t]]

    def generate(key, *, n=limit, stop_at_eos=True, choose=None):
        ids, tokens, logprobs = list(prompt), [], []
        for position in range(n):
            logits, probs, cdf = distribution(ids)
            uniform = forced_uniform(key, 0, position)
            token, _ = forced_pick(logits, uniform, temperature)
            if choose is not None:
                token = choose(position, probs, cdf, uniform, token)
            tokens.append(token)
            logprobs.append(float(probs[token].log()))
            ids.append(token)
            if stop_at_eos and token == eos:
                break
        return tokens, logprobs

    def verdict(key, tokens, logprobs):
        episode = Episode("ep", "miner", "two_pool_cycle", 1, 0, 1, [EpisodeTurn(
            0, {}, {}, tokens, logprobs, [], prompt_ids=list(prompt))], 0.0, True, sampling_key=key)
        verify_episode_traces(model, [episode], temperature=temperature, eos_token_id=eos,
                              max_tokens=limit, tolerance=cuda,
                              logprob_tolerance=LOGPROB_TOLERANCE["cuda"])
        return episode.infra_error

    def first_key(predicate):
        return next(key for key in keys if predicate(generate(key)[0]))

    # Honest traces pass, whether they stop at EOS or at the limit.
    stops = first_key(lambda t: t[-1] == eos and len(t) < limit - 2)
    runs = first_key(lambda t: len(t) == limit and eos not in t)
    assert verdict(stops, *generate(stops)) is None
    assert verdict(runs, *generate(runs)) is None

    # Stretching past the forced EOS (swap it for a reachable token, keep going).
    def no_eos(position, probs, cdf, uniform, token):
        options = reachable(probs, cdf, uniform, token) if token == eos else [token]
        return options[0] if options else token
    stretched = generate(stops, stop_at_eos=False, choose=no_eos)
    assert verdict(stops, *stretched).startswith("TraceWarning")
    assert "eos_mismatch=1" in verdict(stops, *stretched)

    # Going beyond the limit, or cutting early without EOS, breaks the completion contract.
    longer = next(key for key in keys
                  if eos not in (t := generate(key, n=limit + 3)[0]) and len(t) == limit + 3)
    assert "limit is 12" in verdict(longer, *generate(longer, n=limit + 3))
    tokens, logprobs = generate(runs)
    assert "without end-of-sequence" in verdict(runs, tokens[:-4], logprobs[:-4])

    # Ending early with a non-forced EOS is rejected as well.
    def early_eos(position, probs, cdf, uniform, token):
        return eos if position == 3 and token != eos else token
    assert verdict(runs, *generate(runs, choose=early_eos)).startswith("TraceWarning")

    # Residual lever: a miner may steer up to the far-flip budget, never beyond it.
    budget = cuda["far_flips"][0]

    def steered(key, swaps):
        made = []

        def steer(position, probs, cdf, uniform, token):
            options = reachable(probs, cdf, uniform, token) if token != eos else []
            if len(made) < swaps and options:
                made.append(position)
                return options[0]
            return token
        tokens, logprobs = generate(key, choose=steer)
        return (tokens, logprobs) if len(made) == swaps else None

    within = next((key, trace) for key in keys if (trace := steered(key, budget)))
    beyond = next((key, trace) for key in keys if (trace := steered(key, budget + 1)))
    assert verdict(within[0], *within[1]) is None
    assert verdict(beyond[0], *beyond[1]).startswith("TraceWarning")
