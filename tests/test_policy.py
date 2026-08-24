from types import SimpleNamespace

import torch

from tiny_wordle.environment import Observation
from tiny_wordle.policy import TokenTrie, TriePolicy


class FakeTokenizer:
    eos_token_id = 99
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [7]}

    def decode(self, ids, skip_special_tokens=True):
        return "WORD"


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        input_ids,
        attention_mask,
        logits_to_keep=0,
        use_cache=True,
    ):
        assert logits_to_keep == 1
        assert use_cache is False
        batch, length = input_ids.shape
        logits = torch.full(
            (batch, length, 100),
            -10.0,
            dtype=self.scale.dtype,
            device=input_ids.device,
        )
        for row in range(batch):
            last = int(input_ids[row, length - 1])
            if last == 7:
                logits[row, length - 1, 1] = 0.0
                logits[row, length - 1, 2] = 0.0
            else:
                logits[row, length - 1, 99] = 0.0
        return SimpleNamespace(logits=logits[:, -1:, :] * self.scale)


def observation():
    return Observation(
        history=(),
        turn=0,
        remaining_turns=6,
        candidate_count=2,
        state_key="",
        prompt="prompt",
    )


def policy():
    trie = TokenTrie(
        {"APPLE": (1, 99), "SHORE": (2, 99)},
        eos_token_id=99,
    )
    sampler = TriePolicy(
        FakeModel(),
        FakeTokenizer(),
        trie,
        device="cpu",
        prompt_renderer=lambda prompt: prompt,
        checkpoint_digest="checkpoint",
        tokenizer_digest="tokenizer",
        memory_probe=lambda: 1.25,
    )
    return sampler


def test_trie_shape_and_allowed_tokens():
    trie = TokenTrie(
        {"APPLE": (1, 99), "SHORE": (2, 99)},
        eos_token_id=99,
    )
    assert trie.allowed_tokens(()) == (1, 2)
    assert trie.allowed_tokens((1,)) == (99,)
    assert trie.word_for_sequence((2, 99)) == "SHORE"
    assert trie.shape()["branching_nodes"] == 1


def test_exact_distribution_is_normalized():
    values = policy().exact_distribution(observation(), temperature=1.0)
    assert set(values) == {"APPLE", "SHORE"}
    assert abs(sum(values.values()) - 1.0) < 1e-7
    assert abs(values["APPLE"] - values["SHORE"]) < 1e-7


def test_sampled_probability_replays():
    sampler = policy()
    decision = sampler.sample(observation(), temperature=1.0, seed=4)
    total, per_token = sampler.log_probability(
        observation(),
        decision.word,
        temperature=decision.temperature,
    )
    assert total == decision.action_log_probability
    assert per_token == decision.per_token_log_probabilities
    assert decision.token_ids in ((1, 99), (2, 99))
    assert sampler.forward_memory_trace
    assert set(sampler.forward_memory_trace) == {1.25}


def test_same_seed_reproduces_action():
    sampler = policy()
    first = sampler.sample(observation(), temperature=0.7, seed=123)
    second = sampler.sample(observation(), temperature=0.7, seed=123)
    assert first == second


def test_greedy_trie_uses_lowest_token_for_equal_logits():
    assert policy().greedy_word(observation()) == "APPLE"


def test_nonpositive_temperature_rejected():
    sampler = policy()
    try:
        sampler.sample(observation(), temperature=0, seed=1)
    except ValueError as error:
        assert "temperature" in str(error)
    else:
        raise AssertionError("nonpositive temperature was accepted")


def test_exact_distribution_enforces_resource_caps():
    sampler = policy()
    with torch.no_grad():
        try:
            sampler.exact_distribution(
                observation(),
                temperature=1.0,
                max_model_calls=0,
            )
        except RuntimeError as error:
            assert "model calls" in str(error)
        else:
            raise AssertionError("model-call cap was ignored")


class CausalFakeModel(torch.nn.Module):
    """A deterministic causal stand-in that honours arbitrary logits_to_keep.

    Position ``i`` depends only on ``input_ids[:i + 1]``, which is what makes
    the equivalence test meaningful: a single multi-position forward has to
    agree with a sequence of single-position forwards over growing prefixes,
    and that only holds if the fake respects causality the way a real decoder
    does.
    """

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.seen_lengths: list[int] = []
        self.seen_keep: list[int] = []

    def forward(self, input_ids, attention_mask, logits_to_keep=0, use_cache=True):
        assert use_cache is False
        batch, length = input_ids.shape
        self.seen_lengths.append(int(length))
        self.seen_keep.append(int(logits_to_keep))
        vocabulary = torch.arange(100, dtype=self.scale.dtype)
        logits = torch.zeros((batch, length, 100), dtype=self.scale.dtype)
        for row in range(batch):
            running = 0.0
            for position in range(length):
                running += float(input_ids[row, position])
                logits[row, position] = torch.sin(vocabulary * 0.7 + running * 1.3)
        keep = logits_to_keep if logits_to_keep else length
        return SimpleNamespace(logits=logits[:, -keep:, :] * self.scale)


def causal_policy():
    trie = TokenTrie(
        {"APPLE": (1, 3, 99), "AMPLE": (1, 4, 99), "SHORE": (2, 99)},
        eos_token_id=99,
    )
    return TriePolicy(
        CausalFakeModel().eval(),
        FakeTokenizer(),
        trie,
        device="cpu",
        prompt_renderer=lambda prompt: prompt,
        checkpoint_digest="checkpoint",
        tokenizer_digest="tokenizer",
    )


def test_score_action_matches_the_sequential_path_token_for_token():
    """The whole point of the cheap path is that it computes the same thing."""
    for word in ("APPLE", "AMPLE", "SHORE"):
        for temperature in (0.5, 1.0, 1.75):
            sampler = causal_policy()
            expected_total, expected_values = sampler.log_probability_tensor(
                observation(),
                word,
                temperature=temperature,
                requires_grad=False,
            )
            actual_total, actual_values = sampler.score_action(
                observation(),
                word,
                temperature=temperature,
                requires_grad=False,
            )
            assert len(actual_values) == len(expected_values)
            for actual, expected in zip(actual_values, expected_values):
                assert torch.allclose(actual, expected, atol=1e-6), word
            assert torch.allclose(actual_total, expected_total, atol=1e-6), word


def test_score_action_uses_one_forward_where_the_walk_uses_several():
    sampler = causal_policy()
    sampler.log_probability_tensor(
        observation(), "APPLE", temperature=1.0, requires_grad=False
    )
    walk_calls = sampler.forward_call_count
    sampler.score_action(
        observation(), "APPLE", temperature=1.0, requires_grad=False
    )
    assert walk_calls == 2
    assert sampler.forward_call_count - walk_calls == 1


def test_score_action_truncates_after_the_last_branching_position():
    """Trailing deterministic tokens need no logits, so they are not fed in."""
    sampler = causal_policy()
    sampler.score_action(
        observation(), "SHORE", temperature=1.0, requires_grad=False
    )
    assert sampler.model.seen_keep == [1]
    assert sampler.model.seen_lengths == [1]

    sampler = causal_policy()
    sampler.score_action(
        observation(), "APPLE", temperature=1.0, requires_grad=False
    )
    assert sampler.model.seen_keep == [2]
    assert sampler.model.seen_lengths == [2]


def test_score_action_returns_exact_zero_for_singleton_positions():
    sampler = causal_policy()
    _, values = sampler.score_action(
        observation(), "APPLE", temperature=1.0, requires_grad=False
    )
    assert float(values[2]) == 0.0


def test_score_action_carries_gradient_to_the_model():
    sampler = causal_policy()
    total, _ = sampler.score_action(observation(), "APPLE", temperature=1.0)
    total.backward()
    assert sampler.model.scale.grad is not None
    assert float(sampler.model.scale.grad.abs()) > 0.0


def test_score_action_rejects_a_word_outside_the_trie():
    sampler = causal_policy()
    try:
        sampler.score_action(observation(), "CRANE", temperature=1.0)
    except ValueError as error:
        assert "token trie" in str(error)
    else:
        raise AssertionError("an out-of-vocabulary action was scored")


def test_score_action_refuses_to_run_with_dropout_active():
    """Dropout during scoring would break the ratio identity gate."""
    sampler = causal_policy()
    sampler.model.train()
    try:
        sampler.score_action(observation(), "APPLE", temperature=1.0)
    except RuntimeError as error:
        assert "eval mode" in str(error)
    else:
        raise AssertionError("scoring ran with dropout active")


def test_score_action_rejects_nonpositive_temperature():
    sampler = causal_policy()
    try:
        sampler.score_action(observation(), "APPLE", temperature=0.0)
    except ValueError as error:
        assert "temperature" in str(error)
    else:
        raise AssertionError("nonpositive temperature was accepted")


def test_score_action_records_a_memory_probe_sample():
    trie = TokenTrie(
        {"APPLE": (1, 3, 99), "AMPLE": (1, 4, 99), "SHORE": (2, 99)},
        eos_token_id=99,
    )
    sampler = TriePolicy(
        CausalFakeModel().eval(),
        FakeTokenizer(),
        trie,
        device="cpu",
        prompt_renderer=lambda prompt: prompt,
        memory_probe=lambda: 4.5,
    )
    sampler.score_action(
        observation(), "APPLE", temperature=1.0, requires_grad=False
    )
    assert sampler.forward_memory_trace == [4.5]
