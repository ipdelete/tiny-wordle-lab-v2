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
    return TriePolicy(
        FakeModel(),
        FakeTokenizer(),
        trie,
        device="cpu",
        prompt_renderer=lambda prompt: prompt,
        checkpoint_digest="checkpoint",
        tokenizer_digest="tokenizer",
    )


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
