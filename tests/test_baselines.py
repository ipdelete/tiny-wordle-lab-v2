from __future__ import annotations

import numpy as np

from tiny_wordle_lab_v2.baselines import EntropyPolicy, FrequencyPolicy, RandomPolicy
from tiny_wordle_lab_v2.baselines.entropy import build_feedback_matrix
from tiny_wordle_lab_v2.evaluate import EvaluationConfig, evaluate
from tiny_wordle_lab_v2.game import Observation, score_guess
from tiny_wordle_lab_v2.lexicon import Lexicon, LexiconEntry


def observation(candidates: tuple[str, ...]) -> Observation:
    return Observation(
        history=(),
        previous_actions=(),
        remaining_opportunities=6,
        candidates=candidates,
    )


def encode(feedback: str) -> int:
    value = 0
    for mark in feedback:
        value = value * 3 + {"B": 0, "Y": 1, "G": 2}[mark]
    return value


def test_feedback_matrix_matches_scalar_scoring(toy_lexicon: Lexicon) -> None:
    matrix = build_feedback_matrix(
        toy_lexicon.legal_guesses,
        toy_lexicon.answers,
    )
    expected = np.asarray(
        [
            [encode(score_guess(answer, guess)) for answer in toy_lexicon.answers]
            for guess in toy_lexicon.legal_guesses
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(matrix, expected)


def test_frequency_uses_frequency_then_alphabetical_order(
    toy_lexicon: Lexicon,
) -> None:
    policy = FrequencyPolicy(toy_lexicon)
    assert policy.choose(observation(toy_lexicon.answers)) == "apple"


def test_random_policy_is_state_deterministic(toy_lexicon: Lexicon) -> None:
    policy = RandomPolicy(toy_lexicon, seed=42)
    state = observation(toy_lexicon.answers)
    assert policy.choose(state) == policy.choose(state)


def test_random_gameplay_does_not_depend_on_evaluation_order(
    toy_lexicon: Lexicon,
) -> None:
    first = evaluate(
        RandomPolicy(toy_lexicon, seed=42),
        toy_lexicon,
        EvaluationConfig("first", answers=("apple", "banal")),
    )
    second = evaluate(
        RandomPolicy(toy_lexicon, seed=42),
        toy_lexicon,
        EvaluationConfig("second", answers=("banal", "apple")),
    )
    first_actions = {
        game.answer: game.actions for game in first.games
    }
    second_actions = {
        game.answer: game.actions for game in second.games
    }
    assert first_actions == second_actions


def test_candidate_entropy_closes_singleton(toy_lexicon: Lexicon) -> None:
    policy = EntropyPolicy(toy_lexicon, candidate_only=True)
    assert policy.choose(observation(("banal",))) == "banal"


def test_entropy_tie_prefers_candidate_then_alphabetical() -> None:
    answers = ("aaaaa", "bbbbb")
    legal = (*answers, "ccccc")
    lexicon = Lexicon(
        answers=answers,
        legal_guesses=legal,
        entries={
            word: LexiconEntry(word, word in answers, 1.0, ("noun",))
            for word in legal
        },
        source_hashes={},
    )
    matrix = np.asarray([[242, 0], [0, 242], [0, 0]], dtype=np.uint8)
    policy = EntropyPolicy(lexicon, candidate_only=False, _matrix=matrix)
    assert policy.choose(observation(answers)) == "aaaaa"
