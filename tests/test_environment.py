import json
from pathlib import Path

import numpy as np
import pytest

from tiny_wordle import benchmark
from tiny_wordle.benchmark import parse_guess
from tiny_wordle.environment import (
    EnvironmentConfig,
    WordleEnvironment,
    replay_trace,
    serialize_trace,
)
from tiny_wordle.expert import EntropyExpert, encode_feedback
from tiny_wordle.game import score_string

ROOT = Path(__file__).parents[1]
ANSWERS = (
    "RAISE",
    "SHORE",
    "CRANE",
    "APPLE",
    "ALLEY",
    "BANAL",
    "SHEEP",
)


def pattern_matrix(words=ANSWERS):
    return np.asarray(
        [
            [
                encode_feedback(score_string(answer, guess))
                for answer in words
            ]
            for guess in words
        ],
        dtype=np.int16,
    )


@pytest.fixture
def config():
    return EnvironmentConfig(answers=ANSWERS, opening="RAISE")


@pytest.fixture
def environment(config):
    patterns = pattern_matrix()
    return WordleEnvironment(
        config,
        expert=EntropyExpert(list(ANSWERS), patterns),
        patterns=patterns,
    )


def test_reset_records_fixed_opening(environment):
    start = environment.reset("SHORE", episode_id="episode-1", seed=7)
    assert start.done is False
    assert start.reward == 0
    assert start.observation.turn == 1
    assert start.observation.remaining_turns == 5
    assert start.opening_record.action == "RAISE"
    assert start.opening_record.feedback == score_string("SHORE", "RAISE")
    assert environment.trace.records == (start.opening_record,)


def test_reset_solves_when_answer_is_the_opening(environment):
    start = environment.reset("RAISE")
    assert start.done is True
    assert start.reward == 1
    assert start.opening_record.terminal_reason == "solved"
    assert start.observation.remaining_turns == 5
    with pytest.raises(RuntimeError, match="terminated"):
        environment.step("SHORE")


def test_valid_step_updates_history_and_candidates(environment):
    start = environment.reset("SHORE")
    observation, reward, done, info = environment.step("CRANE")
    record = environment.last_record
    assert reward == 0
    assert done is False
    assert info["repeated"] is False
    assert record.candidate_count_before == start.observation.candidate_count
    assert record.candidate_count_after <= record.candidate_count_before
    assert observation.history[-1].guess == "CRANE"
    assert observation.history[-1].feedback == score_string("SHORE", "CRANE")


def test_repeated_guess_consumes_a_turn(environment):
    environment.reset("SHORE")
    environment.step("CRANE")
    observation, _, done, info = environment.step("CRANE")
    record = environment.last_record
    assert done is False
    assert info["repeated"] is True
    assert record.turn == 3
    assert len(observation.history) == 3


@pytest.mark.parametrize("action", ["crane", "CRAN", "A1AAA", "MIGHT", 4, None])
def test_invalid_action_terminates_without_scoring(environment, action):
    environment.reset("SHORE")
    observation, reward, done, info = environment.step(action)
    record = environment.last_record
    assert done is True
    assert reward == 0
    assert info["terminal_reason"] == "contract_violation"
    assert record.feedback is None
    assert observation.turn == 1
    with pytest.raises(RuntimeError, match="terminated"):
        environment.step("CRANE")


def test_exhaustion_without_opening():
    words = ("SHORE", "CRANE", "APPLE")
    patterns = pattern_matrix(words)
    env = WordleEnvironment(
        EnvironmentConfig(answers=words, opening=None, max_turns=2),
        patterns=patterns,
    )
    env.reset("SHORE")
    env.step("CRANE")
    _, reward, done, info = env.step("APPLE")
    assert done is True
    assert reward == 0
    assert info["terminal_reason"] == "exhausted"
    assert env.last_record.turn == 2


def test_observation_does_not_leak_answer(environment):
    start = environment.reset("SHORE")
    assert "SHORE" not in start.observation.prompt
    assert "SHORE" not in json.dumps(start.observation.to_dict())
    _, _, done, info = environment.step("CRANE")
    assert done is False
    assert "answer" not in info


def test_reset_live_episode_is_rejected(environment):
    environment.reset("SHORE")
    with pytest.raises(RuntimeError, match="live"):
        environment.reset("CRANE")


def test_reset_after_terminal_episode_is_allowed(environment):
    environment.reset("SHORE")
    environment.step("SHORE")
    start = environment.reset("CRANE")
    assert start.observation.history[0].guess == "RAISE"


def test_trace_round_trip_and_tamper_detection(environment, config):
    environment.reset("SHORE", episode_id="round-trip", seed=11)
    environment.step("CRANE")
    environment.step("APPLE")
    payload = serialize_trace(environment.trace)
    replayed = replay_trace(payload, config=config, patterns=pattern_matrix())
    assert replayed.to_dict() == payload

    tampered = json.loads(json.dumps(payload))
    tampered["records"][1]["feedback"] = "GGGGG"
    with pytest.raises(ValueError, match="mismatch"):
        replay_trace(tampered, config=config, patterns=pattern_matrix())


def test_shared_valid_domain_matches_benchmark_mechanics(environment):
    guesses = iter(["RAISE", "CRANE", "SHORE"])
    original = benchmark.generate_raw_guess
    benchmark.generate_raw_guess = lambda *args, **kwargs: next(guesses)
    try:
        result = benchmark.play_model_game(
            "SHORE",
            tokenizer=None,
            model=None,
            device=None,
        )
    finally:
        benchmark.generate_raw_guess = original

    env = WordleEnvironment(
        EnvironmentConfig(answers=ANSWERS, opening=None),
        patterns=pattern_matrix(),
    )
    env.reset("SHORE")
    for guess in ("RAISE", "CRANE", "SHORE"):
        env.step(guess)
    assert [record.feedback for record in env.trace.records] == [
        step["feedback"] for step in result.trace
    ]
    assert env.last_record.terminal_reason == "solved"
    assert result.solved
    assert env.last_record.turn == result.turns_used


def test_out_of_vocabulary_diverges_from_benchmark(environment):
    assert parse_guess("ZZZZZ") == "ZZZZZ"
    environment.reset("SHORE")
    _, _, done, info = environment.step("ZZZZZ")
    assert done
    assert info["terminal_reason"] == "contract_violation"


def test_pattern_matrix_orientation_matches_score_string():
    patterns = pattern_matrix()
    for guess_index, guess in enumerate(ANSWERS):
        for answer_index, answer in enumerate(ANSWERS):
            assert patterns[guess_index, answer_index] == encode_feedback(
                score_string(answer, guess)
            )
