from dataclasses import replace

import numpy as np
import pytest

from tiny_wordle.environment import EnvironmentConfig, WordleEnvironment
from tiny_wordle.expert import EntropyExpert, encode_feedback
from tiny_wordle.game import score_string
from tiny_wordle.rollout import (
    PolicyDecision,
    collect_trajectory,
    replay_trajectory,
)

ANSWERS = ("CRANE", "SHORE", "APPLE")


def patterns():
    return np.asarray(
        [
            [
                encode_feedback(score_string(answer, guess))
                for answer in ANSWERS
            ]
            for guess in ANSWERS
        ],
        dtype=np.int16,
    )


class StubPolicy:
    def sample(self, observation, *, temperature, seed):
        word = "CRANE" if not observation.history else "SHORE"
        return PolicyDecision(
            word=word,
            token_ids=(1,),
            per_token_log_probabilities=(-0.5,),
            action_log_probability=-0.5,
            checkpoint_digest="behavior",
            temperature=temperature,
            mask_version="stub",
            sampling_seed=seed,
            tokenizer_digest="tokenizer",
        )

    def log_probability(self, observation, word, *, temperature):
        return -0.5, (-0.5,)


def environment():
    matrix = patterns()
    return WordleEnvironment(
        EnvironmentConfig(answers=ANSWERS, opening=None),
        expert=EntropyExpert(list(ANSWERS), matrix),
        patterns=matrix,
    )


def test_collector_binds_decision_to_transition():
    trajectory = collect_trajectory(
        environment(),
        StubPolicy(),
        "SHORE",
        group_id="group-1",
        answer_split="dev",
        reference_checkpoint_digest="reference",
        temperature=0.8,
        sampling_seed=10,
    )
    assert trajectory.return_value == 1
    assert trajectory.terminal_reason == "solved"
    assert len(trajectory.steps) == 2
    assert trajectory.steps[0].observation == (
        trajectory.steps[0].transition.observation_before
    )
    assert trajectory.steps[0].decision.word == "CRANE"
    assert trajectory.steps[-1].transition.terminal_reason == "solved"


def test_trajectory_replay_recomputes_probability_and_transition():
    trajectory = collect_trajectory(
        environment(),
        StubPolicy(),
        "SHORE",
        group_id="group-1",
        answer_split="dev",
        temperature=1.0,
        sampling_seed=3,
    )
    trace = replay_trajectory(
        trajectory,
        config=EnvironmentConfig(answers=ANSWERS, opening=None),
        patterns=patterns(),
        policy=StubPolicy(),
    )
    assert trace.records == trajectory.records


def test_tampered_transition_is_rejected():
    trajectory = collect_trajectory(
        environment(),
        StubPolicy(),
        "SHORE",
        group_id="group-1",
        answer_split="dev",
        temperature=1.0,
        sampling_seed=3,
    )
    tampered_transition = replace(
        trajectory.steps[0].transition,
        feedback="GGGGG",
    )
    tampered_step = replace(
        trajectory.steps[0],
        transition=tampered_transition,
    )
    tampered = replace(
        trajectory,
        steps=(tampered_step,) + trajectory.steps[1:],
    )
    with pytest.raises(ValueError, match="replay mismatch"):
        replay_trajectory(
            tampered,
            config=EnvironmentConfig(answers=ANSWERS, opening=None),
            patterns=patterns(),
            policy=StubPolicy(),
        )


def test_tampered_policy_metadata_is_rejected():
    trajectory = collect_trajectory(
        environment(),
        StubPolicy(),
        "SHORE",
        group_id="group-1",
        answer_split="dev",
        temperature=1.0,
        sampling_seed=3,
    )
    tampered_decision = replace(
        trajectory.steps[0].decision,
        checkpoint_digest="different-checkpoint",
    )
    tampered_step = replace(
        trajectory.steps[0],
        decision=tampered_decision,
    )
    tampered = replace(
        trajectory,
        steps=(tampered_step,) + trajectory.steps[1:],
    )
    with pytest.raises(ValueError, match="checkpoint digest"):
        replay_trajectory(
            tampered,
            config=EnvironmentConfig(answers=ANSWERS, opening=None),
            patterns=patterns(),
            policy=StubPolicy(),
        )


def test_tampered_action_probability_is_rejected():
    trajectory = collect_trajectory(
        environment(),
        StubPolicy(),
        "SHORE",
        group_id="group-1",
        answer_split="dev",
        temperature=1.0,
        sampling_seed=3,
    )
    tampered_decision = replace(
        trajectory.steps[0].decision,
        per_token_log_probabilities=(-0.25,),
        action_log_probability=-0.25,
    )
    tampered_step = replace(
        trajectory.steps[0],
        decision=tampered_decision,
    )
    tampered = replace(
        trajectory,
        steps=(tampered_step,) + trajectory.steps[1:],
    )
    with pytest.raises(ValueError, match="log probability"):
        replay_trajectory(
            tampered,
            config=EnvironmentConfig(answers=ANSWERS, opening=None),
            patterns=patterns(),
            policy=StubPolicy(),
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("return_value", 0.0, "return"),
        ("terminal_reason", "exhausted", "terminal reason"),
    ],
)
def test_tampered_outcome_is_rejected(field, value, match):
    trajectory = collect_trajectory(
        environment(),
        StubPolicy(),
        "SHORE",
        group_id="group-1",
        answer_split="dev",
        temperature=1.0,
        sampling_seed=3,
    )
    tampered = replace(trajectory, **{field: value})
    with pytest.raises(ValueError, match=match):
        replay_trajectory(
            tampered,
            config=EnvironmentConfig(answers=ANSWERS, opening=None),
            patterns=patterns(),
            policy=StubPolicy(),
        )


def test_restriction_algorithm_comes_from_policy_decisions():
    trajectory = collect_trajectory(
        environment(),
        StubPolicy(),
        "SHORE",
        group_id="group-1",
        answer_split="dev",
        temperature=1.0,
        sampling_seed=3,
    )
    assert trajectory.restriction_algorithm == "stub"

    tampered = replace(
        trajectory,
        restriction_algorithm="different-mask",
    )
    with pytest.raises(ValueError, match="restriction algorithm"):
        replay_trajectory(
            tampered,
            config=EnvironmentConfig(answers=ANSWERS, opening=None),
            patterns=patterns(),
            policy=StubPolicy(),
        )


def test_trace_contains_teacher_diagnostics():
    trajectory = collect_trajectory(
        environment(),
        StubPolicy(),
        "SHORE",
        group_id="group-1",
        answer_split="dev",
        temperature=1.0,
        sampling_seed=3,
    )
    diagnostics = trajectory.steps[0].transition.teacher_diagnostics
    assert diagnostics.candidate_teacher_guess in ANSWERS
    assert diagnostics.candidate_teacher_entropy_bits is not None
    assert diagnostics.action_entropy_bits is not None
