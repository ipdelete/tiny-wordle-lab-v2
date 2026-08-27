from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from tiny_wordle_lab_v2.evaluate import EvaluationConfig, evaluate
from tiny_wordle_lab_v2.game import Observation
from tiny_wordle_lab_v2.lexicon import Lexicon
from tiny_wordle_lab_v2.policy import PolicyDescriptor


@dataclass
class ScriptedPolicy:
    outputs: list[str]

    @property
    def descriptor(self) -> PolicyDescriptor:
        return PolicyDescriptor("scripted", {})

    def choose(self, observation: Observation) -> str:
        return self.outputs.pop(0)


def test_runner_freezes_illegal_and_repeat_semantics(
    toy_lexicon: Lexicon,
) -> None:
    result = evaluate(
        ScriptedPolicy(["no", "apple", "apple", "banal"]),
        toy_lexicon,
        EvaluationConfig(
            experiment_id="semantics",
            answers=("banal",),
            max_opportunities=4,
        ),
    )
    game = result.games[0]
    assert game.solved
    assert game.opportunities_used == 4
    assert [action.status for action in game.actions] == [
        "illegal",
        "accepted",
        "repeat",
        "solved",
    ]
    assert result.summary.illegal_actions == 1
    assert result.summary.repeat_actions == 1
    assert game.policy_calls == 4
    assert game.accepted_guesses == ("apple", "apple", "banal")
    assert result.summary.penalized_turns == 4


def test_failure_always_costs_seven(toy_lexicon: Lexicon) -> None:
    result = evaluate(
        ScriptedPolicy(["no"] * 3),
        toy_lexicon,
        EvaluationConfig(
            experiment_id="failure",
            answers=("banal",),
            max_opportunities=3,
        ),
    )
    assert not result.games[0].solved
    assert result.summary.penalized_turns == 7


def test_evaluation_result_is_deterministic(toy_lexicon: Lexicon) -> None:
    first = evaluate(
        ScriptedPolicy(["banal"]),
        toy_lexicon,
        EvaluationConfig("deterministic", answers=("banal",)),
    )
    second = evaluate(
        ScriptedPolicy(["banal"]),
        toy_lexicon,
        EvaluationConfig("deterministic", answers=("banal",)),
    )
    assert first == second


def test_end_to_end_result_matches_golden_fixture(
    toy_lexicon: Lexicon,
    repo_root: Path,
) -> None:
    result = evaluate(
        ScriptedPolicy(["no", "apple", "apple", "banal"]),
        toy_lexicon,
        EvaluationConfig("golden", answers=("banal",)),
    )
    expected = json.loads(
        (repo_root / "tests" / "fixtures" / "golden-evaluation.json").read_text()
    )
    actual = json.loads(json.dumps(asdict(result)))
    assert actual == expected
