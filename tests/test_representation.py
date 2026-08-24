import json
from pathlib import Path

import numpy as np

from tiny_wordle.expert import EntropyExpert
from tiny_wordle.game import Turn, filter_candidates, score_string
from tiny_wordle.representation import (
    answer_branch_of,
    candidate_indices_from_history,
    candidate_stratum,
    derive_constraints,
    parse_state_key,
    render_state_key,
    render_structured_state,
    structured_next_guess_prompt,
)

ROOT = Path(__file__).parents[1]
ANSWERS = [
    line.strip().upper()
    for line in (ROOT / "data/wordle-answers-original.txt").read_text().splitlines()
    if line.strip()
]
PATTERNS = np.load(ROOT / "data/wordle-patterns-original-2315.npy")
EXPERT = EntropyExpert(ANSWERS, PATTERNS)


def test_state_key_round_trip():
    history = [
        Turn("RAISE", "BGBBY"),
        Turn("NAVEL", "BGBGG"),
    ]
    assert parse_state_key(render_state_key(history)) == history


def test_derived_state_matches_game_candidate_filter():
    history = [
        Turn("RAISE", score_string("BAGEL", "RAISE")),
        Turn("NAVEL", score_string("BAGEL", "NAVEL")),
    ]
    candidate_indices = candidate_indices_from_history(
        history,
        ANSWERS,
        PATTERNS,
        expert=EXPERT,
    )
    expected = filter_candidates(ANSWERS, history)
    rebuilt = [ANSWERS[int(index)] for index in candidate_indices]
    assert sorted(rebuilt) == sorted(expected)
    assert len(rebuilt) == 5


def test_prompt_reproduces_every_stored_structured_state():
    files = [
        ROOT / "data/generated/wordle-part2-structured-train.jsonl",
        ROOT / "data/generated/wordle-part2-structured-dev.jsonl",
    ]
    seen = set()
    for path in files:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["task"] != "NEXT_GUESS":
                continue
            key = (row["split"], row["state_key"])
            if key in seen:
                continue
            seen.add(key)
            history = parse_state_key(row["state_key"])
            indices = candidate_indices_from_history(
                history,
                ANSWERS,
                PATTERNS,
                expert=EXPERT,
            )
            assert len(indices) == row["candidate_count"]
            assert structured_next_guess_prompt(
                history,
                len(indices),
            ) == row["prompt"]


def test_duplicate_letter_constraints_are_explicit():
    history = [Turn("APPLE", "GYBYB")]
    state = derive_constraints(history)
    assert state["greens"] == ["A", None, None, None, None]
    assert state["minimum"]["A"] == 1
    assert state["maximum"]["P"] == 1
    assert state["excluded"]["P"] == [2, 3]


def test_conflicting_constraints_raise():
    history = [
        Turn("CRANE", "GGBBB"),
        Turn("SLATE", "BGGBB"),
    ]
    try:
        derive_constraints(history)
    except ValueError as error:
        assert "conflicting green" in str(error)
    else:
        raise AssertionError("conflicting greens were accepted")


def test_answer_branch_is_none_for_non_opening_histories():
    assert answer_branch_of([]) is None
    assert answer_branch_of([Turn("FJORD", "BBBBB")]) is None
    assert answer_branch_of([Turn("RAISE", "BGBBY")]) == "BGBBY"


def test_candidate_strata():
    assert [candidate_stratum(value) for value in [1, 2, 3, 10, 11]] == [
        "1",
        "2",
        "3-10",
        "3-10",
        "11+",
    ]


def test_rendered_state_contains_no_unrequested_teacher_fields():
    rendered = render_structured_state([], 2315)
    assert "TEACHER" not in rendered
    assert "ENTROPY" not in rendered
