from __future__ import annotations

import pytest

from tiny_wordle_lab_v2.game import Turn, filter_candidates, is_consistent, score_guess


@pytest.mark.parametrize(
    "answer,guess,expected",
    [
        ("crane", "crane", "GGGGG"),
        ("fuzzy", "crane", "BBBBB"),
        ("apple", "alley", "GYBYB"),
        ("crane", "mamma", "BYBBB"),
        ("sheep", "eerie", "YYBBB"),
    ],
)
def test_score_guess(answer: str, guess: str, expected: str) -> None:
    assert score_guess(answer, guess) == expected


def test_score_rejects_non_ascii_or_wrong_length() -> None:
    with pytest.raises(ValueError):
        score_guess("crane", "cat")
    with pytest.raises(ValueError):
        score_guess("crane", "éclat")


def test_candidate_replay_preserves_hidden_answer() -> None:
    candidates = ("apple", "alley", "banal", "crane")
    answer = "banal"
    for guess in ("crane", "alley"):
        feedback = score_guess(answer, guess)
        candidates = filter_candidates(candidates, guess, feedback)
        assert answer in candidates


def test_consistency_replays_history() -> None:
    history = (
        Turn("crane", score_guess("banal", "crane")),
        Turn("alley", score_guess("banal", "alley")),
    )
    assert is_consistent("banal", history)
    assert not is_consistent("crane", history)
