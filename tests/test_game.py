import pytest

from tiny_wordle.game import Turn, filter_candidates, is_consistent, score_string


@pytest.mark.parametrize(
    "answer,guess,expected",
    [
        ("CRANE", "CRANE", "GGGGG"),
        ("FUZZY", "CRANE", "BBBBB"),
        ("APPLE", "ALLEY", "GYBYB"),
        ("CRANE", "MAMMA", "BYBBB"),
        ("SHEEP", "EERIE", "YYBBB"),
    ],
)
def test_score_guess(answer, guess, expected):
    assert score_string(answer, guess) == expected


def test_invalid_length():
    with pytest.raises(ValueError):
        score_string("CRANE", "CAT")


def test_consistency_replays_feedback():
    history = [
        Turn("CRANE", score_string("PLANT", "CRANE")),
        Turn("SLATE", score_string("PLANT", "SLATE")),
    ]
    assert is_consistent("PLANT", history)
    assert not is_consistent("CRANE", history)


def test_filter_candidates():
    history = [Turn("CRANE", score_string("PLANT", "CRANE"))]
    words = ["PLANT", "CRANE", "POINT", "CHANT"]
    result = filter_candidates(words, history)

    assert "PLANT" in result
    assert "CRANE" not in result
