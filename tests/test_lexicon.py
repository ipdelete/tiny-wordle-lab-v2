from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiny_wordle_lab_v2 import lexicon


def test_load_lexicon_validates_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "wordle-answers-original.txt").write_text("apple\n")
    (data / "wordle-guesses-original.txt").write_text("apple\ncrane\n")
    records = [
        {
            "word": "apple",
            "is_original_answer": True,
            "zipf_frequency": 4.76,
            "parts_of_speech": ["noun"],
        },
        {
            "word": "crane",
            "is_original_answer": False,
            "zipf_frequency": 3.92,
            "parts_of_speech": ["noun"],
        },
    ]
    (data / "wordle-lexicon.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    monkeypatch.setattr(lexicon, "ANSWER_COUNT", 1)
    monkeypatch.setattr(lexicon, "LEGAL_GUESS_COUNT", 2)

    loaded = lexicon.load_lexicon(tmp_path)
    assert loaded.answers == ("apple",)
    assert loaded.legal_guesses == ("apple", "crane")
    assert loaded.entries["apple"].is_answer


def test_load_lexicon_rejects_wrong_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "wordle-answers-original.txt").write_text("apple\n")
    (data / "wordle-guesses-original.txt").write_text("apple\n")
    record = {
        "word": "apple",
        "is_original_answer": False,
        "zipf_frequency": 4.76,
        "parts_of_speech": ["noun"],
    }
    (data / "wordle-lexicon.jsonl").write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(lexicon, "ANSWER_COUNT", 1)
    monkeypatch.setattr(lexicon, "LEGAL_GUESS_COUNT", 1)

    with pytest.raises(ValueError, match="answer membership"):
        lexicon.load_lexicon(tmp_path)


def test_load_lexicon_enforces_record_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "wordle-answers-original.txt").write_text("apple\n")
    (data / "wordle-guesses-original.txt").write_text("apple\n")
    record = {
        "word": "apple",
        "is_original_answer": True,
        "zipf_frequency": 4.76,
        "parts_of_speech": ["mystery"],
    }
    (data / "wordle-lexicon.jsonl").write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(lexicon, "ANSWER_COUNT", 1)
    monkeypatch.setattr(lexicon, "LEGAL_GUESS_COUNT", 1)

    with pytest.raises(ValueError, match="record schema"):
        lexicon.load_lexicon(tmp_path)
