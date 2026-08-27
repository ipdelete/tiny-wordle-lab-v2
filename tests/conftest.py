from __future__ import annotations

from pathlib import Path

import pytest

from tiny_wordle_lab_v2.lexicon import Lexicon, LexiconEntry


@pytest.fixture
def toy_lexicon() -> Lexicon:
    answers = ("apple", "alley", "banal", "crane")
    legal_guesses = (*answers, "adieu", "raise", "slate")
    frequencies = {
        "apple": 4.76,
        "alley": 3.77,
        "banal": 3.08,
        "crane": 3.92,
        "adieu": 2.63,
        "raise": 5.16,
        "slate": 3.86,
    }
    return Lexicon(
        answers=answers,
        legal_guesses=legal_guesses,
        entries={
            word: LexiconEntry(
                word=word,
                is_answer=word in answers,
                zipf_frequency=frequencies[word],
                parts_of_speech=("noun",),
            )
            for word in legal_guesses
        },
        source_hashes={"toy.txt": "0" * 64},
    )


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
