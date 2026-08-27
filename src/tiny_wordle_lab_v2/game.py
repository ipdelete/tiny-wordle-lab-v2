from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


WORD_LENGTH = 5
SOLVED_FEEDBACK = "GGGGG"


class Mark(str, Enum):
    GRAY = "B"
    YELLOW = "Y"
    GREEN = "G"


@dataclass(frozen=True)
class Turn:
    guess: str
    feedback: str


@dataclass(frozen=True)
class Observation:
    history: tuple[Turn, ...]
    previous_actions: tuple[str, ...]
    remaining_opportunities: int
    candidates: tuple[str, ...]


def parse_word(value: str) -> str | None:
    word = value.strip().lower()
    if (
        len(word) != WORD_LENGTH
        or not word.isascii()
        or not word.isalpha()
    ):
        return None
    return word


def _require_word(value: str, label: str) -> str:
    word = parse_word(value)
    if word is None:
        raise ValueError(f"{label} must contain exactly five ASCII letters")
    return word


def score_guess(answer: str, guess: str) -> str:
    answer = _require_word(answer, "answer")
    guess = _require_word(guess, "guess")
    marks = [Mark.GRAY] * WORD_LENGTH
    remaining: Counter[str] = Counter()

    for index, (answer_letter, guess_letter) in enumerate(zip(answer, guess)):
        if answer_letter == guess_letter:
            marks[index] = Mark.GREEN
        else:
            remaining[answer_letter] += 1

    for index, guess_letter in enumerate(guess):
        if marks[index] == Mark.GREEN:
            continue
        if remaining[guess_letter] > 0:
            marks[index] = Mark.YELLOW
            remaining[guess_letter] -= 1

    return "".join(mark.value for mark in marks)


def is_consistent(candidate: str, history: Iterable[Turn]) -> bool:
    return all(
        score_guess(candidate, turn.guess) == turn.feedback for turn in history
    )


def filter_candidates(
    candidates: Iterable[str],
    guess: str,
    feedback: str,
) -> tuple[str, ...]:
    if len(feedback) != WORD_LENGTH or any(mark not in "BYG" for mark in feedback):
        raise ValueError("feedback must contain exactly five B, Y, or G marks")
    return tuple(
        candidate
        for candidate in candidates
        if score_guess(candidate, guess) == feedback
    )
