from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum


class Mark(str, Enum):
    GRAY = "B"
    YELLOW = "Y"
    GREEN = "G"


@dataclass(frozen=True)
class Turn:
    guess: str
    feedback: str


def score_guess(answer: str, guess: str) -> tuple[Mark, ...]:
    answer = answer.upper()
    guess = guess.upper()

    if len(answer) != 5 or len(guess) != 5:
        raise ValueError("answer and guess must both contain exactly 5 letters")
    if not answer.isalpha() or not guess.isalpha():
        raise ValueError("answer and guess must contain letters only")

    marks = [Mark.GRAY] * 5
    remaining = Counter()

    for i, (a, g) in enumerate(zip(answer, guess)):
        if a == g:
            marks[i] = Mark.GREEN
        else:
            remaining[a] += 1

    for i, g in enumerate(guess):
        if marks[i] == Mark.GREEN:
            continue
        if remaining[g] > 0:
            marks[i] = Mark.YELLOW
            remaining[g] -= 1

    return tuple(marks)


def score_string(answer: str, guess: str) -> str:
    return "".join(mark.value for mark in score_guess(answer, guess))


def play_turn(answer: str, guess: str) -> Turn:
    return Turn(guess=guess.upper(), feedback=score_string(answer, guess))


def is_consistent(candidate: str, history: list[Turn]) -> bool:
    candidate = candidate.upper()
    return all(
        score_string(candidate, turn.guess) == turn.feedback
        for turn in history
    )


def filter_candidates(words: list[str], history: list[Turn]) -> list[str]:
    return [
        word.upper()
        for word in words
        if len(word) == 5
        and word.isalpha()
        and is_consistent(word, history)
    ]
