from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from .expert import EntropyExpert
from .game import Turn

REPRESENTATION_VERSION = "derived_state_v1"


def parse_state_key(state_key: str) -> list[Turn]:
    if not state_key:
        return []
    history = []
    for line in state_key.splitlines():
        guess_text, feedback_text = line.split(" -> ")
        history.append(
            Turn(
                guess=guess_text.replace(" ", ""),
                feedback=feedback_text.replace(" ", ""),
            )
        )
    return history


def render_state_key(history: Sequence[Turn]) -> str:
    return "\n".join(
        f"{' '.join(turn.guess)} -> {' '.join(turn.feedback)}"
        for turn in history
    )


def derive_constraints(history: Sequence[Turn]) -> dict:
    greens = [None] * 5
    minimum = defaultdict(int)
    maximum = defaultdict(lambda: 5)
    excluded = defaultdict(set)
    for turn in history:
        marks_by_letter = defaultdict(list)
        for position, (letter, mark) in enumerate(
            zip(turn.guess, turn.feedback), 1
        ):
            marks_by_letter[letter].append(mark)
            if mark == "G":
                if greens[position - 1] not in (None, letter):
                    raise ValueError("conflicting green constraints")
                greens[position - 1] = letter
            else:
                excluded[letter].add(position)
        for letter, marks in marks_by_letter.items():
            matched = sum(mark in {"Y", "G"} for mark in marks)
            minimum[letter] = max(minimum[letter], matched)
            if matched < len(marks):
                maximum[letter] = min(maximum[letter], matched)
    for letter in minimum:
        if minimum[letter] > maximum.get(letter, 5):
            raise ValueError(f"impossible count constraint for {letter}")
    return {
        "greens": greens,
        "minimum": dict(minimum),
        "maximum": dict(maximum),
        "excluded": {
            letter: sorted(positions)
            for letter, positions in excluded.items()
        },
        "previous_guesses": [turn.guess for turn in history],
    }


def render_structured_state(
    history: Sequence[Turn],
    candidate_count: int,
) -> str:
    state = derive_constraints(history)
    greens = " ".join(letter or "_" for letter in state["greens"])
    present = sorted(
        letter
        for letter, count in state["minimum"].items()
        if count > 0
    )
    counts = []
    for letter in present:
        low = state["minimum"][letter]
        high = state["maximum"].get(letter, 5)
        counts.append(
            f"{letter}={low}..{high}" if high < 5 else f"{letter}>={low}"
        )
    absent = sorted(
        letter
        for letter, count in state["maximum"].items()
        if count == 0
    )
    excluded = []
    for letter in sorted(state["excluded"]):
        positions = ",".join(map(str, state["excluded"][letter]))
        excluded.append(f"{letter}@{positions}")
    return "\n".join(
        [
            f"GREENS: {greens}",
            f"LETTER_COUNTS: {', '.join(counts) or 'NONE'}",
            f"EXCLUDED_POSITIONS: {', '.join(excluded) or 'NONE'}",
            f"ABSENT_LETTERS: {' '.join(absent) or 'NONE'}",
            (
                "PREVIOUS_GUESSES: "
                f"{', '.join(state['previous_guesses']) or 'NONE'}"
            ),
            f"CANDIDATE_COUNT: {candidate_count}",
        ]
    )


def structured_next_guess_prompt(
    history: Sequence[Turn],
    candidate_count: int,
) -> str:
    return (
        "Task: NEXT_GUESS\n"
        "You are playing Wordle.\n"
        "Use the game history to choose the next guess.\n"
        "Return exactly one uppercase five-letter word.\n\n"
        "Derived state:\n"
        + render_structured_state(history, candidate_count)
    )


def candidate_indices_from_history(
    history: Sequence[Turn],
    answers: Sequence[str],
    patterns: np.ndarray,
    *,
    expert: EntropyExpert | None = None,
) -> np.ndarray:
    if expert is None:
        expert = EntropyExpert(list(answers), patterns)
    word_to_index = expert.word_to_index
    indices = expert.all_indices
    for turn in history:
        if turn.guess not in word_to_index:
            raise ValueError(
                f"guess outside the answer lexicon: {turn.guess}"
            )
        indices = expert.update(
            indices,
            word_to_index[turn.guess],
            turn.feedback,
        )
    if len(indices) == 0:
        raise ValueError("state key produced an empty candidate set")
    return indices


def candidate_stratum(candidate_count: int) -> str:
    if candidate_count == 1:
        return "1"
    if candidate_count == 2:
        return "2"
    if candidate_count <= 10:
        return "3-10"
    return "11+"


def answer_branch_of(
    history: Sequence[Turn],
    opening: str = "RAISE",
) -> str | None:
    if not history or history[0].guess != opening:
        return None
    return history[0].feedback
