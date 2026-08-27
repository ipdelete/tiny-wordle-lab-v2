from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..game import Observation
from ..lexicon import Lexicon
from ..policy import PolicyDescriptor


def _letter_codes(words: tuple[str, ...]) -> NDArray[np.uint8]:
    return np.frombuffer("".join(words).encode("ascii"), dtype=np.uint8).reshape(
        len(words), 5
    ) - ord("a")


def build_feedback_matrix(
    actions: tuple[str, ...],
    answers: tuple[str, ...],
) -> NDArray[np.uint8]:
    answer_codes = _letter_codes(answers)
    action_codes = _letter_codes(actions)
    matrix = np.empty((len(actions), len(answers)), dtype=np.uint8)

    for action_index, guess in enumerate(action_codes):
        green = answer_codes == guess
        marks = green.astype(np.uint8) * 2
        for position, letter in enumerate(guess):
            nongreen = ~green[:, position]
            available = ((answer_codes == letter) & ~green).sum(axis=1)
            prior = np.zeros(len(answers), dtype=np.uint8)
            for earlier in range(position):
                if guess[earlier] == letter:
                    prior += ~green[:, earlier]
            marks[:, position] = np.where(
                nongreen & (prior < available),
                1,
                marks[:, position],
            )
        matrix[action_index] = (
            (((marks[:, 0] * 3 + marks[:, 1]) * 3 + marks[:, 2]) * 3
             + marks[:, 3])
            * 3
            + marks[:, 4]
        )
    return matrix


@dataclass
class EntropyPolicy:
    lexicon: Lexicon
    candidate_only: bool
    _matrix: NDArray[np.uint8] | None = None
    _choice_cache: dict[tuple[int, ...], str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._answer_indices = {
            word: index for index, word in enumerate(self.lexicon.answers)
        }
        self._action_words = (
            self.lexicon.answers
            if self.candidate_only
            else self.lexicon.legal_guesses
        )
        self._action_indices = {
            word: index for index, word in enumerate(self._action_words)
        }
        if self._matrix is None:
            self._matrix = build_feedback_matrix(
                self._action_words,
                self.lexicon.answers,
            )
        expected_shape = (len(self._action_words), len(self.lexicon.answers))
        if self._matrix.shape != expected_shape:
            raise ValueError(
                f"feedback matrix has shape {self._matrix.shape}, "
                f"expected {expected_shape}"
            )
        self._x_log2_x = np.zeros(len(self.lexicon.answers) + 1)
        positive = np.arange(1, len(self.lexicon.answers) + 1)
        self._x_log2_x[1:] = positive * np.log2(positive)

    @property
    def descriptor(self) -> PolicyDescriptor:
        return PolicyDescriptor(
            "candidate-entropy" if self.candidate_only else "open-entropy",
            {
                "action_space": (
                    "remaining_answers"
                    if self.candidate_only
                    else "all_legal_guesses"
                ),
                "action_count": len(self._action_words),
            },
        )

    def _entropies(
        self,
        candidate_indices: NDArray[np.int64],
        action_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        candidate_count = len(candidate_indices)
        values = np.empty(len(action_indices), dtype=np.float64)
        chunk_size = 1_024
        for start in range(0, len(action_indices), chunk_size):
            selected = action_indices[start : start + chunk_size]
            patterns = self._matrix[selected][:, candidate_indices].astype(
                np.int64,
                copy=False,
            )
            offsets = np.arange(len(selected), dtype=np.int64)[:, None] * 243
            counts = np.bincount(
                (patterns + offsets).ravel(),
                minlength=len(selected) * 243,
            ).reshape(len(selected), 243)
            values[start : start + len(selected)] = (
                math.log2(candidate_count)
                - self._x_log2_x[counts].sum(axis=1) / candidate_count
            )
        return values

    def choose(self, observation: Observation) -> str:
        if not observation.candidates:
            raise RuntimeError("entropy policy received an empty candidate set")
        if len(observation.candidates) == 1:
            return observation.candidates[0]

        key = tuple(
            self._answer_indices[word] for word in observation.candidates
        )
        cached = self._choice_cache.get(key)
        if cached is not None:
            return cached

        candidate_indices = np.asarray(key, dtype=np.int64)
        if self.candidate_only:
            action_indices = np.asarray(
                [self._action_indices[word] for word in observation.candidates],
                dtype=np.int64,
            )
        else:
            action_indices = np.arange(len(self._action_words), dtype=np.int64)
        entropies = self._entropies(candidate_indices, action_indices)
        best = float(entropies.max())
        tied = action_indices[
            np.isclose(entropies, best, rtol=0.0, atol=1e-12)
        ]

        if not self.candidate_only:
            candidate_action_indices = {
                self._action_indices[word] for word in observation.candidates
            }
            tied_candidates = np.asarray(
                [index for index in tied if int(index) in candidate_action_indices],
                dtype=np.int64,
            )
            if len(tied_candidates):
                tied = tied_candidates

        choice = min(self._action_words[int(index)] for index in tied)
        self._choice_cache[key] = choice
        return choice
