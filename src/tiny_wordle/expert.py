from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MARK_DIGIT = {"B": 0, "Y": 1, "G": 2}
DIGIT_MARK = {0: "B", 1: "Y", 2: "G"}


def encode_feedback(feedback: str) -> int:
    value = 0
    for mark in feedback:
        value = value * 3 + MARK_DIGIT[mark]
    return value


def decode_feedback(value: int) -> str:
    digits = [0] * 5
    for i in range(4, -1, -1):
        digits[i] = value % 3
        value //= 3
    return "".join(DIGIT_MARK[d] for d in digits)


@dataclass
class EntropyExpert:
    answers: list[str]
    patterns: np.ndarray

    def __post_init__(self):
        self.word_to_index = {
            word: i for i, word in enumerate(self.answers)
        }
        self.all_indices = np.arange(
            len(self.answers),
            dtype=np.int32,
        )

    def entropy(
        self,
        guess_index: int,
        candidate_indices: np.ndarray,
    ) -> float:
        patterns = self.patterns[
            guess_index,
            candidate_indices,
        ]
        counts = np.bincount(
            patterns,
            minlength=243,
        )
        counts = counts[counts > 0]
        probabilities = counts / counts.sum()

        return float(
            -(probabilities * np.log2(probabilities)).sum()
        )

    def choose(
        self,
        candidate_indices: np.ndarray,
    ) -> int:
        if len(candidate_indices) == 0:
            raise ValueError("candidate set is empty")

        if len(candidate_indices) == 1:
            return int(candidate_indices[0])

        best_index = None
        best_entropy = -1.0
        best_word = None

        for guess_index in candidate_indices:
            guess_index = int(guess_index)
            entropy = self.entropy(
                guess_index,
                candidate_indices,
            )
            word = self.answers[guess_index]

            if (
                entropy > best_entropy + 1e-12
                or (
                    abs(entropy - best_entropy) <= 1e-12
                    and (
                        best_word is None
                        or word < best_word
                    )
                )
            ):
                best_index = guess_index
                best_entropy = entropy
                best_word = word

        return int(best_index)

    def update(
        self,
        candidate_indices: np.ndarray,
        guess_index: int,
        feedback: str,
    ) -> np.ndarray:
        pattern_id = encode_feedback(feedback)
        matches = (
            self.patterns[
                guess_index,
                candidate_indices,
            ]
            == pattern_id
        )
        return candidate_indices[matches]
