from __future__ import annotations

import hashlib
import random

from ..game import Observation, parse_word
from ..lexicon import Lexicon
from ..policy import PolicyDescriptor


class RandomPolicy:
    def __init__(self, lexicon: Lexicon, seed: int = 0) -> None:
        self._legal_guesses = lexicon.legal_guesses
        self._seed = seed

    @property
    def descriptor(self) -> PolicyDescriptor:
        return PolicyDescriptor("random", {"seed": self._seed})

    def choose(self, observation: Observation) -> str:
        used = {
            word
            for action in observation.previous_actions
            if (word := parse_word(action)) is not None
        }
        available = tuple(word for word in self._legal_guesses if word not in used)
        if not available:
            raise RuntimeError("random policy exhausted legal guesses")
        state = "\0".join(
            [
                str(self._seed),
                *(f"{turn.guess}:{turn.feedback}" for turn in observation.history),
            ]
        )
        seed = int.from_bytes(
            hashlib.sha256(state.encode()).digest()[:8],
            byteorder="big",
        )
        return random.Random(seed).choice(available)
