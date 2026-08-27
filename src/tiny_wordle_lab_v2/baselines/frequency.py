from __future__ import annotations

from ..game import Observation
from ..lexicon import Lexicon
from ..policy import PolicyDescriptor


class FrequencyPolicy:
    def __init__(self, lexicon: Lexicon) -> None:
        self._entries = lexicon.entries

    @property
    def descriptor(self) -> PolicyDescriptor:
        return PolicyDescriptor("frequency", {})

    def choose(self, observation: Observation) -> str:
        if not observation.candidates:
            raise RuntimeError("frequency policy received an empty candidate set")
        return min(
            observation.candidates,
            key=lambda word: (
                -(
                    self._entries[word].zipf_frequency
                    if self._entries[word].zipf_frequency is not None
                    else float("-inf")
                ),
                word,
            ),
        )
