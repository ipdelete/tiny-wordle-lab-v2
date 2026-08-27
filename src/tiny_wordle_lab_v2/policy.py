from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .game import Observation


PolicyParameter = str | int | float | bool | None


@dataclass(frozen=True)
class PolicyDescriptor:
    name: str
    parameters: dict[str, PolicyParameter]


class Policy(Protocol):
    @property
    def descriptor(self) -> PolicyDescriptor: ...

    def choose(self, observation: Observation) -> str: ...
