from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .lexicon import ROOT


DEFAULT_PROMPT_PATH = ROOT / "prompts" / "wordle-player" / "v1-baseline.md"


@dataclass(frozen=True)
class WordlePrompt:
    content: str
    source: str
    sha256: str

    @classmethod
    def from_text(cls, content: str, *, source: str) -> WordlePrompt:
        normalized = content.rstrip() + "\n"
        return cls(
            content=normalized,
            source=source,
            sha256=hashlib.sha256(normalized.encode()).hexdigest(),
        )

    @classmethod
    def from_path(cls, path: Path = DEFAULT_PROMPT_PATH) -> WordlePrompt:
        resolved = path.resolve()
        try:
            source = str(resolved.relative_to(ROOT))
        except ValueError:
            source = str(resolved)
        return cls.from_text(resolved.read_text(), source=source)


def load_answer_file(path: Path) -> tuple[str, ...]:
    return tuple(
        line.strip().lower()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
