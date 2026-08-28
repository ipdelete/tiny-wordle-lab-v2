from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from jsonschema import Draft202012Validator


ANSWER_COUNT = 2_315
LEGAL_GUESS_COUNT = 12_972
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LexiconEntry:
    word: str
    is_answer: bool
    zipf_frequency: float | None
    parts_of_speech: tuple[str, ...]


@dataclass(frozen=True)
class Lexicon:
    answers: tuple[str, ...]
    legal_guesses: tuple[str, ...]
    entries: Mapping[str, LexiconEntry]
    source_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))
        object.__setattr__(
            self,
            "source_hashes",
            MappingProxyType(dict(self.source_hashes)),
        )


def _read_words(path: Path) -> tuple[str, ...]:
    words = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    invalid = [
        word
        for word in words
        if len(word) != 5
        or not word.isascii()
        or not word.isalpha()
        or not word.islower()
    ]
    if invalid:
        raise ValueError(f"{path} contains invalid words: {invalid[:5]}")
    if len(words) != len(set(words)):
        raise ValueError(f"{path} contains duplicate words")
    return words


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lexicon(root: Path = ROOT) -> Lexicon:
    data_dir = root / "data"
    answers_path = data_dir / "wordle-answers-original.txt"
    guesses_path = data_dir / "wordle-guesses-original.txt"
    records_path = data_dir / "wordle-lexicon.jsonl"
    schema = json.loads(
        (ROOT / "data" / "wordle-lexicon-record.schema.json").read_text()
    )
    validator = Draft202012Validator(schema)

    answers = _read_words(answers_path)
    legal_guesses = _read_words(guesses_path)
    if len(answers) != ANSWER_COUNT:
        raise ValueError(f"expected {ANSWER_COUNT} answers, found {len(answers)}")
    if len(legal_guesses) != LEGAL_GUESS_COUNT:
        raise ValueError(
            f"expected {LEGAL_GUESS_COUNT} legal guesses, found {len(legal_guesses)}"
        )

    answer_set = set(answers)
    legal_set = set(legal_guesses)
    missing_answers = answer_set - legal_set
    if missing_answers:
        raise ValueError(f"legal guesses omit answers: {sorted(missing_answers)[:5]}")

    entries: dict[str, LexiconEntry] = {}
    record_order: list[str] = []
    with records_path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            errors = list(validator.iter_errors(record))
            if errors:
                raise ValueError(
                    f"{records_path}:{line_number} violates the record schema: "
                    f"{errors[0].message}"
                )
            word = record["word"]
            frequency = record["zipf_frequency"]
            parts = record["parts_of_speech"]
            if word not in legal_set or (
                frequency is not None and not math.isfinite(frequency)
            ):
                raise ValueError(f"{records_path}:{line_number} is invalid")
            if word in entries:
                raise ValueError(f"{records_path} contains duplicate word {word}")
            if record["is_original_answer"] != (word in answer_set):
                raise ValueError(f"{word} has incorrect answer membership")
            entries[word] = LexiconEntry(
                word=word,
                is_answer=record["is_original_answer"],
                zipf_frequency=float(frequency) if frequency is not None else None,
                parts_of_speech=tuple(parts),
            )
            record_order.append(word)

    if set(entries) != legal_set:
        missing = legal_set - set(entries)
        extra = set(entries) - legal_set
        raise ValueError(
            f"enriched lexicon mismatch: missing={sorted(missing)[:5]}, "
            f"extra={sorted(extra)[:5]}"
        )
    if record_order != sorted(record_order):
        raise ValueError(f"{records_path} must be alphabetically ordered")

    return Lexicon(
        answers=answers,
        legal_guesses=legal_guesses,
        entries=entries,
        source_hashes={
            path.name: _sha256(path)
            for path in (answers_path, guesses_path, records_path)
        },
    )
