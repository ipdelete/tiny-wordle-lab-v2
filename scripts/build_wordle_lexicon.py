from __future__ import annotations

import argparse
import gzip
import html
import importlib.metadata
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from wordfreq import zipf_frequency


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_CACHE_DIR = ROOT / ".cache" / "wordle-lexicon"
ANSWERS_PATH = DATA_DIR / "wordle-answers-original.txt"
GUESSES_PATH = DATA_DIR / "wordle-guesses-original.txt"
SCHEMA_PATH = DATA_DIR / "wordle-lexicon-record.schema.json"
OUTPUT_PATH = DATA_DIR / "wordle-lexicon.jsonl"
METADATA_PATH = DATA_DIR / "wordle-lexicon.metadata.json"
UNCLASSIFIED_PATH = DATA_DIR / "wordle-pos-unclassified.txt"
QWEN_POS_PATH = DATA_DIR / "wordle-pos-qwen.jsonl"
QWEN_METADATA_PATH = DATA_DIR / "wordle-pos-qwen.metadata.json"

ANSWER_COUNT = 2_315
GUESS_COUNT = 12_972
KAIKKI_POS = {
    "adj": "adjective",
    "adv": "adverb",
    "conj": "conjunction",
    "contraction": "contraction",
    "det": "determiner",
    "intj": "interjection",
    "name": "proper_noun",
    "noun": "noun",
    "num": "numeral",
    "particle": "particle",
    "phrase": "phrase",
    "postp": "postposition",
    "prep": "preposition",
    "prep_phrase": "prepositional_phrase",
    "pron": "pronoun",
    "verb": "verb",
}
MOBY_POS = {
    "N": "noun",
    "p": "noun",
    "h": "phrase",
    "V": "verb",
    "t": "verb",
    "i": "verb",
    "A": "adjective",
    "v": "adverb",
    "C": "conjunction",
    "P": "preposition",
    "!": "interjection",
    "r": "pronoun",
    "D": "determiner",
    "I": "determiner",
    "o": "noun",
}
GCIDE_BLOCK = re.compile(r"<p><ent>(.*?)</ent>(.*?)</p>", re.DOTALL)
GCIDE_POS = re.compile(r"<pos>(.*?)</pos>", re.DOTALL)
MARKUP = re.compile(r"<[^>]+>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    return parser.parse_args()


def read_words(path: Path) -> list[str]:
    words = [line.strip() for line in path.read_text().splitlines() if line.strip()]
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


def add_kaikki_entry(
    classifications: dict[str, set[str]],
    entry: dict[str, Any],
    targets: set[str],
) -> None:
    word = entry.get("word")
    part_of_speech = KAIKKI_POS.get(entry.get("pos"))
    if not word or not part_of_speech:
        return

    if word in targets:
        classifications[word].add(part_of_speech)

    for form in entry.get("forms", []):
        surface = form.get("form")
        if surface not in targets or surface == word:
            continue
        classifications[surface].add(part_of_speech)


def load_kaikki(
    path: Path,
    targets: set[str],
) -> dict[str, set[str]]:
    classifications: dict[str, set[str]] = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            if entry.get("lang_code") == "en":
                add_kaikki_entry(classifications, entry, targets)
    return dict(classifications)


def load_moby(
    path: Path,
    targets: set[str],
) -> dict[str, set[str]]:
    classifications: dict[str, set[str]] = defaultdict(set)
    for raw_line in path.read_bytes().splitlines():
        if b"\\" not in raw_line:
            continue
        word_bytes, codes_bytes = raw_line.rsplit(b"\\", 1)
        try:
            word = word_bytes.decode("cp1252")
            codes = codes_bytes.decode("ascii")
        except UnicodeDecodeError:
            continue
        if word not in targets:
            continue
        classifications[word].update(
            MOBY_POS[code] for code in codes if code in MOBY_POS
        )
    return {
        word: parts for word, parts in classifications.items() if parts
    }


def gcide_parts_of_speech(value: str) -> set[str]:
    value = MARKUP.sub("", html.unescape(value)).lower()
    parts = set()
    if "prop. n." in value:
        parts.add("proper_noun")
    if re.search(r"(?<!pro)(?<!pro\.)\bn\.", value):
        parts.add("noun")
    if any(token in value for token in ("v.", "vb.", "imp.", "p. p.", "p. pr.")):
        parts.add("verb")
    if re.search(r"(^|[ &])a\.", value) or any(
        token in value for token in ("adj.", "compar.", "superl.", "p. a.")
    ):
        parts.add("adjective")
    if "adv." in value:
        parts.add("adverb")
    if "interj." in value:
        parts.add("interjection")
    if "conj." in value:
        parts.add("conjunction")
    if "prep." in value:
        parts.add("preposition")
    if "pron." in value:
        parts.add("pronoun")
    return parts


def load_gcide(
    directory: Path,
    targets: set[str],
) -> dict[str, set[str]]:
    classifications: dict[str, set[str]] = defaultdict(set)
    for path in sorted(directory.glob("CIDE.*")):
        text = path.read_text(encoding="latin-1")
        for headword_markup, body in GCIDE_BLOCK.findall(text):
            headword = MARKUP.sub("", html.unescape(headword_markup)).strip().lower()
            if headword not in targets:
                continue
            for pos_markup in GCIDE_POS.findall(body):
                classifications[headword].update(
                    gcide_parts_of_speech(pos_markup)
                )
    return {
        word: parts for word, parts in classifications.items() if parts
    }


def load_qwen(
    path: Path,
    targets: set[str],
) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    classifications = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        word = record.get("word")
        parts = record.get("parts_of_speech")
        if word not in targets:
            raise ValueError(f"{path} contains unexpected word {word!r}")
        if (
            not isinstance(parts, list)
            or not parts
            or not all(isinstance(part, str) for part in parts)
        ):
            raise ValueError(f"{path} contains invalid POS labels for {word!r}")
        if word in classifications:
            raise ValueError(f"{path} contains duplicate word {word!r}")
        classifications[word] = set(parts)
    return classifications


def build_record(
    word: str,
    answers: set[str],
    classifications: dict[str, list[str]],
) -> dict[str, Any]:
    frequency = zipf_frequency(word, "en", wordlist="best")
    return {
        "word": word,
        "is_original_answer": word in answers,
        "zipf_frequency": frequency if frequency > 0 else None,
        "parts_of_speech": classifications.get(word, []),
    }


def main() -> None:
    args = parse_args()
    answers = read_words(ANSWERS_PATH)
    guesses = read_words(GUESSES_PATH)
    answer_set = set(answers)
    guess_set = set(guesses)

    if len(answers) != ANSWER_COUNT:
        raise ValueError(f"expected {ANSWER_COUNT} answers, found {len(answers)}")
    if len(guesses) != GUESS_COUNT:
        raise ValueError(f"expected {GUESS_COUNT} legal guesses, found {len(guesses)}")
    if not answer_set <= guess_set:
        missing = sorted(answer_set - guess_set)
        raise ValueError(f"legal guesses are missing answers: {missing[:5]}")

    source_metadata_path = args.cache_dir / "sources.json"
    if not source_metadata_path.exists():
        raise FileNotFoundError(
            "lexical sources are missing; run scripts/fetch_lexical_sources.py"
        )

    kaikki = load_kaikki(
        args.cache_dir / "kaikki.org-dictionary-English.jsonl.gz",
        guess_set,
    )
    unresolved = guess_set - set(kaikki)
    moby = load_moby(args.cache_dir / "mobypos.txt", unresolved)
    unresolved -= set(moby)
    gcide = load_gcide(args.cache_dir / "gcide-0.54", unresolved)
    unresolved -= set(gcide)
    qwen = load_qwen(QWEN_POS_PATH, unresolved)

    classifications: dict[str, list[str]] = {}
    classification_sources: dict[str, str] = {}
    for source, source_evidence in (
        ("kaikki", kaikki),
        ("moby_pos_ii", moby),
        ("gcide", gcide),
        ("omlx_qwen_38_27b", qwen),
    ):
        for word, parts in source_evidence.items():
            classifications[word] = sorted(parts)
            classification_sources[word] = source

    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    records = [
        build_record(word, answer_set, classifications)
        for word in sorted(guesses)
    ]
    for record in records:
        validator.validate(record)

    OUTPUT_PATH.write_text(
        "".join(
            f"{json.dumps(record, ensure_ascii=True, separators=(',', ':'))}\n"
            for record in records
        )
    )

    source_counts = Counter(
        classification_sources.values()
    )
    metadata = {
        "record_count": len(records),
        "answer_count": sum(record["is_original_answer"] for record in records),
        "parts_of_speech": {
            "source_precedence": [
                "kaikki",
                "moby_pos_ii",
                "gcide",
                "omlx_qwen_38_27b",
            ],
            "source_counts": dict(source_counts),
            "classified_count": len(classifications),
            "unclassified_count": len(guess_set - set(classifications)),
            "all_answers_classified": answer_set <= set(classifications),
            "sources": json.loads(source_metadata_path.read_text()),
            "qwen": (
                json.loads(QWEN_METADATA_PATH.read_text())
                if QWEN_METADATA_PATH.exists()
                else None
            ),
        },
        "wordfreq": {
            "package_version": importlib.metadata.version("wordfreq"),
            "language": "en",
            "wordlist": "best",
            "zero_frequency_representation": "null",
        },
        "schema": SCHEMA_PATH.name,
    }
    METADATA_PATH.write_text(f"{json.dumps(metadata, indent=2)}\n")
    UNCLASSIFIED_PATH.write_text(
        "".join(
            f"{word}\n" for word in sorted(guess_set - set(classifications))
        )
    )

    print(f"Wrote {len(records):,} records to {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Original answers: {metadata['answer_count']:,}")
    print(
        "POS coverage: "
        f"{len(classifications):,}/{len(guess_set):,} "
        f"({metadata['parts_of_speech']['unclassified_count']:,} unresolved)"
    )
    print(f"Source counts: {dict(source_counts)}")


if __name__ == "__main__":
    main()
