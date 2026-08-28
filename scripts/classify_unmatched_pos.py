from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError

from tiny_wordle_lab_v2.litellm_policy import read_api_key


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "wordle-pos-unclassified.txt"
DEFAULT_OUTPUT = ROOT / "data" / "wordle-pos-qwen.jsonl"
DEFAULT_METADATA = ROOT / "data" / "wordle-pos-qwen.metadata.json"
DEFAULT_ENV_FILE = Path("~/src/wmd-router/.env").expanduser()
DEFAULT_GATEWAY_URL = "http://127.0.0.1:4000/v1/chat/completions"
MODEL = "omlx-qwen-38-27b"
PROMPT_VERSION = 2
PARTS_OF_SPEECH = (
    "adjective",
    "adverb",
    "conjunction",
    "contraction",
    "determiner",
    "interjection",
    "noun",
    "numeral",
    "particle",
    "postposition",
    "preposition",
    "prepositional_phrase",
    "pronoun",
    "proper_noun",
    "verb",
    "phrase",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--batch-size", type=int, default=10)
    return parser.parse_args()


def read_words(path: Path) -> list[str]:
    words = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(words) != len(set(words)):
        raise ValueError(f"{path} contains duplicate words")
    return words


def validate_record(record: dict[str, Any]) -> None:
    if set(record) != {"word", "parts_of_speech"}:
        raise ValueError(f"unexpected fields for {record.get('word')!r}")
    parts = record["parts_of_speech"]
    if not isinstance(parts, list) or not parts or len(parts) != len(set(parts)):
        raise ValueError(f"invalid parts of speech for {record['word']!r}")
    if any(part not in PARTS_OF_SPEECH for part in parts):
        raise ValueError(f"unknown part of speech for {record['word']!r}")


def read_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        validate_record(record)
        word = record["word"]
        if word in completed:
            raise ValueError(f"{path} contains duplicate word {word!r}")
        completed[word] = record
    return completed


def response_format(words: list[str]) -> dict[str, Any]:
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["parts_of_speech"],
        "properties": {
            "parts_of_speech": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "enum": list(PARTS_OF_SPEECH)},
            },
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pos_classifications",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["classifications"],
                "properties": {
                    "classifications": {
                        "type": "array",
                        "minItems": len(words),
                        "maxItems": len(words),
                        "items": item,
                    }
                },
            },
        },
    }


def classify_batch(
    words: list[str],
    *,
    api_key: str,
    gateway_url: str,
) -> list[dict[str, Any]]:
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 2_048,
        "seed": 38,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
            "thinking_budget": 0,
        },
        "response_format": response_format(words),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful lexicographer classifying obscure English "
                    "word-game entries. The words may be archaic, dialectal, Scots, "
                    "inflected, borrowed, or proper names. Return established lexical "
                    "roles, not roles invented from spelling. Use multiple labels only "
                    "when the word genuinely has multiple established roles. Classify "
                    "every word and output only the requested structure."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Classify these words: {', '.join(words)}. Return exactly one "
                    "classification for each word, in the same order."
                ),
            },
        ],
    }
    client = OpenAI(
        api_key=api_key,
        base_url=gateway_url.removesuffix("/chat/completions"),
        timeout=180,
        max_retries=0,
    )
    max_attempts = 12
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(**payload)
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("model returned no content")
            classifications = json.loads(content)["classifications"]
            records = [
                {
                    "word": word,
                    "parts_of_speech": sorted(item["parts_of_speech"]),
                }
                for word, item in zip(words, classifications, strict=True)
            ]
            if len(records) != len(words):
                raise ValueError("model returned the wrong number of classifications")
            for record in records:
                validate_record(record)
            return sorted(records, key=lambda record: record["word"])
        except (
            KeyError,
            json.JSONDecodeError,
            OpenAIError,
            ValueError,
        ):
            if attempt == max_attempts:
                raise
            time.sleep(min(5 * attempt, 30))

    raise AssertionError("unreachable")


def append_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("a") as stream:
        for record in records:
            stream.write(f"{json.dumps(record, separators=(',', ':'))}\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_metadata(path: Path, *, completed: int, total: int) -> None:
    metadata = {
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "temperature": 0,
        "seed": 38,
        "thinking": False,
        "classified_count": completed,
        "input_count": total,
    }
    path.write_text(f"{json.dumps(metadata, indent=2)}\n")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    words = read_words(args.input)
    completed = read_completed(args.output)
    unexpected = set(completed) - set(words)
    if unexpected:
        raise ValueError(f"output contains unexpected words: {sorted(unexpected)[:5]}")

    pending = [word for word in words if word not in completed]
    if not pending:
        write_metadata(args.metadata, completed=len(completed), total=len(words))
        print(f"All {len(words):,} words are classified")
        return

    api_key = read_api_key(args.env_file)
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        records = classify_batch(
            batch,
            api_key=api_key,
            gateway_url=args.gateway_url,
        )
        append_records(args.output, records)
        completed.update({record["word"]: record for record in records})
        write_metadata(args.metadata, completed=len(completed), total=len(words))
        print(f"Classified {len(completed):,}/{len(words):,}", flush=True)


if __name__ == "__main__":
    main()
