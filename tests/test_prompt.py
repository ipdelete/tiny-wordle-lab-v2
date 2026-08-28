from __future__ import annotations

import hashlib
from pathlib import Path

from tiny_wordle_lab_v2.prompt import WordlePrompt, load_answer_file


def test_prompt_normalizes_trailing_newline_and_hashes_content(tmp_path: Path) -> None:
    path = tmp_path / "prompt.md"
    path.write_text("Play Wordle.  \n\n")

    prompt = WordlePrompt.from_path(path)

    assert prompt.content == "Play Wordle.\n"
    assert prompt.sha256 == hashlib.sha256(b"Play Wordle.\n").hexdigest()
    assert prompt.source == str(path.resolve())


def test_answers_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "answers.txt"
    path.write_text("# frozen\n\nCRANE\nlight\n")

    assert load_answer_file(path) == ("crane", "light")
