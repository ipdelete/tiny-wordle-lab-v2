from __future__ import annotations

from pathlib import Path

from tiny_wordle_lab_v2.prompt import load_answer_file


def test_answers_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "answers.txt"
    path.write_text("# frozen\n\nCRANE\nlight\n")

    assert load_answer_file(path) == ("crane", "light")
