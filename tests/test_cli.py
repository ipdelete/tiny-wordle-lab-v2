from __future__ import annotations

from pathlib import Path

import pytest

from tiny_wordle_lab_v2 import cli
from tiny_wordle_lab_v2.lexicon import Lexicon


def test_cli_evaluate_and_compare(
    tmp_path: Path,
    toy_lexicon: Lexicon,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "load_lexicon", lambda: toy_lexicon)
    cli.main(
        [
            "evaluate",
            "--policy",
            "frequency",
            "--experiment-id",
            "cli-frequency",
            "--output-root",
            str(tmp_path),
        ]
    )
    run_path = tmp_path / "cli-frequency" / "run.json"
    assert run_path.exists()

    cli.main(["compare", str(run_path)])
    output = capsys.readouterr().out
    assert "cli-frequency" in output
    assert "policy=frequency" in output


def test_compare_rejects_different_answer_sets(monkeypatch) -> None:
    runs = {
        Path("first.json"): {
            "experiment_id": "first",
            "config": {"answers": ["apple"], "max_opportunities": 6},
            "inputs": {"words": "a"},
            "policy": {"name": "one"},
            "summary": {
                "solved": 1,
                "games": 1,
                "penalized_turns": 1,
                "illegal_actions": 0,
                "repeat_actions": 0,
            },
        },
        Path("second.json"): {
            "experiment_id": "second",
            "config": {"answers": ["banal"], "max_opportunities": 6},
            "inputs": {"words": "a"},
            "policy": {"name": "two"},
            "summary": {
                "solved": 1,
                "games": 1,
                "penalized_turns": 1,
                "illegal_actions": 0,
                "repeat_actions": 0,
            },
        },
    }
    monkeypatch.setattr(cli, "load_run", lambda path: runs[path])

    with pytest.raises(ValueError, match="answers differs"):
        cli._compare(list(runs))
