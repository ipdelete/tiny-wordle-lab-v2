from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from tiny_wordle_lab_v2 import artifacts
from tiny_wordle_lab_v2.artifacts import RunTiming, load_run, write_artifacts
from tiny_wordle_lab_v2.artifacts import git_provenance
from tiny_wordle_lab_v2.evaluate import EvaluationConfig, evaluate
from tiny_wordle_lab_v2.game import Observation
from tiny_wordle_lab_v2.lexicon import Lexicon
from tiny_wordle_lab_v2.policy import PolicyDescriptor


@dataclass
class AnswerPolicy:
    @property
    def descriptor(self) -> PolicyDescriptor:
        return PolicyDescriptor("answer", {})

    def choose(self, observation: Observation) -> str:
        return observation.candidates[0]


def test_artifacts_validate_and_hash_games(
    tmp_path: Path,
    repo_root: Path,
    toy_lexicon: Lexicon,
) -> None:
    result = evaluate(
        AnswerPolicy(),
        toy_lexicon,
        EvaluationConfig("artifact", answers=("apple",)),
    )
    destination = write_artifacts(
        result,
        toy_lexicon,
        tmp_path,
        RunTiming("start", "finish", 1.0),
        repo_root,
    )
    run = load_run(destination / "run.json", repo_root)
    games_bytes = (destination / "games.jsonl").read_bytes()
    assert run["games_artifact"]["sha256"] == hashlib.sha256(games_bytes).hexdigest()
    assert json.loads(games_bytes)["answer"] == "apple"

    with pytest.raises(FileExistsError):
        write_artifacts(
            result,
            toy_lexicon,
            tmp_path,
            RunTiming("start", "finish", 1.0),
            repo_root,
        )


def test_artifacts_reject_unsafe_experiment_id(
    tmp_path: Path,
    repo_root: Path,
    toy_lexicon: Lexicon,
) -> None:
    result = evaluate(
        AnswerPolicy(),
        toy_lexicon,
        EvaluationConfig("../escape", answers=("apple",)),
    )
    with pytest.raises(ValueError):
        write_artifacts(
            result,
            toy_lexicon,
            tmp_path,
            RunTiming("start", "finish", 1.0),
            repo_root,
        )


def test_git_provenance_hashes_untracked_contents(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("tracked\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "initial"],
        cwd=tmp_path,
        check=True,
    )
    untracked = tmp_path / "policy.py"
    untracked.write_text("first\n")
    first = git_provenance(tmp_path)["worktree_sha256"]
    untracked.write_text("second\n")
    second = git_provenance(tmp_path)["worktree_sha256"]
    assert first != second


def test_provenance_is_captured_before_output_creation(
    tmp_path: Path,
    repo_root: Path,
    toy_lexicon: Lexicon,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "custom-output"

    def capture_before_write(_repo_root: Path) -> dict:
        assert not output_root.exists()
        return {
            "commit": "0" * 40,
            "dirty": False,
            "worktree_sha256": None,
        }

    monkeypatch.setattr(artifacts, "git_provenance", capture_before_write)
    result = evaluate(
        AnswerPolicy(),
        toy_lexicon,
        EvaluationConfig("capture-order", answers=("apple",)),
    )
    write_artifacts(
        result,
        toy_lexicon,
        output_root,
        RunTiming("start", "finish", 1.0),
        repo_root,
    )
