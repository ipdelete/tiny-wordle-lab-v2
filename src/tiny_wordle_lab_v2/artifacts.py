from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .evaluate import EvaluationResult
from .lexicon import Lexicon, ROOT


SCHEMA_VERSION = 1
EXPERIMENT_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")


@dataclass(frozen=True)
class RunTiming:
    started_at: str
    finished_at: str
    elapsed_seconds: float

    @classmethod
    def measure(cls, function):
        started = datetime.now(UTC)
        start = time.perf_counter()
        value = function()
        elapsed = time.perf_counter() - start
        finished = datetime.now(UTC)
        return value, cls(started.isoformat(), finished.isoformat(), elapsed)


def _run_git(repo_root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout


def git_provenance(repo_root: Path) -> dict[str, Any]:
    commit = _run_git(repo_root, "rev-parse", "HEAD").decode().strip()
    status = _run_git(repo_root, "status", "--porcelain=v1")
    dirty = bool(status)
    worktree_sha256 = None
    if dirty:
        diff = _run_git(repo_root, "diff", "--binary", "HEAD")
        untracked = _run_git(
            repo_root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ).split(b"\0")
        digest = hashlib.sha256(status + b"\0" + diff)
        for encoded_path in sorted(path for path in untracked if path):
            path = repo_root / encoded_path.decode("utf-8", errors="surrogateescape")
            digest.update(b"\0untracked\0")
            digest.update(encoded_path)
            digest.update(b"\0")
            digest.update(path.read_bytes())
        worktree_sha256 = digest.hexdigest()
    return {
        "commit": commit,
        "dirty": dirty,
        "worktree_sha256": worktree_sha256,
    }


def _game_payload(result: EvaluationResult) -> list[dict[str, Any]]:
    return [asdict(game) for game in result.games]


def _run_payload(
    result: EvaluationResult,
    lexicon: Lexicon,
    games_sha256: str,
    provenance: dict[str, Any],
    timing: RunTiming,
) -> dict[str, Any]:
    config = asdict(result.config)
    config["answers"] = (
        list(result.config.answers)
        if result.config.answers is not None
        else "all"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": result.config.experiment_id,
        "policy": asdict(result.policy),
        "config": config,
        "summary": asdict(result.summary),
        "execution": {
            **asdict(timing),
            "policy_calls_per_second": (
                result.summary.policy_calls / timing.elapsed_seconds
                if timing.elapsed_seconds
                else 0.0
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "git": provenance,
        "inputs": dict(lexicon.source_hashes),
        "games_artifact": {
            "path": "games.jsonl",
            "sha256": games_sha256,
            "records": len(result.games),
        },
    }


def _load_schema(repo_root: Path) -> dict[str, Any]:
    return json.loads(
        (repo_root / "schemas" / "experiment-result.schema.json").read_text()
    )


def load_run(path: Path, repo_root: Path = ROOT) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    Draft202012Validator(_load_schema(repo_root)).validate(payload)
    return payload


def write_artifacts(
    result: EvaluationResult,
    lexicon: Lexicon,
    output_root: Path,
    timing: RunTiming,
    repo_root: Path = ROOT,
) -> Path:
    experiment_id = result.config.experiment_id
    if not EXPERIMENT_ID.fullmatch(experiment_id):
        raise ValueError(
            "experiment_id may contain only letters, numbers, '.', '_', and '-'"
        )

    provenance = git_provenance(repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / experiment_id
    if destination.exists():
        raise FileExistsError(f"experiment output already exists: {destination}")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{experiment_id}-", dir=output_root)
    )
    try:
        games_path = temporary / "games.jsonl"
        digest = hashlib.sha256()
        with games_path.open("wb") as stream:
            for game in _game_payload(result):
                line = (
                    json.dumps(game, separators=(",", ":"), sort_keys=True) + "\n"
                ).encode()
                stream.write(line)
                digest.update(line)

        run_payload = _run_payload(
            result,
            lexicon,
            digest.hexdigest(),
            provenance,
            timing,
        )
        Draft202012Validator(_load_schema(repo_root)).validate(run_payload)
        (temporary / "run.json").write_text(
            json.dumps(run_payload, indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return destination
