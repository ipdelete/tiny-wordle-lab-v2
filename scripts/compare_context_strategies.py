from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_wordle_lab_v2.artifacts import RunTiming, write_artifacts
from tiny_wordle_lab_v2.evaluate import EvaluationConfig, EvaluationResult, evaluate
from tiny_wordle_lab_v2.game import Turn, is_consistent
from tiny_wordle_lab_v2.lexicon import ROOT, Lexicon, load_lexicon
from tiny_wordle_lab_v2.litellm_policy import (
    AgentsSessionWordlePolicy,
    OpenAIWordlePolicy,
    read_api_key,
)


FROZEN_ANSWERS = (
    "foyer",
    "banal",
    "sissy",
    "apple",
    "alley",
    "sheep",
    "civic",
    "queue",
    "jazzy",
    "glyph",
    "crane",
    "light",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answers",
        default=",".join(FROZEN_ANSWERS),
        help="Comma-separated answer list",
    )
    parser.add_argument("--suffix", default="v1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    return parser.parse_args()


def consistency(result: EvaluationResult) -> dict[str, int | float | None]:
    checked = 0
    consistent = 0
    for game in result.games:
        history: list[Turn] = []
        for action in game.actions:
            if action.guess is None or action.feedback is None:
                continue
            if history:
                checked += 1
                consistent += int(is_consistent(action.guess, history))
            history.append(Turn(action.guess, action.feedback))
    return {
        "checked": checked,
        "consistent": consistent,
        "rate": consistent / checked if checked else None,
    }


def run_arm(
    *,
    name: str,
    policy,
    lexicon: Lexicon,
    answers: tuple[str, ...],
    suffix: str,
    output_root: Path,
) -> tuple[EvaluationResult, dict]:
    experiment_id = f"context-{name}-{suffix}"
    result, timing = RunTiming.measure(
        lambda: evaluate(
            policy,
            lexicon,
            EvaluationConfig(experiment_id=experiment_id, answers=answers),
        )
    )
    destination = write_artifacts(
        result,
        lexicon,
        output_root,
        timing,
    )
    report = {
        "arm": name,
        "experiment_id": experiment_id,
        "summary": {
            "solved": result.summary.solved,
            "games": result.summary.games,
            "penalized_turns": result.summary.penalized_turns,
            "illegal_actions": result.summary.illegal_actions,
            "repeat_actions": result.summary.repeat_actions,
        },
        "consistency": consistency(result),
        "usage": policy.usage.as_dict(),
        "elapsed_seconds": timing.elapsed_seconds,
        "artifact": str(destination),
    }
    print(json.dumps(report, indent=2), flush=True)
    return result, report


def paired_results(
    results: dict[str, EvaluationResult],
    answers: tuple[str, ...],
) -> list[dict]:
    by_arm = {
        arm: {game.answer: game for game in result.games}
        for arm, result in results.items()
    }
    return [
        {
            "answer": answer,
            **{
                arm: {
                    "solved": games[answer].solved,
                    "opportunities": games[answer].opportunities_used,
                    "illegal": sum(
                        action.status == "illegal"
                        for action in games[answer].actions
                    ),
                }
                for arm, games in by_arm.items()
            },
        }
        for answer in answers
    ]


def main() -> None:
    arguments = parse_args()
    answers = tuple(
        answer.strip().lower()
        for answer in arguments.answers.split(",")
        if answer.strip()
    )
    lexicon = load_lexicon()
    unknown = set(answers) - set(lexicon.answers)
    if unknown:
        raise ValueError(f"unknown answers: {sorted(unknown)}")
    api_key = read_api_key()

    policies = {
        "snapshot": OpenAIWordlePolicy(
            api_key=api_key,
            context_mode="snapshot",
        ),
        "transcript": OpenAIWordlePolicy(
            api_key=api_key,
            context_mode="transcript",
        ),
        "agents": AgentsSessionWordlePolicy(api_key=api_key),
    }
    results: dict[str, EvaluationResult] = {}
    reports = []
    for name, policy in policies.items():
        result, report = run_arm(
            name=name,
            policy=policy,
            lexicon=lexicon,
            answers=answers,
            suffix=arguments.suffix,
            output_root=arguments.output_root,
        )
        results[name] = result
        reports.append(report)

    comparison = {
        "answers": list(answers),
        "arms": reports,
        "paired_results": paired_results(results, answers),
    }
    path = arguments.output_root / f"context-comparison-{arguments.suffix}.json"
    path.write_text(json.dumps(comparison, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
