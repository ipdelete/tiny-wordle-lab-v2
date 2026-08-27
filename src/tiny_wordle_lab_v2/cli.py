from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .artifacts import RunTiming, load_run, write_artifacts
from .baselines import EntropyPolicy, FrequencyPolicy, RandomPolicy
from .evaluate import EvaluationConfig, evaluate
from .lexicon import ROOT, Lexicon, load_lexicon
from .litellm_policy import (
    DEFAULT_ENV_FILE,
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
    OpenAIWordlePolicy,
    read_api_key,
)
from .policy import Policy
from .prompt import DEFAULT_PROMPT_PATH, WordlePrompt


POLICIES = (
    "random",
    "frequency",
    "candidate-entropy",
    "open-entropy",
    "litellm",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tiny-wordle-lab-v2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--policy", choices=POLICIES, required=True)
    evaluate_parser.add_argument("--experiment-id", required=True)
    evaluate_parser.add_argument("--seed", type=int, default=0)
    evaluate_parser.add_argument("--model", default=DEFAULT_MODEL)
    evaluate_parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    evaluate_parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    evaluate_parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    evaluate_parser.add_argument(
        "--answers",
        help="Comma-separated answer subset; defaults to all 2,315 answers",
    )
    evaluate_parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results",
    )

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("runs", type=Path, nargs="+")
    return parser


def _make_policy(
    name: str,
    lexicon: Lexicon,
    seed: int,
    *,
    model: str = DEFAULT_MODEL,
    api_base: str = DEFAULT_API_BASE,
    env_file: Path = DEFAULT_ENV_FILE,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
) -> Policy:
    if name == "random":
        return RandomPolicy(lexicon, seed=seed)
    if name == "frequency":
        return FrequencyPolicy(lexicon)
    if name == "candidate-entropy":
        return EntropyPolicy(lexicon, candidate_only=True)
    if name == "open-entropy":
        return EntropyPolicy(lexicon, candidate_only=False)
    if name == "litellm":
        return OpenAIWordlePolicy(
            api_key=read_api_key(env_file),
            context_mode="snapshot",
            model=model,
            api_base=api_base,
            seed=seed,
            prompt=WordlePrompt.from_path(prompt_path),
        )
    raise ValueError(f"unknown policy: {name}")


def _print_summary(run: dict) -> None:
    summary = run["summary"]
    print(
        f"{run['experiment_id']}: policy={run['policy']['name']} "
        f"solved={summary['solved']}/{summary['games']} "
        f"penalized_turns={summary['penalized_turns']} "
        f"illegal={summary['illegal_actions']} "
        f"repeats={summary['repeat_actions']}"
    )


def _compare(paths: Sequence[Path]) -> None:
    runs = [load_run(path) for path in paths]
    reference = runs[0]
    for run in runs[1:]:
        for field, expected, actual in (
            ("answers", reference["config"]["answers"], run["config"]["answers"]),
            (
                "max_opportunities",
                reference["config"]["max_opportunities"],
                run["config"]["max_opportunities"],
            ),
            ("inputs", reference["inputs"], run["inputs"]),
        ):
            if actual != expected:
                raise ValueError(
                    f"cannot compare {run['experiment_id']} with "
                    f"{reference['experiment_id']}: {field} differs"
                )
    runs.sort(
        key=lambda run: (
            -run["summary"]["solved"],
            run["summary"]["penalized_turns"],
            run["experiment_id"],
        )
    )
    for run in runs:
        _print_summary(run)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "compare":
        _compare(arguments.runs)
        return

    lexicon = load_lexicon()
    answers = (
        tuple(word.strip().lower() for word in arguments.answers.split(","))
        if arguments.answers
        else None
    )
    result, timing = RunTiming.measure(
        lambda: evaluate(
            _make_policy(
                arguments.policy,
                lexicon,
                arguments.seed,
                model=arguments.model,
                api_base=arguments.api_base,
                env_file=arguments.env_file,
                prompt_path=arguments.prompt,
            ),
            lexicon,
            EvaluationConfig(
                experiment_id=arguments.experiment_id,
                answers=answers,
            ),
        )
    )
    destination = write_artifacts(
        result,
        lexicon,
        output_root=arguments.output_root,
        timing=timing,
    )
    _print_summary(load_run(destination / "run.json"))
    print(destination)
