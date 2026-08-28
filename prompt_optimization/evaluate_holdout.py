from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from tiny_wordle_lab_v2.evaluate import EvaluationConfig, evaluate
from tiny_wordle_lab_v2.lexicon import load_lexicon
from tiny_wordle_lab_v2.litellm_policy import (
    DEFAULT_API_BASE,
    DEFAULT_ENV_FILE,
    DEFAULT_MODEL,
    OpenAIWordlePolicy,
    read_api_key,
)
from tiny_wordle_lab_v2.prompt import WordlePrompt, load_answer_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive")
    if arguments.output.exists():
        raise FileExistsError(arguments.output)

    prompt = WordlePrompt.from_path(arguments.prompt)
    answers = load_answer_file(arguments.answers)
    api_key = read_api_key(arguments.env_file.expanduser())
    lexicon = load_lexicon()
    runs = []
    for repeat in range(1, arguments.repeats + 1):
        policy = OpenAIWordlePolicy(
            api_key=api_key,
            context_mode="snapshot",
            model=arguments.model,
            api_base=arguments.api_base,
            temperature=0,
            seed=20260827 + repeat - 1,
            reasoning_effort="low",
            max_tokens=2_048,
            prompt=prompt,
        )
        result = evaluate(
            policy,
            lexicon,
            EvaluationConfig(
                experiment_id=f"final-holdout-r{repeat}",
                answers=answers,
            ),
        )
        runs.append(result)

    games = [game for result in runs for game in result.games]
    payload = {
        "schema_version": 1,
        "prompt": {
            "path": str(arguments.prompt),
            "sha256": prompt.sha256,
        },
        "answer_split": {
            "path": str(arguments.answers),
            "sha256": hashlib.sha256(arguments.answers.read_bytes()).hexdigest(),
            "answers": len(answers),
        },
        "model": arguments.model,
        "repeats": arguments.repeats,
        "games": len(games),
        "solved": sum(game.solved for game in games),
        "penalized_turns": sum(
            game.opportunities_used if game.solved else 7 for game in games
        ),
        "runs": [
            {
                "repeat": repeat,
                "summary": asdict(result.summary),
            }
            for repeat, result in enumerate(runs, start=1)
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(arguments.output)


if __name__ == "__main__":
    main()
