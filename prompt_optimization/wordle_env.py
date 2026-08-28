from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from skillopt.datasets.base import BatchSpec, SplitDataLoader
from skillopt.envs.base import EnvAdapter

from tiny_wordle_lab_v2.evaluate import EvaluationConfig, GameResult, evaluate
from tiny_wordle_lab_v2.lexicon import load_lexicon
from tiny_wordle_lab_v2.litellm_policy import (
    DEFAULT_API_BASE,
    DEFAULT_ENV_FILE,
    DEFAULT_MODEL,
    INITIAL_USER_MESSAGE,
    OpenAIWordlePolicy,
    read_api_key,
)
from tiny_wordle_lab_v2.prompt import WordlePrompt
from tiny_wordle_lab_v2.prompt_scoring import (
    lexicographic_game_score,
    penalized_turns,
)


class WordleDataLoader(SplitDataLoader):
    def load_split_items(self, split_path: str) -> list[dict]:
        path = Path(split_path) / "answers.txt"
        if not path.is_file():
            raise FileNotFoundError(f"missing Wordle answer split: {path}")
        answers = [
            line.strip().lower()
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        return [
            {"id": answer, "answer": answer, "task_type": "wordle"}
            for answer in answers
        ]


def _conversation(
    game: GameResult,
    *,
    prompt: WordlePrompt,
    repeat: int,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": prompt.content},
        {"role": "user", "content": f"Repeat {repeat}: {INITIAL_USER_MESSAGE}"},
    ]
    for action in game.actions:
        messages.append({"role": "assistant", "content": action.raw_output})
        if action.status == "illegal":
            feedback = (
                "Harness: that response was not a legal Wordle guess. "
                "It consumed an opportunity and produced no feedback."
            )
        else:
            feedback = (
                f"Harness: {action.guess.upper()} produced {action.feedback}. "
                f"Status: {action.status}."
            )
        messages.append({"role": "user", "content": feedback})
    messages.append(
        {
            "role": "system",
            "content": (
                f"Evaluation: the hidden answer was {game.answer.upper()}. "
                f"Solved: {game.solved}. Penalized turns: "
                f"{penalized_turns(game)}."
            ),
        }
    )
    return messages


class WordleAdapter(EnvAdapter):
    def __init__(
        self,
        split_dir: str,
        split_mode: str = "split_dir",
        split_seed: int = 42,
        seed: int = 42,
        limit: int = 0,
        analyst_workers: int = 1,
        failure_only: bool = False,
        minibatch_size: int = 4,
        edit_budget: int = 2,
        rollout_repeats: int = 2,
        api_base: str = DEFAULT_API_BASE,
        env_file: str = str(DEFAULT_ENV_FILE),
        target_model: str = DEFAULT_MODEL,
        target_temperature: float = 0,
        target_reasoning_effort: str = "low",
        target_max_tokens: int = 2_048,
        target_timeout_seconds: float = 180,
    ) -> None:
        if rollout_repeats < 1:
            raise ValueError("rollout_repeats must be positive")
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.rollout_repeats = rollout_repeats
        self.api_base = api_base
        self.env_file = Path(env_file).expanduser()
        self.target_model = target_model
        self.target_temperature = target_temperature
        self.target_reasoning_effort = target_reasoning_effort
        self.target_max_tokens = target_max_tokens
        self.target_timeout_seconds = target_timeout_seconds
        self.seed = seed
        self.dataloader = WordleDataLoader(
            split_dir=split_dir,
            split_mode=split_mode,
            split_seed=split_seed,
            seed=seed,
            limit=limit,
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self) -> WordleDataLoader:
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs) -> list[dict]:
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs) -> list[dict]:
        batch = self.dataloader.build_train_batch(
            batch_size=batch_size,
            seed=seed,
            **kwargs,
        )
        return self.build_env_from_batch(batch)

    def build_eval_env(
        self,
        env_num: int,
        split: str,
        seed: int,
        **kwargs,
    ) -> list[dict]:
        batch = self.dataloader.build_eval_batch(
            env_num=env_num,
            split=split,
            seed=seed,
            **kwargs,
        )
        return self.build_env_from_batch(batch)

    def rollout(
        self,
        env_manager: list[dict],
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict]:
        del kwargs
        output_root = Path(out_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        rollouts_path = output_root / "rollouts.json"
        if rollouts_path.is_file():
            return json.loads(rollouts_path.read_text())

        answers = tuple(item["answer"] for item in env_manager)
        if not answers:
            return []
        prompt = WordlePrompt.from_text(
            skill_content,
            source="skillopt-candidate",
        )
        api_key = read_api_key(self.env_file)
        lexicon = load_lexicon()
        repeated_games: dict[str, list[GameResult]] = {
            answer: [] for answer in answers
        }
        total_games = len(answers) * self.rollout_repeats

        for repeat in range(1, self.rollout_repeats + 1):
            policy = OpenAIWordlePolicy(
                api_key=api_key,
                model=self.target_model,
                api_base=self.api_base,
                temperature=self.target_temperature,
                seed=self.seed + repeat - 1,
                reasoning_effort=self.target_reasoning_effort,
                max_tokens=self.target_max_tokens,
                timeout_seconds=self.target_timeout_seconds,
                prompt=prompt,
            )
            evaluation = evaluate(
                policy,
                lexicon,
                EvaluationConfig(
                    experiment_id=f"skillopt-{prompt.sha256[:12]}-r{repeat}",
                    answers=answers,
                ),
            )
            for game in evaluation.games:
                repeated_games[game.answer].append(game)

        results = []
        for answer in answers:
            games = repeated_games[answer]
            hard = sum(game.solved for game in games) / len(games)
            soft = sum(
                lexicographic_game_score(game, total_games=total_games)
                for game in games
            ) / len(games)
            failures = [
                (
                    f"repeat {index}: solved={game.solved}, "
                    f"penalized_turns={penalized_turns(game)}, "
                    f"actions={[action.raw_output for action in game.actions]}"
                )
                for index, game in enumerate(games, start=1)
                if not game.solved
            ]
            prediction_dir = output_root / "predictions" / answer
            prediction_dir.mkdir(parents=True, exist_ok=True)
            conversation = []
            for repeat, game in enumerate(games, start=1):
                conversation.extend(
                    _conversation(game, prompt=prompt, repeat=repeat)
                )
            (prediction_dir / "conversation.json").write_text(
                json.dumps(conversation, indent=2) + "\n"
            )
            result = {
                "id": answer,
                "hard": hard,
                "soft": soft,
                "answer": answer,
                "task_type": "wordle",
                "task_description": (
                    "Play one complete Wordle game using only public feedback."
                ),
                "reference_text": f"The hidden answer was {answer.upper()}.",
                "target_system_prompt": prompt.content,
                "target_user_prompt": INITIAL_USER_MESSAGE,
                "n_turns": sum(game.policy_calls for game in games),
                "fail_reason": "\n".join(failures),
                "repeats": [asdict(game) for game in games],
                "score_contract": (
                    "Solve count dominates; penalized turns break ties."
                ),
            }
            results.append(result)

        rollouts_path.write_text(json.dumps(results, indent=2) + "\n")
        return results

    def get_task_types(self) -> list[str]:
        return ["wordle"]
