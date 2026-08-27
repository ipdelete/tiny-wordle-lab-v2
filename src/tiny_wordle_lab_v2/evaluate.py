from __future__ import annotations

from dataclasses import dataclass

from .game import (
    SOLVED_FEEDBACK,
    Observation,
    Turn,
    filter_candidates,
    parse_word,
    score_guess,
)
from .lexicon import Lexicon
from .policy import Policy, PolicyDescriptor


@dataclass(frozen=True)
class EvaluationConfig:
    experiment_id: str
    answers: tuple[str, ...] | None = None
    max_opportunities: int = 6


@dataclass(frozen=True)
class ActionResult:
    opportunity: int
    raw_output: str
    guess: str | None
    status: str
    feedback: str | None


@dataclass(frozen=True)
class GameResult:
    answer: str
    solved: bool
    policy_calls: int
    opportunities_used: int
    accepted_guesses: tuple[str, ...]
    actions: tuple[ActionResult, ...]


@dataclass(frozen=True)
class EvaluationSummary:
    games: int
    solved: int
    failed: int
    solve_rate: float
    penalized_turns: int
    mean_turns_on_wins: float | None
    policy_calls: int
    legal_actions: int
    illegal_actions: int
    repeat_actions: int


@dataclass(frozen=True)
class EvaluationResult:
    config: EvaluationConfig
    policy: PolicyDescriptor
    summary: EvaluationSummary
    games: tuple[GameResult, ...]


def _validate_config(config: EvaluationConfig, lexicon: Lexicon) -> tuple[str, ...]:
    if not config.experiment_id:
        raise ValueError("experiment_id must not be empty")
    if config.max_opportunities < 1:
        raise ValueError("max_opportunities must be positive")
    answers = config.answers if config.answers is not None else lexicon.answers
    if not answers:
        raise ValueError("evaluation answer set must not be empty")
    unknown = set(answers) - set(lexicon.answers)
    if unknown:
        raise ValueError(f"unknown evaluation answers: {sorted(unknown)[:5]}")
    if len(answers) != len(set(answers)):
        raise ValueError("evaluation answers contain duplicates")
    return answers


def _play_game(
    answer: str,
    policy: Policy,
    lexicon: Lexicon,
    max_opportunities: int,
) -> GameResult:
    history: list[Turn] = []
    previous_actions: list[str] = []
    candidates = lexicon.answers
    legal_guesses = set(lexicon.legal_guesses)
    accepted: set[str] = set()
    actions: list[ActionResult] = []

    for opportunity in range(1, max_opportunities + 1):
        observation = Observation(
            history=tuple(history),
            previous_actions=tuple(previous_actions),
            remaining_opportunities=max_opportunities - opportunity + 1,
            candidates=candidates,
        )
        raw_output = policy.choose(observation)
        if not isinstance(raw_output, str):
            raise TypeError("Policy.choose() must return a string")
        previous_actions.append(raw_output)
        guess = parse_word(raw_output)

        if guess is None or guess not in legal_guesses:
            actions.append(
                ActionResult(
                    opportunity=opportunity,
                    raw_output=raw_output,
                    guess=guess,
                    status="illegal",
                    feedback=None,
                )
            )
            continue

        feedback = score_guess(answer, guess)
        repeated = guess in accepted
        accepted.add(guess)
        history.append(Turn(guess=guess, feedback=feedback))
        candidates = filter_candidates(candidates, guess, feedback)
        if answer not in candidates:
            raise AssertionError("candidate filtering removed the hidden answer")

        status = "solved" if feedback == SOLVED_FEEDBACK else (
            "repeat" if repeated else "accepted"
        )
        actions.append(
            ActionResult(
                opportunity=opportunity,
                raw_output=raw_output,
                guess=guess,
                status=status,
                feedback=feedback,
            )
        )
        if feedback == SOLVED_FEEDBACK:
            return GameResult(
                answer=answer,
                solved=True,
                policy_calls=opportunity,
                opportunities_used=opportunity,
                accepted_guesses=tuple(turn.guess for turn in history),
                actions=tuple(actions),
            )

    return GameResult(
        answer=answer,
        solved=False,
        policy_calls=max_opportunities,
        opportunities_used=max_opportunities,
        accepted_guesses=tuple(turn.guess for turn in history),
        actions=tuple(actions),
    )


def evaluate(
    policy: Policy,
    lexicon: Lexicon,
    config: EvaluationConfig,
) -> EvaluationResult:
    answers = _validate_config(config, lexicon)
    games = tuple(
        _play_game(answer, policy, lexicon, config.max_opportunities)
        for answer in answers
    )

    solved_games = tuple(game for game in games if game.solved)
    solved = len(solved_games)
    policy_calls = sum(game.policy_calls for game in games)
    legal_actions = sum(
        action.guess is not None and action.feedback is not None
        for game in games
        for action in game.actions
    )
    illegal_actions = sum(
        action.status == "illegal" for game in games for action in game.actions
    )
    repeat_actions = sum(
        action.status == "repeat" for game in games for action in game.actions
    )
    penalized_turns = sum(
        game.opportunities_used if game.solved else 7
        for game in games
    )
    mean_turns_on_wins = (
        sum(game.opportunities_used for game in solved_games) / solved
        if solved
        else None
    )
    return EvaluationResult(
        config=config,
        policy=policy.descriptor,
        summary=EvaluationSummary(
            games=len(games),
            solved=solved,
            failed=len(games) - solved,
            solve_rate=solved / len(games),
            penalized_turns=penalized_turns,
            mean_turns_on_wins=mean_turns_on_wins,
            policy_calls=policy_calls,
            legal_actions=legal_actions,
            illegal_actions=illegal_actions,
            repeat_actions=repeat_actions,
        ),
        games=games,
    )
