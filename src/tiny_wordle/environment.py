from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .expert import EntropyExpert
from .game import Turn, score_string
from .representation import (
    REPRESENTATION_VERSION,
    render_state_key,
    structured_next_guess_prompt,
)


@dataclass(frozen=True)
class EnvironmentConfig:
    answers: tuple[str, ...]
    opening: str | None = "RAISE"
    max_turns: int = 6
    policy_view: str = REPRESENTATION_VERSION

    def __post_init__(self) -> None:
        answers = tuple(self.answers)
        if not answers:
            raise ValueError("answers must not be empty")
        if len(set(answers)) != len(answers):
            raise ValueError("answers must be unique")
        if any(
            not isinstance(word, str)
            or len(word) != 5
            or word != word.upper()
            or not word.isascii()
            or not word.isalpha()
            for word in answers
        ):
            raise ValueError("answers must be uppercase ASCII five-letter words")
        object.__setattr__(self, "answers", answers)
        if self.opening is not None and self.opening not in answers:
            raise ValueError("opening must be present in answers")
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        if self.policy_view != REPRESENTATION_VERSION:
            raise ValueError(
                f"unsupported policy view: {self.policy_view}"
            )
@dataclass(frozen=True)
class Observation:
    history: tuple[Turn, ...]
    turn: int
    remaining_turns: int
    candidate_count: int
    state_key: str
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": [
                {"guess": turn.guess, "feedback": turn.feedback}
                for turn in self.history
            ],
            "turn": self.turn,
            "remaining_turns": self.remaining_turns,
            "candidate_count": self.candidate_count,
            "state_key": self.state_key,
            "prompt": self.prompt,
        }


@dataclass(frozen=True)
class TeacherDiagnostics:
    candidate_teacher_guess: str | None
    candidate_teacher_entropy_bits: float | None
    action_entropy_bits: float | None
    action_is_candidate: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_teacher_guess": self.candidate_teacher_guess,
            "candidate_teacher_entropy_bits": (
                self.candidate_teacher_entropy_bits
            ),
            "action_entropy_bits": self.action_entropy_bits,
            "action_is_candidate": self.action_is_candidate,
        }


@dataclass(frozen=True)
class StepRecord:
    episode_id: str
    turn: int
    observation_before: Observation
    action: Any
    feedback: str | None
    reward: float
    done: bool
    terminal_reason: str | None
    repeated: bool
    candidate_count_before: int
    candidate_count_after: int
    observation_after: Observation
    teacher_diagnostics: TeacherDiagnostics

    @property
    def history_before(self) -> tuple[Turn, ...]:
        return self.observation_before.history

    @property
    def history_after(self) -> tuple[Turn, ...]:
        return self.observation_after.history

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "turn": self.turn,
            "observation_before": self.observation_before.to_dict(),
            "action": self.action,
            "feedback": self.feedback,
            "reward": self.reward,
            "done": self.done,
            "terminal_reason": self.terminal_reason,
            "repeated": self.repeated,
            "candidate_count_before": self.candidate_count_before,
            "candidate_count_after": self.candidate_count_after,
            "observation_after": self.observation_after.to_dict(),
            "teacher_diagnostics": self.teacher_diagnostics.to_dict(),
        }


@dataclass(frozen=True)
class ResetResult:
    observation: Observation
    reward: float
    done: bool
    info: Mapping[str, Any]
    opening_record: StepRecord | None


@dataclass(frozen=True)
class EpisodeTrace:
    episode_id: str
    answer_id: str
    seed: int | None
    records: tuple[StepRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "answer_id": self.answer_id,
            "seed": self.seed,
            "records": [record.to_dict() for record in self.records],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


class WordleEnvironment:
    def __init__(
        self,
        config: EnvironmentConfig,
        *,
        expert: EntropyExpert | None = None,
        patterns: np.ndarray | None = None,
    ) -> None:
        self.config = config
        if expert is None:
            if patterns is None:
                raise ValueError("patterns are required when expert is omitted")
            expert = EntropyExpert(list(config.answers), np.asarray(patterns))
        if tuple(expert.answers) != config.answers:
            raise ValueError("expert answers do not match environment answers")
        if patterns is not None and not np.array_equal(
            expert.patterns,
            np.asarray(patterns),
        ):
            raise ValueError("expert and patterns disagree")
        if expert.patterns.shape != (len(config.answers), len(config.answers)):
            raise ValueError("patterns must be square and match answers")

        self._expert = expert
        self._answer_set = set(config.answers)
        self._word_to_index = expert.word_to_index
        self._initial_candidates = expert.all_indices.copy()
        self._answer: str | None = None
        self._episode_id: str | None = None
        self._seed: int | None = None
        self._history: list[Turn] = []
        self._candidate_indices = self._initial_candidates.copy()
        self._turn_count = 0
        self._done = False
        self._records: list[StepRecord] = []

    @property
    def done(self) -> bool:
        return self._done

    @property
    def answer_id(self) -> str | None:
        return self._answer

    @property
    def observation(self) -> Observation:
        self._require_reset()
        return self._observation()

    @property
    def trace(self) -> EpisodeTrace:
        self._require_reset()
        return EpisodeTrace(
            episode_id=self._episode_id,
            answer_id=self._answer,
            seed=self._seed,
            records=tuple(self._records),
        )

    @property
    def last_record(self) -> StepRecord:
        self._require_reset()
        if not self._records:
            raise RuntimeError("episode has no transitions")
        return self._records[-1]

    def reset(
        self,
        answer: str,
        *,
        episode_id: str = "episode",
        seed: int | None = None,
    ) -> ResetResult:
        if self._answer is not None and not self._done:
            raise RuntimeError("cannot reset a live episode")
        if not isinstance(answer, str) or answer not in self._answer_set:
            raise ValueError("answer must be an uppercase word in answers")

        self._answer = answer
        self._episode_id = str(episode_id)
        self._seed = seed
        self._history = []
        self._candidate_indices = self._initial_candidates.copy()
        self._turn_count = 0
        self._done = False
        self._records = []

        if self.config.opening is None:
            observation = self._observation()
            return ResetResult(
                observation=observation,
                reward=0.0,
                done=False,
                info=MappingProxyType(
                    {
                        "episode_id": self._episode_id,
                        "turn": 0,
                        "terminal_reason": None,
                    }
                ),
                opening_record=None,
            )

        opening_record, info = self._take_action(self.config.opening)
        self._records.append(opening_record)
        return ResetResult(
            observation=opening_record.observation_after,
            reward=opening_record.reward,
            done=opening_record.done,
            info=info,
            opening_record=opening_record,
        )

    def step(
        self,
        action: Any,
    ) -> tuple[Observation, float, bool, Mapping[str, Any]]:
        self._require_reset()
        if self._done:
            raise RuntimeError("cannot step a terminated episode")
        record, info = self._take_action(action)
        self._records.append(record)
        return record.observation_after, record.reward, record.done, info

    def _take_action(
        self,
        action: Any,
    ) -> tuple[StepRecord, Mapping[str, Any]]:
        self._require_reset()
        before = self._observation()
        candidate_count_before = len(self._candidate_indices)
        valid = self._is_valid_action(action)
        teacher_diagnostics = self._teacher_diagnostics(
            action if valid else None
        )
        if not valid:
            self._done = True
            after = self._observation()
            record = StepRecord(
                episode_id=self._episode_id,
                turn=self._turn_count + 1,
                observation_before=before,
                action=action,
                feedback=None,
                reward=0.0,
                done=True,
                terminal_reason="contract_violation",
                repeated=False,
                candidate_count_before=candidate_count_before,
                candidate_count_after=candidate_count_before,
                observation_after=after,
                teacher_diagnostics=teacher_diagnostics,
            )
            return record, self._info(record)

        repeated = action in {turn.guess for turn in self._history}
        feedback = score_string(self._answer, action)
        self._history.append(Turn(guess=action, feedback=feedback))
        self._candidate_indices = self._expert.update(
            self._candidate_indices,
            self._word_to_index[action],
            feedback,
        )
        self._turn_count += 1

        if feedback == "GGGGG":
            done = True
            terminal_reason = "solved"
            reward = 1.0
        elif self._turn_count >= self.config.max_turns:
            done = True
            terminal_reason = "exhausted"
            reward = 0.0
        else:
            done = False
            terminal_reason = None
            reward = 0.0
        self._done = done

        after = self._observation()
        record = StepRecord(
            episode_id=self._episode_id,
            turn=self._turn_count,
            observation_before=before,
            action=action,
            feedback=feedback,
            reward=reward,
            done=done,
            terminal_reason=terminal_reason,
            repeated=repeated,
            candidate_count_before=candidate_count_before,
            candidate_count_after=len(self._candidate_indices),
            observation_after=after,
            teacher_diagnostics=teacher_diagnostics,
        )
        return record, self._info(record)

    def _observation(self) -> Observation:
        state_key = render_state_key(self._history)
        candidate_count = len(self._candidate_indices)
        return Observation(
            history=tuple(self._history),
            turn=self._turn_count,
            remaining_turns=max(self.config.max_turns - self._turn_count, 0),
            candidate_count=candidate_count,
            state_key=state_key,
            prompt=structured_next_guess_prompt(
                self._history,
                candidate_count,
            ),
        )

    def _info(self, record: StepRecord) -> Mapping[str, Any]:
        values: dict[str, Any] = {
            "episode_id": record.episode_id,
            "turn": record.turn,
            "repeated": record.repeated,
            "candidate_count_before": record.candidate_count_before,
            "candidate_count_after": record.candidate_count_after,
            "terminal_reason": record.terminal_reason,
        }
        if record.done:
            values["answer"] = self._answer
        return MappingProxyType(values)

    def _teacher_diagnostics(
        self,
        action: str | None,
    ) -> TeacherDiagnostics:
        if len(self._candidate_indices) == 0:
            return TeacherDiagnostics(None, None, None, None)
        teacher_index = self._expert.choose(self._candidate_indices)
        teacher_entropy = self._expert.entropy(
            teacher_index,
            self._candidate_indices,
        )
        if action is None:
            action_entropy = None
            action_is_candidate = None
        else:
            action_index = self._word_to_index[action]
            action_entropy = self._expert.entropy(
                action_index,
                self._candidate_indices,
            )
            action_is_candidate = bool(
                np.any(self._candidate_indices == action_index)
            )
        return TeacherDiagnostics(
            candidate_teacher_guess=self.config.answers[teacher_index],
            candidate_teacher_entropy_bits=teacher_entropy,
            action_entropy_bits=action_entropy,
            action_is_candidate=action_is_candidate,
        )

    def _is_valid_action(self, action: Any) -> bool:
        return (
            isinstance(action, str)
            and len(action) == 5
            and action.isascii()
            and action.isalpha()
            and action == action.upper()
            and action in self._answer_set
        )

    def _require_reset(self) -> None:
        if self._answer is None or self._episode_id is None:
            raise RuntimeError("call reset before using the environment")


def serialize_trace(trace: EpisodeTrace) -> dict[str, Any]:
    return trace.to_dict()


def replay_trace(
    trace: EpisodeTrace | Mapping[str, Any] | str,
    *,
    config: EnvironmentConfig,
    patterns: np.ndarray,
    answer: str | None = None,
) -> EpisodeTrace:
    if isinstance(trace, str):
        payload = json.loads(trace)
    elif isinstance(trace, EpisodeTrace):
        payload = trace.to_dict()
    else:
        payload = dict(trace)
    replay_answer = answer or payload.get("answer_id")
    if not isinstance(replay_answer, str):
        raise TypeError("replay requires a protected answer_id or answer")
    episode_id = str(payload.get("episode_id", "episode"))
    seed = payload.get("seed")
    environment = WordleEnvironment(config, patterns=patterns)
    environment.reset(
        replay_answer,
        episode_id=episode_id,
        seed=seed,
    )

    expected_records = payload.get("records")
    if not isinstance(expected_records, list):
        raise TypeError("trace records must be a list")
    expected_actions = [
        record.get("action") for record in expected_records
        if isinstance(record, Mapping)
    ]
    if len(expected_actions) != len(expected_records):
        raise ValueError("trace records must be objects")

    if config.opening is None:
        for action in expected_actions:
            if environment.done:
                break
            environment.step(action)
    else:
        if not expected_actions:
            raise ValueError("trace omitted the fixed opening record")
        if expected_actions[0] != config.opening:
            raise ValueError("trace opening does not match environment config")
        for action in expected_actions[1:]:
            if environment.done:
                break
            environment.step(action)

    actual = environment.trace
    actual_payload = actual.to_dict()
    if actual_payload["records"] != expected_records:
        for index, (actual_record, expected_record) in enumerate(
            zip(actual_payload["records"], expected_records)
        ):
            if actual_record != expected_record:
                raise ValueError(f"trace replay mismatch at record {index}")
        if len(actual_payload["records"]) != len(expected_records):
            raise ValueError("trace replay length mismatch")
    return actual
