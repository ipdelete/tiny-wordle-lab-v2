from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .environment import (
    EnvironmentConfig,
    EpisodeTrace,
    Observation,
    StepRecord,
    WordleEnvironment,
    replay_trace,
)


class RolloutPolicy(Protocol):
    def sample(
        self,
        observation: Observation,
        *,
        temperature: float,
        seed: int,
    ) -> PolicyDecision:
        ...

    def log_probability(
        self,
        observation: Observation,
        word: str,
        *,
        temperature: float,
    ) -> tuple[float, tuple[float, ...]]:
        ...


@dataclass(frozen=True)
class PolicyDecision:
    word: str
    token_ids: tuple[int, ...]
    per_token_log_probabilities: tuple[float, ...]
    action_log_probability: float
    checkpoint_digest: str
    temperature: float
    mask_version: str
    sampling_seed: int
    tokenizer_digest: str = ""
    model_calls: int = 0

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.per_token_log_probabilities):
            raise ValueError("token IDs and log probabilities must have equal length")
        expected = math.fsum(self.per_token_log_probabilities)
        if not math.isclose(
            expected,
            self.action_log_probability,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("action log probability does not match token values")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.model_calls < 0:
            raise ValueError("model_calls must not be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "token_ids": list(self.token_ids),
            "per_token_log_probabilities": list(
                self.per_token_log_probabilities
            ),
            "action_log_probability": self.action_log_probability,
            "checkpoint_digest": self.checkpoint_digest,
            "temperature": self.temperature,
            "mask_version": self.mask_version,
            "sampling_seed": self.sampling_seed,
            "tokenizer_digest": self.tokenizer_digest,
            "model_calls": self.model_calls,
        }


@dataclass(frozen=True)
class TrajectoryStep:
    observation: Observation
    decision: PolicyDecision
    transition: StepRecord

    def __post_init__(self) -> None:
        if self.observation != self.transition.observation_before:
            raise ValueError("decision observation does not match transition")
        if self.decision.word != self.transition.action:
            raise ValueError("decision word does not match transition action")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "decision": self.decision.to_dict(),
            "transition": self.transition.to_dict(),
        }


@dataclass(frozen=True)
class Trajectory:
    episode_id: str
    group_id: str
    policy_checkpoint_digest: str
    reference_checkpoint_digest: str
    policy_view_version: str
    answer_split: str
    protected_answer_id: str
    temperature: float
    sampling_seed: int
    opening_record: StepRecord | None
    steps: tuple[TrajectoryStep, ...]
    terminal_reason: str | None
    return_value: float
    action_vocabulary_digest: str = ""
    tokenizer_digest: str = ""
    eos_convention: str = "append_eos_token_id"
    restriction_algorithm: str = ""

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.return_value not in (0.0, 1.0):
            raise ValueError("Wordle return must be 0 or 1")
        object.__setattr__(self, "steps", tuple(self.steps))

    @property
    def records(self) -> tuple[StepRecord, ...]:
        opening = (self.opening_record,) if self.opening_record else ()
        return opening + tuple(step.transition for step in self.steps)

    @property
    def environment_trace(self) -> EpisodeTrace:
        return EpisodeTrace(
            episode_id=self.episode_id,
            answer_id=self.protected_answer_id,
            seed=self.sampling_seed,
            records=self.records,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "group_id": self.group_id,
            "policy_checkpoint_digest": self.policy_checkpoint_digest,
            "reference_checkpoint_digest": self.reference_checkpoint_digest,
            "policy_view_version": self.policy_view_version,
            "answer_split": self.answer_split,
            "protected_answer_id": self.protected_answer_id,
            "temperature": self.temperature,
            "sampling_seed": self.sampling_seed,
            "opening_record": (
                self.opening_record.to_dict()
                if self.opening_record is not None
                else None
            ),
            "steps": [step.to_dict() for step in self.steps],
            "terminal_reason": self.terminal_reason,
            "return_value": self.return_value,
            "action_vocabulary_digest": self.action_vocabulary_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "eos_convention": self.eos_convention,
            "restriction_algorithm": self.restriction_algorithm,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def collect_trajectory(
    environment: WordleEnvironment,
    policy: RolloutPolicy,
    answer: str,
    *,
    group_id: str,
    answer_split: str,
    protected_answer_id: str | None = None,
    reference_checkpoint_digest: str = "",
    policy_checkpoint_digest: str | None = None,
    temperature: float,
    sampling_seed: int,
    episode_id: str = "episode",
) -> Trajectory:
    start = environment.reset(
        answer,
        episode_id=episode_id,
        seed=sampling_seed,
    )
    steps: list[TrajectoryStep] = []
    observation = start.observation
    action_index = 0
    while not environment.done:
        decision = policy.sample(
            observation,
            temperature=temperature,
            seed=sampling_seed + action_index,
        )
        before = observation
        observation, _, done, _ = environment.step(decision.word)
        transition = environment.last_record
        steps.append(
            TrajectoryStep(
                observation=before,
                decision=decision,
                transition=transition,
            )
        )
        action_index += 1
        if done:
            break

    if not environment.done:
        raise RuntimeError("trajectory collector stopped before termination")
    terminal_reason = environment.last_record.terminal_reason
    return_value = environment.last_record.reward
    checkpoint_digests = {
        step.decision.checkpoint_digest
        for step in steps
    }
    if len(checkpoint_digests) > 1:
        raise ValueError("trajectory contains multiple behavior checkpoints")
    observed_checkpoint_digest = next(iter(checkpoint_digests), "")
    if (
        policy_checkpoint_digest is not None
        and observed_checkpoint_digest
        and policy_checkpoint_digest != observed_checkpoint_digest
    ):
        raise ValueError("decision checkpoint differs from collector metadata")
    policy_checkpoint_digest = (
        policy_checkpoint_digest
        if policy_checkpoint_digest is not None
        else observed_checkpoint_digest
    )
    tokenizer_digests = {
        step.decision.tokenizer_digest
        for step in steps
        if step.decision.tokenizer_digest
    }
    if len(tokenizer_digests) > 1:
        raise ValueError("trajectory contains multiple tokenizer digests")
    tokenizer_digest = next(iter(tokenizer_digests), "")
    mask_versions = {
        step.decision.mask_version
        for step in steps
        if step.decision.mask_version
    }
    if len(mask_versions) > 1:
        raise ValueError("trajectory contains multiple restriction algorithms")
    restriction_algorithm = next(
        iter(mask_versions),
        getattr(policy, "mask_version", ""),
    )
    action_vocabulary_digest = getattr(
        policy,
        "action_vocabulary_digest",
        "",
    )
    return Trajectory(
        episode_id=episode_id,
        group_id=group_id,
        policy_checkpoint_digest=policy_checkpoint_digest,
        reference_checkpoint_digest=reference_checkpoint_digest,
        policy_view_version=environment.config.policy_view,
        answer_split=answer_split,
        protected_answer_id=protected_answer_id or answer,
        temperature=temperature,
        sampling_seed=sampling_seed,
        opening_record=start.opening_record,
        steps=tuple(steps),
        terminal_reason=terminal_reason,
        return_value=return_value,
        action_vocabulary_digest=action_vocabulary_digest,
        tokenizer_digest=tokenizer_digest,
        restriction_algorithm=restriction_algorithm,
    )


def replay_trajectory(
    trajectory: Trajectory,
    *,
    config: EnvironmentConfig,
    patterns: np.ndarray,
    policy: RolloutPolicy,
    answer: str | None = None,
) -> EpisodeTrace:
    replay_answer = answer or trajectory.protected_answer_id
    actual = replay_trace(
        trajectory.environment_trace,
        config=config,
        patterns=patterns,
        answer=replay_answer,
    )
    final_record = actual.records[-1]
    if final_record.reward != trajectory.return_value:
        raise ValueError("trajectory return does not replay")
    if final_record.terminal_reason != trajectory.terminal_reason:
        raise ValueError("trajectory terminal reason does not replay")

    for index, step in enumerate(trajectory.steps):
        if (
            trajectory.policy_checkpoint_digest
            and step.decision.checkpoint_digest
            != trajectory.policy_checkpoint_digest
        ):
            raise ValueError(
                f"checkpoint digest mismatch at trajectory step {index}"
            )
        if (
            trajectory.tokenizer_digest
            and step.decision.tokenizer_digest
            != trajectory.tokenizer_digest
        ):
            raise ValueError(
                f"tokenizer digest mismatch at trajectory step {index}"
            )
        if step.decision.temperature != trajectory.temperature:
            raise ValueError(f"temperature mismatch at trajectory step {index}")
        if step.decision.mask_version != trajectory.restriction_algorithm:
            raise ValueError(
                f"restriction algorithm mismatch at trajectory step {index}"
            )
        if step.decision.sampling_seed != trajectory.sampling_seed + index:
            raise ValueError(f"sampling seed mismatch at trajectory step {index}")
        total, per_token = policy.log_probability(
            step.observation,
            step.decision.word,
            temperature=step.decision.temperature,
        )
        if not math.isclose(
            total,
            step.decision.action_log_probability,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"action log probability mismatch at trajectory step {index}"
            )
        if len(per_token) != len(step.decision.per_token_log_probabilities) or any(
            not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
            for actual, expected in zip(
                per_token,
                step.decision.per_token_log_probabilities,
            )
        ):
            raise ValueError(
                f"token log probability mismatch at trajectory step {index}"
            )
    return actual
