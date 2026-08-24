"""Group-relative policy optimization objective for the Wordle simulator.

This module owns the SQ34 update math and nothing else. It imports ``torch``
and the standard library, and it deliberately imports no model, tokenizer,
environment, or rollout code. Everything here operates on tensors and plain
Python values, so the whole objective is testable on CPU with analytic
expectations and no 0.6B checkpoint.

The side quest PRD draws a hard boundary between SQ31 and SQ34: SQ31 owns the
environment, the policy, and the immutable trace, and must not contain
optimizer code. SQ34 owns the objective. That is why this file is new rather
than an addition to ``rollout.py``.

Conventions frozen here, matching the ``## SQ34`` section of
``.ai/sidequest-simulator-grpo-prd.md``:

* the importance ratio is word-level, ``exp(sum of token log-ratios))``, so the
  unit of the ratio equals the unit of the environment's action;
* clipping is word-level, matching the ratio;
* each episode's action objectives are averaged before the group average, so a
  six-turn failure does not outweigh an early win;
* the group-relative advantage is ``(R - mean(R)) / (std(R) + epsilon)`` using
  the population standard deviation;
* the reference term is a sampled k3 estimator of ``KL(current || reference)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

#: Bound applied to word-level log ratios before exponentiation.
#:
#: This exists for numerical safety, not to change the objective. The clipped
#: surrogate already selects the clipped branch whenever the ratio is far
#: outside ``[clip_lower, clip_upper]``, so saturating the log ratio at a value
#: this extreme cannot change which branch ``minimum`` picks. Without it a
#: single pathological action can produce ``inf`` and destroy the batch.
LOG_RATIO_LIMIT = 10.0


@dataclass(frozen=True)
class GrpoHyperparameters:
    """The preregistered pieces of the objective.

    These are frozen before the first optimizer step and written to the run
    manifest. ``clip_lower`` and ``clip_upper`` are the ratio bounds
    themselves, not a symmetric epsilon around one, because the PRD asks for
    the bounds to be recorded unambiguously.
    """

    clip_lower: float = 0.8
    clip_upper: float = 1.2
    advantage_epsilon: float = 1e-4
    kl_coefficient: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 < self.clip_lower < 1.0:
            raise ValueError("clip_lower must lie in (0, 1)")
        if not self.clip_upper > 1.0:
            raise ValueError("clip_upper must exceed 1")
        if self.advantage_epsilon <= 0.0:
            raise ValueError("advantage_epsilon must be positive")
        if self.kl_coefficient < 0.0:
            raise ValueError("kl_coefficient must be nonnegative")

    def as_manifest(self) -> dict[str, float]:
        return {
            "clip_lower": self.clip_lower,
            "clip_upper": self.clip_upper,
            "advantage_epsilon": self.advantage_epsilon,
            "kl_coefficient": self.kl_coefficient,
            "log_ratio_limit": LOG_RATIO_LIMIT,
            "advantage_normalization": "population_std",
            "ratio_unit": "word",
            "clip_unit": "word",
            "episode_normalization": "mean_over_actions",
            "kl_estimator": "k3",
        }


@dataclass(frozen=True)
class UpdateDiagnostics:
    """Detached statistics describing one loss evaluation.

    Every field is a plain Python number. Nothing here holds a graph, so these
    can be accumulated across a whole run without pinning activations.
    """

    episode_count: int
    updated_episode_count: int
    action_count: int
    degenerate: bool
    mean_ratio: float
    min_ratio: float
    max_ratio: float
    ratio_spread: float
    clipped_fraction: float
    saturated_log_ratio_count: int
    mean_kl: float
    max_absolute_advantage: float


def is_degenerate(rewards: Sequence[float]) -> bool:
    """Report whether a group carries no learning signal.

    An all-zero or all-one group produces identical returns, so every
    group-relative advantage is zero and the update is a no-op. The PRD
    requires recording these and skipping their optimizer update rather than
    manufacturing variance with a local proxy reward.
    """
    if len(rewards) == 0:
        return True
    first = rewards[0]
    return all(reward == first for reward in rewards)


def group_advantages(
    rewards: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Return ``(R - mean(R)) / (std(R) + epsilon)``.

    Uses the population standard deviation, so a degenerate group yields
    exactly zero rather than a NaN. The epsilon guards the all-equal case,
    where the numerator is already zero.
    """
    if rewards.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    if rewards.numel() == 0:
        raise ValueError("rewards must not be empty")
    centered = rewards - rewards.mean()
    deviation = torch.sqrt((centered * centered).mean())
    return centered / (deviation + epsilon)


def mean_only_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Return ``R - mean(R)``, recorded alongside the specified advantage.

    The PRD specifies the standard-deviation-normalized form, so that is what
    trains. This variant is a diagnostic. With binary reward and four episodes
    the population standard deviation takes only two nonzero values, so
    dividing by it pushes roughly 1.73 times harder on a one-of-four group's
    single winner than on a two-of-four group's winners. Recording both lets a
    later lab see whether that weighting mattered here.
    """
    if rewards.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    if rewards.numel() == 0:
        raise ValueError("rewards must not be empty")
    return rewards - rewards.mean()


def word_level_log_ratio(
    current_log_probabilities: torch.Tensor,
    behavior_log_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Return the word-level log importance ratio for each action.

    Both inputs are already summed over an action's tokens, so subtracting
    gives ``sum(token_log_probability_current - token_log_probability_behavior)``
    directly. The result is saturated at :data:`LOG_RATIO_LIMIT`.
    """
    if current_log_probabilities.shape != behavior_log_probabilities.shape:
        raise ValueError("current and behavior log probabilities must have equal shape")
    difference = current_log_probabilities - behavior_log_probabilities
    return torch.clamp(difference, min=-LOG_RATIO_LIMIT, max=LOG_RATIO_LIMIT)


def clipped_surrogate(
    log_ratio: torch.Tensor,
    advantage: float,
    *,
    clip_lower: float,
    clip_upper: float,
) -> torch.Tensor:
    """Return the per-action clipped surrogate objective, higher being better.

    This is the usual pessimistic bound, ``min(r * A, clip(r) * A)``. The
    asymmetry is intentional: for a positive advantage a ratio above
    ``clip_upper`` selects the constant branch and stops the gradient, while a
    ratio below ``clip_lower`` keeps its gradient so the action can recover.
    """
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, min=clip_lower, max=clip_upper)
    return torch.minimum(ratio * advantage, clipped * advantage)


def k3_kl(
    current_log_probabilities: torch.Tensor,
    reference_log_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Return the per-action k3 estimator of ``KL(current || reference)``.

    With ``d = log reference - log current`` the estimator is
    ``exp(d) - d - 1``. It is nonnegative for every sample rather than only in
    expectation, it is exactly zero when the policies agree, and its gradient
    vanishes there too, so it applies no pressure at initialization.
    """
    if current_log_probabilities.shape != reference_log_probabilities.shape:
        raise ValueError("current and reference log probabilities must have equal shape")
    difference = torch.clamp(
        reference_log_probabilities - current_log_probabilities,
        min=-LOG_RATIO_LIMIT,
        max=LOG_RATIO_LIMIT,
    )
    return torch.exp(difference) - difference - 1.0


def episode_objective(
    current_log_probabilities: torch.Tensor,
    behavior_log_probabilities: torch.Tensor,
    reference_log_probabilities: torch.Tensor,
    advantage: float,
    *,
    hyperparameters: GrpoHyperparameters,
) -> torch.Tensor:
    """Return one episode's mean action objective, higher being better.

    Averaging over the episode's actions before the caller averages over the
    group is what keeps a six-turn failure from outweighing a two-turn win.
    Without it an episode's influence would scale with its length, which is
    itself correlated with losing.
    """
    surrogate = clipped_surrogate(
        word_level_log_ratio(current_log_probabilities, behavior_log_probabilities),
        advantage,
        clip_lower=hyperparameters.clip_lower,
        clip_upper=hyperparameters.clip_upper,
    )
    penalty = k3_kl(current_log_probabilities, reference_log_probabilities)
    return (surrogate - hyperparameters.kl_coefficient * penalty).mean()


@dataclass(frozen=True)
class GroupBatch:
    """One group's collected evidence, with no gradient-bearing tensors.

    A group holds the hidden answer fixed and samples several complete
    episodes, so ``rewards`` has one entry per episode and the log-probability
    tuples have one inner tuple per episode and one entry per policy action
    within it. The fixed opening is not a policy action and does not appear.

    The behavior and reference values are recorded numbers rather than tensors
    because both come from frozen checkpoints. Only the current policy's
    scores carry a graph, and those are passed separately to :func:`group_loss`
    so this record stays cheap to serialize and safe to keep for the whole run.
    """

    group_id: str
    answer: str
    behavior_checkpoint_digest: str
    reference_checkpoint_digest: str
    rewards: tuple[float, ...]
    behavior_log_probabilities: tuple[tuple[float, ...], ...]
    reference_log_probabilities: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        episodes = len(self.rewards)
        if episodes == 0:
            raise ValueError("a group must contain at least one episode")
        if len(self.behavior_log_probabilities) != episodes:
            raise ValueError("behavior log probabilities must have one entry per episode")
        if len(self.reference_log_probabilities) != episodes:
            raise ValueError("reference log probabilities must have one entry per episode")
        for index in range(episodes):
            behavior = self.behavior_log_probabilities[index]
            reference = self.reference_log_probabilities[index]
            if len(behavior) != len(reference):
                raise ValueError(
                    f"episode {index} has {len(behavior)} behavior actions "
                    f"and {len(reference)} reference actions"
                )
            for value in behavior + reference:
                if not math.isfinite(value):
                    raise ValueError(f"episode {index} carries a non-finite log probability")

    @property
    def episode_count(self) -> int:
        return len(self.rewards)

    @property
    def action_counts(self) -> tuple[int, ...]:
        return tuple(len(entry) for entry in self.behavior_log_probabilities)

    @property
    def degenerate(self) -> bool:
        return is_degenerate(self.rewards)


def group_loss(
    batch: GroupBatch,
    current_log_probabilities: Sequence[torch.Tensor],
    *,
    hyperparameters: GrpoHyperparameters,
) -> tuple[torch.Tensor, UpdateDiagnostics]:
    """Return the scalar loss for one group and its detached diagnostics.

    The loss is the negated mean episode objective, so calling ``backward`` on
    it ascends the surrogate. A degenerate group returns a zero loss that still
    carries a graph, so the caller can treat every group uniformly, and the
    diagnostics record that it contributed nothing.

    An episode whose opening already solved the answer has no policy actions.
    It still counts toward the group's reward distribution, because that is the
    outcome the answer actually produced, but it cannot contribute a gradient
    and is excluded from the objective average.
    """
    if len(current_log_probabilities) != batch.episode_count:
        raise ValueError("current log probabilities must have one entry per episode")

    device = _resolve_device(current_log_probabilities)
    rewards = torch.tensor(batch.rewards, dtype=torch.float32)
    advantages = group_advantages(rewards, epsilon=hyperparameters.advantage_epsilon)

    objectives: list[torch.Tensor] = []
    ratios: list[torch.Tensor] = []
    clipped_flags: list[torch.Tensor] = []
    penalties: list[torch.Tensor] = []
    saturated = 0
    action_count = 0

    for index in range(batch.episode_count):
        current = current_log_probabilities[index]
        behavior = torch.tensor(
            batch.behavior_log_probabilities[index],
            dtype=current.dtype,
            device=current.device,
        )
        reference = torch.tensor(
            batch.reference_log_probabilities[index],
            dtype=current.dtype,
            device=current.device,
        )
        if current.shape != behavior.shape:
            raise ValueError(
                f"episode {index} scored {tuple(current.shape)} actions "
                f"but recorded {tuple(behavior.shape)}"
            )
        if current.numel() == 0:
            continue

        action_count += current.numel()
        advantage = float(advantages[index])
        objectives.append(
            episode_objective(
                current,
                behavior,
                reference,
                advantage,
                hyperparameters=hyperparameters,
            )
        )

        with torch.no_grad():
            raw = current - behavior
            saturated += int((raw.abs() > LOG_RATIO_LIMIT).sum())
            ratio = torch.exp(
                torch.clamp(raw, min=-LOG_RATIO_LIMIT, max=LOG_RATIO_LIMIT)
            )
            ratios.append(ratio)
            clipped_flags.append(
                (ratio < hyperparameters.clip_lower) | (ratio > hyperparameters.clip_upper)
            )
            penalties.append(k3_kl(current, reference))

    if objectives:
        loss = -torch.stack(objectives).mean()
    else:
        loss = torch.zeros((), dtype=torch.float32, device=device)

    diagnostics = _diagnostics(
        batch=batch,
        updated_episode_count=len(objectives),
        action_count=action_count,
        ratios=ratios,
        clipped_flags=clipped_flags,
        penalties=penalties,
        saturated=saturated,
        advantages=advantages,
    )
    return loss, diagnostics


def _resolve_device(tensors: Sequence[torch.Tensor]) -> torch.device:
    for tensor in tensors:
        return tensor.device
    return torch.device("cpu")


def _diagnostics(
    *,
    batch: GroupBatch,
    updated_episode_count: int,
    action_count: int,
    ratios: list[torch.Tensor],
    clipped_flags: list[torch.Tensor],
    penalties: list[torch.Tensor],
    saturated: int,
    advantages: torch.Tensor,
) -> UpdateDiagnostics:
    if ratios:
        flat = torch.cat(ratios)
        flags = torch.cat(clipped_flags)
        penalty = torch.cat(penalties)
        mean_ratio = float(flat.mean())
        min_ratio = float(flat.min())
        max_ratio = float(flat.max())
        clipped_fraction = float(flags.to(torch.float32).mean())
        mean_kl = float(penalty.mean())
    else:
        mean_ratio = math.nan
        min_ratio = math.nan
        max_ratio = math.nan
        clipped_fraction = math.nan
        mean_kl = math.nan

    return UpdateDiagnostics(
        episode_count=batch.episode_count,
        updated_episode_count=updated_episode_count,
        action_count=action_count,
        degenerate=batch.degenerate,
        mean_ratio=mean_ratio,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        ratio_spread=max_ratio - min_ratio if ratios else math.nan,
        clipped_fraction=clipped_fraction,
        saturated_log_ratio_count=saturated,
        mean_kl=mean_kl,
        max_absolute_advantage=float(advantages.abs().max()),
    )
