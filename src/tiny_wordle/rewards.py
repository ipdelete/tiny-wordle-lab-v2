from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RewardBreakdown:
    format: float
    repeat: float
    history: float
    progress: float
    solve: float
    efficiency: float

    @property
    def total(self) -> float:
        return (
            self.format
            + self.repeat
            + self.history
            + self.progress
            + self.solve
            + self.efficiency
        )


def candidate_reduction_value(
    before: int,
    after: int,
) -> float:
    if (
        before <= 0
        or after <= 0
    ):
        return 0.0

    return math.log(
        before / after
    )


def reward_breakdown(
    *,
    valid_format: bool,
    repeated: bool,
    history_consistent: bool,
    has_history: bool,
    solved: bool,
    candidates_before: int,
    candidates_after: int,
    turn_number: int,
) -> RewardBreakdown:
    format_reward = (
        0.05
        if valid_format
        else -0.25
    )

    repeat_reward = (
        -0.20
        if repeated
        else 0.0
    )

    if (
        has_history
        and valid_format
    ):
        history_reward = (
            0.15
            if history_consistent
            else -0.15
        )
    else:
        history_reward = 0.0

    if valid_format:
        reduction = (
            candidate_reduction_value(
                candidates_before,
                candidates_after,
            )
        )

        progress_reward = (
            0.20
            * math.tanh(
                reduction
            )
        )
    else:
        progress_reward = 0.0

    solve_reward = (
        2.0
        if solved
        else 0.0
    )

    efficiency_reward = (
        max(
            0.0,
            0.10 * (
                6 - turn_number
            ),
        )
        if solved
        else 0.0
    )

    return RewardBreakdown(
        format=format_reward,
        repeat=repeat_reward,
        history=history_reward,
        progress=progress_reward,
        solve=solve_reward,
        efficiency=efficiency_reward,
    )
