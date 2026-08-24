"""Analytic tests for the SQ34 GRPO objective.

Every test here runs on CPU with hand-computed expectations and no model. The
objective is the one component whose bugs are invisible in the run output: a
sign error, a missing normalization, or a mis-scaled advantage all produce
numbers that look like training. These tests are the only place those are
cheap to catch.
"""

from __future__ import annotations

import math

import pytest
import torch

from tiny_wordle.grpo import (
    LOG_RATIO_LIMIT,
    GroupBatch,
    GrpoHyperparameters,
    clipped_surrogate,
    episode_objective,
    group_advantages,
    group_loss,
    is_degenerate,
    k3_kl,
    mean_only_advantages,
    word_level_log_ratio,
)

HYPERPARAMETERS = GrpoHyperparameters()


def scores(values: list[float], *, grad: bool = False) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, requires_grad=grad)


def batch(
    rewards: tuple[float, ...],
    action_counts: tuple[int, ...],
    *,
    behavior: float = -3.0,
) -> GroupBatch:
    return GroupBatch(
        group_id="group-0",
        answer="SHORE",
        behavior_checkpoint_digest="a" * 64,
        reference_checkpoint_digest="a" * 64,
        rewards=rewards,
        behavior_log_probabilities=tuple(
            tuple([behavior] * count) for count in action_counts
        ),
        reference_log_probabilities=tuple(
            tuple([behavior] * count) for count in action_counts
        ),
    )


class TestGroupAdvantages:
    def test_two_of_four_gives_unit_advantages(self):
        """A balanced binary group has population std 0.5, so advantages are +-1."""
        advantages = group_advantages(
            scores([1.0, 1.0, 0.0, 0.0]), epsilon=HYPERPARAMETERS.advantage_epsilon
        )
        assert advantages.tolist() == pytest.approx([1.0, 1.0, -1.0, -1.0], abs=1e-3)

    def test_one_of_four_pushes_harder_on_the_single_winner(self):
        """Population std is sqrt(0.1875) = 0.4330, so the winner gets 0.75/0.4330."""
        advantages = group_advantages(
            scores([1.0, 0.0, 0.0, 0.0]), epsilon=HYPERPARAMETERS.advantage_epsilon
        )
        assert advantages.tolist() == pytest.approx(
            [1.7320, -0.5773, -0.5773, -0.5773], abs=1e-3
        )

    def test_normalization_weights_lopsided_groups_more(self):
        """The recorded caveat, pinned as a number rather than a comment.

        Standard-deviation normalization makes a one-of-four group's winner
        carry 1.73 times the push of a two-of-four group's winner. The PRD
        specifies this form, so it trains; this test documents its size.
        """
        lopsided = group_advantages(
            scores([1.0, 0.0, 0.0, 0.0]), epsilon=HYPERPARAMETERS.advantage_epsilon
        )
        balanced = group_advantages(
            scores([1.0, 1.0, 0.0, 0.0]), epsilon=HYPERPARAMETERS.advantage_epsilon
        )
        assert float(lopsided.max()) / float(balanced.max()) == pytest.approx(
            1.7320, abs=1e-3
        )

    def test_advantages_sum_to_zero(self):
        advantages = group_advantages(
            scores([1.0, 0.0, 1.0, 0.0]), epsilon=HYPERPARAMETERS.advantage_epsilon
        )
        assert float(advantages.sum()) == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize("reward", [0.0, 1.0])
    def test_degenerate_group_yields_exactly_zero(self, reward):
        """Zero variance must produce zero, not a NaN from dividing by zero."""
        advantages = group_advantages(
            scores([reward] * 4), epsilon=HYPERPARAMETERS.advantage_epsilon
        )
        assert advantages.tolist() == [0.0, 0.0, 0.0, 0.0]
        assert not torch.isnan(advantages).any()

    def test_epsilon_does_not_materially_distort_a_real_group(self):
        advantages = group_advantages(scores([1.0, 1.0, 0.0, 0.0]), epsilon=1e-4)
        assert float(advantages[0]) == pytest.approx(1.0, abs=1e-3)

    def test_mean_only_advantages_skip_normalization(self):
        advantages = mean_only_advantages(scores([1.0, 0.0, 0.0, 0.0]))
        assert advantages.tolist() == pytest.approx([0.75, -0.25, -0.25, -0.25])

    def test_rejects_empty_and_multidimensional_rewards(self):
        with pytest.raises(ValueError, match="empty"):
            group_advantages(scores([]), epsilon=1e-4)
        with pytest.raises(ValueError, match="one-dimensional"):
            group_advantages(torch.zeros((2, 2)), epsilon=1e-4)


class TestDegeneracy:
    @pytest.mark.parametrize(
        "rewards,expected",
        [
            ((0.0, 0.0, 0.0, 0.0), True),
            ((1.0, 1.0, 1.0, 1.0), True),
            ((1.0, 0.0, 0.0, 0.0), False),
            ((1.0, 1.0, 1.0, 0.0), False),
            ((), True),
        ],
    )
    def test_detects_groups_without_signal(self, rewards, expected):
        assert is_degenerate(rewards) is expected


class TestWordLevelRatio:
    def test_identical_policies_give_zero_log_ratio(self):
        values = [-3.0, -1.5, -7.25]
        ratio = word_level_log_ratio(scores(values), scores(values))
        assert ratio.tolist() == [0.0, 0.0, 0.0]

    def test_log_ratio_is_the_difference_of_word_totals(self):
        ratio = word_level_log_ratio(scores([-2.0, -5.0]), scores([-3.0, -1.0]))
        assert ratio.tolist() == pytest.approx([1.0, -4.0])

    def test_extreme_ratios_saturate_rather_than_overflow(self):
        """A single pathological action must not turn the batch into inf."""
        ratio = word_level_log_ratio(scores([0.0, -400.0]), scores([-400.0, 0.0]))
        assert ratio.tolist() == [LOG_RATIO_LIMIT, -LOG_RATIO_LIMIT]
        assert torch.isfinite(torch.exp(ratio)).all()

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="equal shape"):
            word_level_log_ratio(scores([-1.0]), scores([-1.0, -2.0]))


class TestClippedSurrogate:
    def test_unit_ratio_returns_the_advantage(self):
        surrogate = clipped_surrogate(
            scores([0.0]), 1.5, clip_lower=0.8, clip_upper=1.2
        )
        assert float(surrogate[0]) == pytest.approx(1.5)

    def test_positive_advantage_is_capped_above_the_upper_bound(self):
        surrogate = clipped_surrogate(
            torch.log(scores([2.0])), 1.0, clip_lower=0.8, clip_upper=1.2
        )
        assert float(surrogate[0]) == pytest.approx(1.2)

    def test_negative_advantage_is_capped_below_the_lower_bound(self):
        surrogate = clipped_surrogate(
            torch.log(scores([0.5])), -1.0, clip_lower=0.8, clip_upper=1.2
        )
        assert float(surrogate[0]) == pytest.approx(-0.8)

    def test_gradient_vanishes_where_the_clip_binds(self):
        """A positive advantage above the upper bound must stop pushing."""
        current = scores([0.0], grad=True)
        surrogate = clipped_surrogate(
            current + math.log(2.0), 1.0, clip_lower=0.8, clip_upper=1.2
        )
        surrogate.sum().backward()
        assert float(current.grad[0]) == pytest.approx(0.0)

    def test_gradient_survives_on_the_recovering_side(self):
        """Below the lower bound a positive advantage keeps its gradient.

        This asymmetry is the point of the pessimistic bound: an action whose
        probability collapsed must still be able to climb back.
        """
        current = scores([0.0], grad=True)
        surrogate = clipped_surrogate(
            current + math.log(0.5), 1.0, clip_lower=0.8, clip_upper=1.2
        )
        surrogate.sum().backward()
        assert float(current.grad[0]) == pytest.approx(0.5)

    def test_gradient_flows_at_unit_ratio(self):
        current = scores([0.0], grad=True)
        clipped_surrogate(current, 2.0, clip_lower=0.8, clip_upper=1.2).sum().backward()
        assert float(current.grad[0]) == pytest.approx(2.0)


class TestK3Kl:
    def test_identical_policies_give_exactly_zero(self):
        values = [-3.0, -1.25]
        assert k3_kl(scores(values), scores(values)).tolist() == [0.0, 0.0]

    @pytest.mark.parametrize("delta", [-4.0, -0.75, -0.1, 0.1, 0.75, 4.0])
    def test_estimator_is_nonnegative_for_every_sample(self, delta):
        """k3 is nonnegative pointwise, not merely in expectation."""
        penalty = k3_kl(scores([-2.0]), scores([-2.0 + delta]))
        assert float(penalty[0]) >= 0.0

    def test_gradient_vanishes_at_agreement(self):
        """The reference term applies no pressure at initialization."""
        current = scores([-3.0], grad=True)
        k3_kl(current, scores([-3.0])).sum().backward()
        assert float(current.grad[0]) == pytest.approx(0.0)

    def test_matches_the_closed_form(self):
        penalty = k3_kl(scores([-2.0]), scores([-1.0]))
        assert float(penalty[0]) == pytest.approx(math.exp(1.0) - 1.0 - 1.0)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="equal shape"):
            k3_kl(scores([-1.0]), scores([-1.0, -2.0]))


class TestEpisodeObjective:
    def test_unit_ratio_and_matching_reference_returns_the_advantage(self):
        values = scores([-3.0, -2.0, -4.0])
        objective = episode_objective(
            values, values, values, 1.5, hyperparameters=HYPERPARAMETERS
        )
        assert float(objective) == pytest.approx(1.5)

    def test_objective_is_independent_of_episode_length(self):
        """A six-action episode and a two-action episode score identically.

        This is the property that stops a long losing game from dominating a
        short winning one purely by contributing more terms.
        """
        short = scores([-3.0] * 2)
        long = scores([-3.0] * 6)
        assert float(
            episode_objective(short, short, short, -1.0, hyperparameters=HYPERPARAMETERS)
        ) == pytest.approx(
            float(
                episode_objective(
                    long, long, long, -1.0, hyperparameters=HYPERPARAMETERS
                )
            )
        )

    def test_gradient_per_action_scales_inversely_with_episode_length(self):
        gradients = []
        for count in (2, 6):
            current = scores([-3.0] * count, grad=True)
            frozen = scores([-3.0] * count)
            episode_objective(
                current, frozen, frozen, 1.0, hyperparameters=HYPERPARAMETERS
            ).backward()
            gradients.append(float(current.grad[0]))
        assert gradients[0] == pytest.approx(1.0 / 2)
        assert gradients[1] == pytest.approx(1.0 / 6)

    def test_kl_penalty_reduces_the_objective(self):
        current = scores([-3.0])
        reference = scores([-1.0])
        penalized = episode_objective(
            current, current, reference, 1.0, hyperparameters=HYPERPARAMETERS
        )
        assert float(penalized) < 1.0


class TestGroupBatch:
    def test_reports_shape(self):
        group = batch((1.0, 0.0, 0.0, 0.0), (3, 5, 6, 4))
        assert group.episode_count == 4
        assert group.action_counts == (3, 5, 6, 4)
        assert group.degenerate is False

    def test_rejects_reward_and_episode_mismatch(self):
        with pytest.raises(ValueError, match="one entry per episode"):
            GroupBatch(
                group_id="g",
                answer="SHORE",
                behavior_checkpoint_digest="a" * 64,
                reference_checkpoint_digest="a" * 64,
                rewards=(1.0, 0.0),
                behavior_log_probabilities=((-1.0,),),
                reference_log_probabilities=((-1.0,), (-1.0,)),
            )

    def test_rejects_behavior_and_reference_action_mismatch(self):
        with pytest.raises(ValueError, match="behavior actions"):
            GroupBatch(
                group_id="g",
                answer="SHORE",
                behavior_checkpoint_digest="a" * 64,
                reference_checkpoint_digest="a" * 64,
                rewards=(1.0,),
                behavior_log_probabilities=((-1.0, -2.0),),
                reference_log_probabilities=((-1.0,),),
            )

    def test_rejects_non_finite_log_probabilities(self):
        with pytest.raises(ValueError, match="non-finite"):
            GroupBatch(
                group_id="g",
                answer="SHORE",
                behavior_checkpoint_digest="a" * 64,
                reference_checkpoint_digest="a" * 64,
                rewards=(1.0,),
                behavior_log_probabilities=((-math.inf,),),
                reference_log_probabilities=((-1.0,),),
            )

    def test_rejects_an_empty_group(self):
        with pytest.raises(ValueError, match="at least one episode"):
            GroupBatch(
                group_id="g",
                answer="SHORE",
                behavior_checkpoint_digest="a" * 64,
                reference_checkpoint_digest="a" * 64,
                rewards=(),
                behavior_log_probabilities=(),
                reference_log_probabilities=(),
            )


class TestGroupLoss:
    def test_loss_value_is_zero_at_collection_but_the_gradient_is_not(self):
        """The headline number lies here, and that is expected.

        Group-relative advantages sum to zero, so at ratio one the loss value
        is zero for every group. A run that reported only the loss would look
        like nothing was happening. The gradient is what carries the signal,
        so the run must report the movement check instead.
        """
        group = batch((1.0, 1.0, 0.0, 0.0), (3, 3, 3, 3))
        current = [scores([-3.0] * 3, grad=True) for _ in range(4)]
        loss, diagnostics = group_loss(
            group, current, hyperparameters=HYPERPARAMETERS
        )
        assert float(loss.detach()) == pytest.approx(0.0, abs=1e-5)

        loss.backward()
        assert float(current[0].grad[0]) == pytest.approx(-1.0 / (3 * 4), abs=1e-4)
        assert float(current[3].grad[0]) == pytest.approx(1.0 / (3 * 4), abs=1e-4)
        assert diagnostics.updated_episode_count == 4

    def test_winning_episodes_are_pushed_up_and_losing_ones_down(self):
        group = batch((1.0, 0.0, 0.0, 0.0), (2, 2, 2, 2))
        current = [scores([-3.0] * 2, grad=True) for _ in range(4)]
        loss, _ = group_loss(group, current, hyperparameters=HYPERPARAMETERS)
        loss.backward()
        assert float(current[0].grad[0]) < 0.0
        assert float(current[1].grad[0]) > 0.0

    def test_degenerate_group_produces_no_gradient(self):
        group = batch((0.0, 0.0, 0.0, 0.0), (4, 4, 4, 4))
        current = [scores([-3.0] * 4, grad=True) for _ in range(4)]
        loss, diagnostics = group_loss(
            group, current, hyperparameters=HYPERPARAMETERS
        )
        loss.backward()
        assert diagnostics.degenerate is True
        assert float(loss.detach()) == pytest.approx(0.0, abs=1e-6)
        assert all(float(tensor.grad.abs().sum()) == pytest.approx(0.0) for tensor in current)

    def test_episode_solved_by_the_opening_contributes_no_actions(self):
        """RAISE is one of the answers, so a zero-action episode is reachable.

        It still counts toward the group's reward distribution, because that
        outcome really happened, but it cannot carry a gradient.
        """
        group = batch((1.0, 0.0, 0.0, 0.0), (0, 3, 3, 3))
        current = [scores([], grad=True)] + [
            scores([-3.0] * 3, grad=True) for _ in range(3)
        ]
        loss, diagnostics = group_loss(
            group, current, hyperparameters=HYPERPARAMETERS
        )
        assert diagnostics.episode_count == 4
        assert diagnostics.updated_episode_count == 3
        assert diagnostics.action_count == 9
        assert torch.isfinite(loss)

    def test_diagnostics_report_ratio_spread_and_clipping(self):
        group = batch((1.0, 0.0), (1, 1))
        current = [scores([-3.0 + math.log(2.0)]), scores([-3.0])]
        _, diagnostics = group_loss(group, current, hyperparameters=HYPERPARAMETERS)
        assert diagnostics.max_ratio == pytest.approx(2.0, abs=1e-4)
        assert diagnostics.min_ratio == pytest.approx(1.0, abs=1e-4)
        assert diagnostics.ratio_spread == pytest.approx(1.0, abs=1e-4)
        assert diagnostics.clipped_fraction == pytest.approx(0.5)
        assert diagnostics.saturated_log_ratio_count == 0

    def test_diagnostics_report_zero_kl_against_a_matching_reference(self):
        group = batch((1.0, 0.0), (2, 2))
        current = [scores([-3.0] * 2), scores([-3.0] * 2)]
        _, diagnostics = group_loss(group, current, hyperparameters=HYPERPARAMETERS)
        assert diagnostics.mean_kl == pytest.approx(0.0, abs=1e-6)
        assert diagnostics.mean_ratio == pytest.approx(1.0, abs=1e-5)

    def test_saturated_log_ratios_are_counted(self):
        group = batch((1.0, 0.0), (1, 1))
        current = [scores([500.0]), scores([-3.0])]
        _, diagnostics = group_loss(group, current, hyperparameters=HYPERPARAMETERS)
        assert diagnostics.saturated_log_ratio_count == 1
        assert math.isfinite(diagnostics.max_ratio)

    def test_rejects_scores_that_do_not_match_the_recorded_actions(self):
        group = batch((1.0, 0.0), (3, 3))
        current = [scores([-3.0] * 2), scores([-3.0] * 3)]
        with pytest.raises(ValueError, match="scored"):
            group_loss(group, current, hyperparameters=HYPERPARAMETERS)

    def test_rejects_a_score_count_that_does_not_match_the_group(self):
        group = batch((1.0, 0.0), (3, 3))
        with pytest.raises(ValueError, match="one entry per episode"):
            group_loss(group, [scores([-3.0] * 3)], hyperparameters=HYPERPARAMETERS)


class TestHyperparameters:
    def test_manifest_records_every_frozen_choice(self):
        manifest = HYPERPARAMETERS.as_manifest()
        assert manifest["ratio_unit"] == "word"
        assert manifest["clip_unit"] == "word"
        assert manifest["kl_estimator"] == "k3"
        assert manifest["advantage_normalization"] == "population_std"
        assert manifest["episode_normalization"] == "mean_over_actions"

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"clip_lower": 1.2}, "clip_lower"),
            ({"clip_lower": 0.0}, "clip_lower"),
            ({"clip_upper": 0.9}, "clip_upper"),
            ({"advantage_epsilon": 0.0}, "advantage_epsilon"),
            ({"kl_coefficient": -0.1}, "kl_coefficient"),
        ],
    )
    def test_rejects_incoherent_settings(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            GrpoHyperparameters(**kwargs)
