from __future__ import annotations

from tiny_wordle_lab_v2.evaluate import GameResult
from tiny_wordle_lab_v2.prompt_scoring import lexicographic_game_score


def game(*, solved: bool, turns: int) -> GameResult:
    return GameResult(
        answer="banal",
        solved=solved,
        policy_calls=turns,
        opportunities_used=turns,
        accepted_guesses=(),
        actions=(),
    )


def test_score_is_bounded() -> None:
    assert lexicographic_game_score(game(solved=False, turns=6), total_games=4) == 0
    assert lexicographic_game_score(game(solved=True, turns=1), total_games=4) == 1


def test_one_more_solve_outweighs_all_turn_savings() -> None:
    total_games = 4
    three_fast_wins = (
        3
        * lexicographic_game_score(
            game(solved=True, turns=1),
            total_games=total_games,
        )
    )
    four_slow_wins = (
        4
        * lexicographic_game_score(
            game(solved=True, turns=6),
            total_games=total_games,
        )
    )
    assert four_slow_wins > three_fast_wins
