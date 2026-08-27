from __future__ import annotations

from .evaluate import GameResult


def penalized_turns(game: GameResult) -> int:
    return game.opportunities_used if game.solved else 7


def lexicographic_game_score(game: GameResult, *, total_games: int) -> float:
    """Encode solve count first and penalized turns second in [0, 1]."""
    if total_games < 1:
        raise ValueError("total_games must be positive")
    solve_weight = 7 * total_games + 1
    numerator = int(game.solved) * solve_weight + 7 - penalized_turns(game)
    return numerator / (solve_weight + 6)
