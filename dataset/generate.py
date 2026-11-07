"""Generation of a balanced dataset of labelled positions, from random games."""

import random
from collections.abc import Iterator

from dataset.board import Board
from dataset.labels import label_position
from dataset.moves import legal_moves

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
LABELS = ("Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black")
MAX_PLIES = 300
MATE_PROBABILITY = 0.9
CHECK_PROBABILITY = 0.3


def label_shares(nothing: float, check: float, checkmate: float) -> dict[str, float]:
    """Split the shares of checks and mates evenly between both colors."""
    return {
        "Nothing": nothing,
        "Check White": check / 2, "Check Black": check / 2,
        "Checkmate White": checkmate / 2, "Checkmate Black": checkmate / 2,
    }


def choose_move(rng: random.Random, board: Board) -> Board | None:
    """Pick the next position of a random game, preferring mates, then checks.

    Without this preference, mates would almost never happen in random games.
    """
    moves = [after for _, after in legal_moves(board)]
    if not moves:
        return None
    checks = [after for after in moves if after.in_check(after.active_color)]
    mates = [after for after in checks if not any(True for _ in legal_moves(after))]
    if mates and rng.random() < MATE_PROBABILITY:
        return rng.choice(mates)
    if checks and rng.random() < CHECK_PROBABILITY:
        return rng.choice(checks)
    return rng.choice(moves)


def play_random_game(rng: random.Random) -> Iterator[Board]:
    """Yield the positions of one random game, from the first move to its end."""
    board = Board.from_fen(START)
    for _ in range(MAX_PLIES):
        board = choose_move(rng, board)
        if board is None:
            return
        yield board


def position_key(board: Board) -> str:
    """Identify a position without its move counters, to find duplicates."""
    return " ".join(board.to_fen().split()[:4])


def generate_dataset(
    size: int, seed: int, shares: dict[str, float] | None = None
) -> tuple[list[str], int]:
    """Return lines 'FEN label' with the requested share of each label, and the games played.

    Each game gives at most one position of each label, chosen at random among its own
    positions, so that a few games do not fill the dataset with similar positions.
    """
    shares = shares or label_shares(1.0, 1.0, 1.0)
    total_share = sum(shares.values())
    targets = {label: round(size * shares.get(label, 0.0) / total_share) for label in LABELS}
    rng = random.Random(seed)
    kept: dict[str, list[str]] = {label: [] for label in LABELS}
    seen: set[str] = set()
    games = 0
    while any(len(kept[label]) < targets[label] for label in LABELS):
        games += 1
        found: dict[str, list[Board]] = {label: [] for label in LABELS}
        for board in play_random_game(rng):
            label = label_position(board)
            if label is not None and position_key(board) not in seen:
                found[label].append(board)
        for label in LABELS:
            if found[label] and len(kept[label]) < targets[label]:
                board = rng.choice(found[label])
                seen.add(position_key(board))
                kept[label].append(f"{board.to_fen()} {label}")
    lines = [line for label in LABELS for line in kept[label]]
    rng.shuffle(lines)
    return lines, games
