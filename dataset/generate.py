"""Generation of a balanced dataset of labelled positions, from random games."""

import random
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor

from dataset.board import Board
from dataset.labels import label_position
from dataset.moves import legal_moves

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
LABELS = ("Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black")
MAX_PLIES = 300
MATE_PROBABILITY = 0.9
CHECK_PROBABILITY = 0.3
CHUNK_SIZE = 5000


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


def share_out(total: int, weights: list[float]) -> list[int]:
    """Share a total in proportion to the weights, so that the parts always add up to it.

    Each part first gets the integer part of its share; what is left goes to the parts
    whose share lost the most when rounded down.
    """
    exact = [total * weight / sum(weights) for weight in weights]
    counts = [int(value) for value in exact]
    by_loss = sorted(range(len(weights)), key=lambda i: exact[i] - counts[i], reverse=True)
    for index in by_loss[:total - sum(counts)]:
        counts[index] += 1
    return counts


def split_size(size: int, shares: dict[str, float]) -> dict[str, int]:
    """Share the size between the labels, so that the counts always add up to the size."""
    counts = share_out(size, [shares.get(label, 0.0) for label in LABELS])
    return dict(zip(LABELS, counts))


def position_key(board: Board) -> str:
    """Identify a position without its move counters, to find duplicates."""
    return " ".join(board.to_fen().split()[:4])


def generate_chunk(
    targets: dict[str, int], seed: int | str, seen: set[str] | None = None
) -> tuple[list[str], int]:
    """Generate lines 'FEN label' with the given number of each label, and count the games.

    Each game gives at most one position of each label, chosen at random among its own
    positions, so that a few games do not fill the dataset with similar positions. The
    positions in seen are never kept.
    """
    rng = random.Random(seed)
    kept: dict[str, list[str]] = {label: [] for label in LABELS}
    seen = set(seen or ())
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


def line_key(line: str) -> str:
    """Identify the position of a line 'FEN label' without its move counters."""
    return " ".join(line.split()[:4])


def generate_dataset(
    size: int, seed: int, shares: dict[str, float] | None = None, jobs: int = 1,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[list[str], int]:
    """Return lines 'FEN label' with the requested share of each label, and the games played.

    The dataset is made of chunks of at most chunk_size positions, each generated from its
    own seed, possibly by several processes at once. The chunks only depend on the size and
    the seed: the dataset is the same whatever the number of jobs. A dataset made of a single
    chunk uses the seed itself.
    """
    shares = shares or label_shares(1.0, 1.0, 1.0)
    targets = split_size(size, shares)
    chunks = max(1, -(-size // chunk_size))
    plans = [dict(zip(LABELS, counts)) for counts in zip(
        *(share_out(targets[label], [1.0] * chunks) for label in LABELS)
    )]
    seeds = [seed] + [f"{seed}-{index}" for index in range(1, chunks)]
    if jobs > 1 and chunks > 1:
        with ProcessPoolExecutor(min(jobs, chunks)) as pool:
            results = list(pool.map(generate_chunk, plans, seeds))
    else:
        results = [generate_chunk(plan, chunk_seed) for plan, chunk_seed in zip(plans, seeds)]
    if chunks == 1:
        return results[0]
    return merge_chunks(results, targets, seed)


def merge_chunks(
    results: list[tuple[list[str], int]], targets: dict[str, int], seed: int
) -> tuple[list[str], int]:
    """Join the chunks, drop the positions found by two chunks, and replace them."""
    lines, seen = [], set()
    games = sum(chunk_games for _, chunk_games in results)
    for chunk_lines, _ in results:
        for line in chunk_lines:
            if line_key(line) not in seen:
                seen.add(line_key(line))
                lines.append(line)
    refill = 0
    while True:
        counts = Counter(line.split(" ", 6)[6] for line in lines)
        missing = {label: targets[label] - counts[label] for label in LABELS}
        if not any(missing.values()):
            break
        extra, extra_games = generate_chunk(missing, f"{seed}-refill-{refill}", seen)
        seen.update(line_key(line) for line in extra)
        lines += extra
        games += extra_games
        refill += 1
    random.Random(f"{seed}-merge").shuffle(lines)
    return lines, games
