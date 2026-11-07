"""Legal moves of a chess position, and the position reached by playing them."""

from collections.abc import Iterator
from dataclasses import dataclass

from dataset.board import (
    BISHOP_DIRECTIONS, KING_STEPS, KNIGHT_STEPS, ROOK_DIRECTIONS, WHITE, Board, color_of,
    opponent, own, square_index,
)

PROMOTIONS = "qrbn"
CASTLINGS = {
    # right: (king from, king to, rook from, rook to, squares that must be empty)
    "K": (60, 62, 63, 61, (61, 62)),
    "Q": (60, 58, 56, 59, (59, 58, 57)),
    "k": (4, 6, 7, 5, (5, 6)),
    "q": (4, 2, 0, 3, (3, 2, 1)),
}
ROOK_CORNERS = {63: "K", 56: "Q", 7: "k", 0: "q"}


@dataclass(frozen=True)
class Move:
    origin: int
    target: int
    promotion: str | None = None


def slide_targets(board: Board, square: int, directions, color: str) -> Iterator[int]:
    row, file = divmod(square, 8)
    for dr, df in directions:
        r, f = row + dr, file + df
        while 0 <= r < 8 and 0 <= f < 8:
            piece = board.squares[square_index(r, f)]
            if piece is None or color_of(piece) != color:
                yield square_index(r, f)
            if piece is not None:
                break
            r, f = r + dr, f + df


def step_targets(board: Board, square: int, steps, color: str) -> Iterator[int]:
    row, file = divmod(square, 8)
    for dr, df in steps:
        r, f = row + dr, file + df
        if 0 <= r < 8 and 0 <= f < 8:
            piece = board.squares[square_index(r, f)]
            if piece is None or color_of(piece) != color:
                yield square_index(r, f)


def pawn_moves(board: Board, square: int, color: str) -> Iterator[Move]:
    row, file = divmod(square, 8)
    forward, start_row, last_row = (-1, 6, 0) if color == WHITE else (1, 1, 7)

    def moves_to(target: int) -> Iterator[Move]:
        if target // 8 == last_row:
            for piece in PROMOTIONS:
                yield Move(square, target, own(piece, color))
        else:
            yield Move(square, target)

    ahead = square_index(row + forward, file)
    if board.squares[ahead] is None:
        yield from moves_to(ahead)
        two_ahead = square_index(row + 2 * forward, file) if row == start_row else None
        if two_ahead is not None and board.squares[two_ahead] is None:
            yield Move(square, two_ahead)
    for side in (-1, 1):
        if 0 <= file + side < 8:
            target = square_index(row + forward, file + side)
            piece = board.squares[target]
            if piece is not None and color_of(piece) != color:
                yield from moves_to(target)
            elif target == board.en_passant:
                yield Move(square, target)


def castling_moves(board: Board, color: str) -> Iterator[Move]:
    for right, (king_from, king_to, rook_from, _, empty) in CASTLINGS.items():
        if (right not in board.castling or color_of(right) != color
                or board.squares[king_from] != own("k", color)
                or board.squares[rook_from] != own("r", color)
                or any(board.squares[square] is not None for square in empty)):
            continue
        passed = range(king_from, king_to + (1 if king_to > king_from else -1),
                       1 if king_to > king_from else -1)
        if not any(board.is_attacked(square, opponent(color)) for square in passed):
            yield Move(king_from, king_to)


def pseudo_legal_moves(board: Board) -> Iterator[Move]:
    """Moves that follow how pieces move, without checking that the own king stays safe."""
    color = board.active_color
    for square, piece in enumerate(board.squares):
        if piece is None or color_of(piece) != color:
            continue
        kind = piece.lower()
        if kind == "p":
            yield from pawn_moves(board, square, color)
        elif kind == "n":
            targets = step_targets(board, square, KNIGHT_STEPS, color)
        elif kind == "k":
            targets = step_targets(board, square, KING_STEPS, color)
        elif kind == "r":
            targets = slide_targets(board, square, ROOK_DIRECTIONS, color)
        elif kind == "b":
            targets = slide_targets(board, square, BISHOP_DIRECTIONS, color)
        else:
            targets = slide_targets(board, square, ROOK_DIRECTIONS + BISHOP_DIRECTIONS, color)
        if kind != "p":
            yield from (Move(square, target) for target in targets)
    yield from castling_moves(board, color)


def play(board: Board, move: Move) -> Board:
    """Return the position reached by playing a move, without modifying the board."""
    after = board.copy()
    squares = after.squares
    piece = squares[move.origin]
    color = board.active_color
    captured = squares[move.target]
    if piece.lower() == "p" and move.target == board.en_passant and captured is None:
        squares[move.target + (8 if color == WHITE else -8)] = None
        captured = own("p", opponent(color))
    squares[move.target] = move.promotion or piece
    squares[move.origin] = None
    if piece.lower() == "k" and abs(move.target - move.origin) == 2:
        for king_from, king_to, rook_from, rook_to, _ in CASTLINGS.values():
            if (move.origin, move.target) == (king_from, king_to):
                squares[rook_to], squares[rook_from] = squares[rook_from], None
    rights = after.castling
    if piece.lower() == "k":
        rights = "".join(r for r in rights if color_of(r) != color)
    for corner in (move.origin, move.target):
        if corner in ROOK_CORNERS:
            rights = rights.replace(ROOK_CORNERS[corner], "")
    after.castling = rights
    double_push = piece.lower() == "p" and abs(move.target - move.origin) == 16
    after.en_passant = (move.origin + move.target) // 2 if double_push else None
    reset = piece.lower() == "p" or captured is not None
    after.halfmove_clock = 0 if reset else board.halfmove_clock + 1
    if color != WHITE:
        after.fullmove_number += 1
    after.active_color = opponent(color)
    return after


def legal_moves(board: Board) -> Iterator[tuple[Move, Board]]:
    """Each legal move with the position it leads to: the own king is never left attacked."""
    color = board.active_color
    for move in pseudo_legal_moves(board):
        after = play(board, move)
        if not after.in_check(color):
            yield move, after


def has_legal_move(board: Board) -> bool:
    return any(True for _ in legal_moves(board))


def perft(board: Board, depth: int) -> int:
    """Count the sequences of legal moves of the given length, to check the rules."""
    if depth == 0:
        return 1
    if depth == 1:
        return sum(1 for _ in legal_moves(board))
    return sum(perft(after, depth - 1) for _, after in legal_moves(board))
