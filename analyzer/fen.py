"""Reading of chessboards written in Forsyth-Edwards Notation (FEN)."""

import re
from dataclasses import dataclass

from analyzer.errors import AnalyzerError

PIECES = "PNBRQKpnbrqk"
COLORS = ("w", "b")
EMPTY_DIGITS = "12345678"
CASTLING_PATTERN = re.compile(r"-|(?!.*(.).*\1)[KQkq]{1,4}")
EN_PASSANT_PATTERN = re.compile(r"-|[a-h][36]")
NUMBER_PATTERN = re.compile(r"[0-9]+")


@dataclass(frozen=True)
class Position:
    """Chess position read from a FEN.

    The board holds 8 ranks, from rank 8 to rank 1, of 8 squares each, from file a
    to file h. A square holds a piece letter, or None when it is empty.
    """

    board: tuple[tuple[str | None, ...], ...]
    active_color: str
    castling: str
    en_passant: str | None
    halfmove_clock: int
    fullmove_number: int


def parse_rank(text: str, rank: int) -> tuple[str | None, ...]:
    """Read one rank of the piece placement, such as 'rnbqkbnr' or '4P3'."""
    squares: list[str | None] = []
    previous_was_digit = False
    for character in text:
        if character in PIECES:
            squares.append(character)
            previous_was_digit = False
        elif character in EMPTY_DIGITS:
            if previous_was_digit:
                raise ValueError(f"rank {rank} has two digits in a row")
            squares.extend([None] * int(character))
            previous_was_digit = True
        else:
            raise ValueError(f"rank {rank} holds an unknown character '{character}'")
    if len(squares) != 8:
        raise ValueError(f"rank {rank} describes {len(squares)} squares instead of 8")
    return tuple(squares)


def parse_board(text: str) -> tuple[tuple[str | None, ...], ...]:
    ranks = text.split("/")
    if len(ranks) != 8:
        raise ValueError(f"the piece placement has {len(ranks)} ranks instead of 8")
    return tuple(parse_rank(rank_text, 8 - index) for index, rank_text in enumerate(ranks))


def parse_number(text: str, name: str) -> int:
    if not NUMBER_PATTERN.fullmatch(text):
        raise ValueError(f"the {name} must be a non-negative integer, got '{text}'")
    return int(text)


def parse_fen(text: str) -> Position:
    """Read a FEN and check its syntax, without checking that the position is legal."""
    fields = text.split()
    try:
        if len(fields) != 6:
            raise ValueError(f"expected 6 fields separated by spaces, got {len(fields)}")
        placement, active_color, castling, en_passant, halfmove, fullmove = fields
        board = parse_board(placement)
        if active_color not in COLORS:
            raise ValueError(f"the active color must be 'w' or 'b', got '{active_color}'")
        if not CASTLING_PATTERN.fullmatch(castling):
            raise ValueError(f"invalid castling rights '{castling}'")
        if not EN_PASSANT_PATTERN.fullmatch(en_passant):
            raise ValueError(f"invalid en passant square '{en_passant}'")
        position = Position(
            board, active_color, castling, None if en_passant == "-" else en_passant,
            parse_number(halfmove, "halfmove clock"),
            parse_number(fullmove, "fullmove number"),
        )
    except ValueError as error:
        raise AnalyzerError(f"invalid FEN '{text}': {error}") from error
    return position
