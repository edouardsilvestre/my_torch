"""Tests of the reading of FEN chessboards."""

import pytest

from analyzer.errors import AnalyzerError
from analyzer.fen import Position, parse_fen

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
SUBJECT_FENS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3",
    "8/8/8/8/8/8/8/k1K5 w - - 0 1",
]


def test_initial_position():
    position = parse_fen(START)
    assert isinstance(position, Position)
    assert position.board[0] == tuple("rnbqkbnr")
    assert position.board[1] == ("p",) * 8
    assert all(position.board[row] == (None,) * 8 for row in range(2, 6))
    assert position.board[6] == ("P",) * 8
    assert position.board[7] == tuple("RNBQKBNR")
    assert position.active_color == "w"
    assert position.castling == "KQkq"
    assert position.en_passant is None
    assert (position.halfmove_clock, position.fullmove_number) == (0, 1)


def test_almost_empty_board():
    position = parse_fen("8/8/8/8/8/8/8/k1K5 w - - 0 1")
    assert position.board[7] == ("k", None, "K", None, None, None, None, None)
    assert sum(piece is not None for rank in position.board for piece in rank) == 2
    assert position.castling == "-"


def test_digits_and_pieces_are_placed_on_the_right_files():
    position = parse_fen("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1")
    assert position.board[4] == (None, None, None, "P", None, None, None, None)
    assert position.board[6][3] is None
    assert position.active_color == "b"
    assert position.en_passant == "d3"


def test_counters_are_read():
    position = parse_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 12 34")
    assert (position.halfmove_clock, position.fullmove_number) == (12, 34)


@pytest.mark.parametrize("fen", SUBJECT_FENS)
def test_fens_of_the_subject_are_accepted(fen):
    parse_fen(fen)


@pytest.mark.parametrize("fen", [
    "8/8/8/8/8/8/8/8 w - - 0 1",
    "PPPPPPPP/8/8/8/8/8/8/pppppppp b - - 99 0",
    "kkkkkkkk/8/8/8/8/8/8/KKKKKKKK w - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w qKkQ - 0 1",
    "  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR  w   KQkq -  0 1  ",
])
def test_syntactically_valid_positions_are_accepted_even_if_illegal(fen):
    parse_fen(fen)


def test_position_cannot_be_modified():
    position = parse_fen(START)
    with pytest.raises(AttributeError):
        position.active_color = "b"


PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


@pytest.mark.parametrize("fen, message", [
    ("", "expected 6 fields"),
    (f"{PLACEMENT} w KQkq - 0", "expected 6 fields"),
    (f"{PLACEMENT} w KQkq - 0 1 Check", "expected 6 fields"),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1", "7 ranks instead of 8"),
    (f"{PLACEMENT}/8 w KQkq - 0 1", "9 ranks instead of 8"),
    ("rnbqkbnr/pppppppp/8/8/8//PPPPPPPP/RNBQKBNR w KQkq - 0 1", "rank 3 describes 0 squares"),
    ("rnbqkbnr/ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "rank 7 describes 7 squares"),
    ("rnbqkbnr/ppppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "rank 7 describes 9 squares"),
    ("rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "unknown character '9'"),
    ("rnbqkbnr/pppppppp/0p7/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "unknown character '0'"),
    ("rnbqkbnr/pppppppp/44/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "two digits in a row"),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNX w KQkq - 0 1", "rank 1 holds an unknown"),
    (f"{PLACEMENT} x KQkq - 0 1", "active color"),
    (f"{PLACEMENT} W KQkq - 0 1", "active color"),
    (f"{PLACEMENT} w KQkqK - 0 1", "castling"),
    (f"{PLACEMENT} w KK - 0 1", "castling"),
    (f"{PLACEMENT} w X - 0 1", "castling"),
    (f"{PLACEMENT} w -K - 0 1", "castling"),
    (f"{PLACEMENT} w KQkq e4 0 1", "en passant"),
    (f"{PLACEMENT} w KQkq i3 0 1", "en passant"),
    (f"{PLACEMENT} w KQkq e33 0 1", "en passant"),
    (f"{PLACEMENT} w KQkq - -1 1", "halfmove clock"),
    (f"{PLACEMENT} w KQkq - ٣ 1", "halfmove clock"),
    (f"{PLACEMENT} w KQkq - 0 x", "fullmove number"),
    (f"{PLACEMENT} w KQkq - 0 1.5", "fullmove number"),
])
def test_invalid_fens_raise_an_error_with_the_reason(fen, message):
    with pytest.raises(AnalyzerError, match="invalid FEN") as error:
        parse_fen(fen)
    assert message in str(error.value)
    assert f"'{fen}'" in str(error.value)
