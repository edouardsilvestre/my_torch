"""Tests of the rules of chess used to label the dataset.

Perft counts every sequence of legal moves of a given length. The reference totals come
from the chessprogramming wiki: a single wrong rule (castling, en passant, promotion, pin)
changes them.
"""

import pytest

from dataset.board import Board, parse_square
from dataset.moves import Move, legal_moves, perft, play

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
REFERENCES = [
    (START, [20, 400, 8902]),
    (KIWIPETE, [48, 2039, 97862]),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379]),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", [46, 2079]),
]


@pytest.mark.parametrize("fen, expected", REFERENCES)
def test_perft_matches_the_reference_totals(fen, expected):
    board = Board.from_fen(fen)
    assert [perft(board, depth) for depth in range(1, len(expected) + 1)] == expected


def test_fen_round_trip():
    for fen, _ in REFERENCES:
        assert Board.from_fen(fen).to_fen() == fen


def moves_from(fen, origin_name, target_name):
    board = Board.from_fen(fen)
    origin, target = parse_square(origin_name), parse_square(target_name)
    return board, [move for move, _ in legal_moves(board)
                   if (move.origin, move.target) == (origin, target)]


def test_castling_moves_the_rook_and_removes_the_rights():
    board, [move] = moves_from(KIWIPETE, "e1", "g1")
    assert play(board, move).to_fen() == (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R4RK1 b kq - 1 1"
    )
    board, [move] = moves_from(KIWIPETE, "e1", "c1")
    assert play(board, move).to_fen() == (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/2KR3R b kq - 1 1"
    )


def test_castling_is_refused_through_an_attacked_square():
    _, moves = moves_from("r3k2r/8/8/8/8/8/8/R3K1r1 w KQkq - 0 1", "e1", "g1")
    assert moves == []


def test_moving_a_rook_or_capturing_it_removes_its_right():
    board, [move] = moves_from(KIWIPETE, "a1", "b1")
    assert play(board, move).castling == "Kkq"
    board, [move] = moves_from("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "a1", "a8")
    assert play(board, move).castling == "Kk"


def test_double_push_sets_the_en_passant_square():
    board, [move] = moves_from(START, "e2", "e4")
    assert play(board, move).to_fen() == (
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )


def test_en_passant_removes_the_captured_pawn():
    board, [move] = moves_from("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5", "d6")
    assert play(board, move).to_fen() == "4k3/8/3P4/8/8/8/8/4K3 b - - 0 1"


def test_promotions_give_four_moves():
    board, moves = moves_from("8/P6k/8/8/8/8/8/K7 w - - 0 1", "a7", "a8")
    assert sorted(move.promotion for move in moves) == ["B", "N", "Q", "R"]
    assert play(board, Move(moves[0].origin, moves[0].target, "N")).squares[0] == "N"


def test_pinned_piece_cannot_leave_the_line():
    _, moves = moves_from("4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1", "e2", "c3")
    assert moves == []


def test_counters():
    board, [move] = moves_from(START, "g1", "f3")
    after = play(board, move)
    assert (after.halfmove_clock, after.fullmove_number, after.active_color) == (1, 1, "b")
    _, [reply] = moves_from(after.to_fen(), "g8", "f6")
    assert play(after, reply).fullmove_number == 2
