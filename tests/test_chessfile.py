"""Tests of the reading of chessboard files."""

import os

import pytest

from analyzer.chessfile import Chessboard, parse_line, read_chessfile
from analyzer.errors import AnalyzerError
from analyzer.fen import parse_fen

SUBJECT_FENS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3",
    "8/8/8/8/8/8/8/k1K5 w - - 0 1",
]
SUBJECT_LABELS = ["Checkmate Black", "Nothing", "Nothing", "Check White", "Nothing"]


def write(tmp_path, content, name="chessboards.txt"):
    path = tmp_path / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, newline="")
    return path


def test_prediction_file_of_the_subject(tmp_path):
    chessboards = read_chessfile(write(tmp_path, "\n".join(SUBJECT_FENS) + "\n"))
    assert [board.position for board in chessboards] == [parse_fen(fen) for fen in SUBJECT_FENS]
    assert [board.label for board in chessboards] == [None] * 5
    assert [board.line_number for board in chessboards] == [1, 2, 3, 4, 5]


def test_training_file_with_labels(tmp_path):
    lines = [f"{fen} {label}" for fen, label in zip(SUBJECT_FENS, SUBJECT_LABELS)]
    chessboards = read_chessfile(write(tmp_path, "\n".join(lines)), require_labels=True)
    assert [board.label for board in chessboards] == SUBJECT_LABELS
    assert all(isinstance(board, Chessboard) for board in chessboards)


def test_labels_are_optional_line_by_line_in_prediction(tmp_path):
    content = f"{SUBJECT_FENS[0]} Checkmate\n{SUBJECT_FENS[1]}\n{SUBJECT_FENS[3]} Check\n"
    labels = [board.label for board in read_chessfile(write(tmp_path, content))]
    assert labels == ["Checkmate", None, "Check"]


def test_empty_lines_are_skipped_and_line_numbers_kept(tmp_path):
    content = f"\n{SUBJECT_FENS[0]}\n   \n\t\n{SUBJECT_FENS[1]}\n\n"
    chessboards = read_chessfile(write(tmp_path, content))
    assert [board.line_number for board in chessboards] == [2, 5]


def test_extra_spaces_are_ignored(tmp_path):
    line = "  " + SUBJECT_FENS[3].replace(" ", "   ") + "   Check    White  "
    [board] = read_chessfile(write(tmp_path, line))
    assert board.position == parse_fen(SUBJECT_FENS[3])
    assert board.label == "Check White"


@pytest.mark.parametrize("content", [
    "",
    "\n\n  \n",
])
def test_file_without_chessboards(tmp_path, content):
    assert read_chessfile(write(tmp_path, content)) == []


@pytest.mark.parametrize("ending", ["\n", "\r\n"])
def test_line_endings(tmp_path, ending):
    content = ending.join(SUBJECT_FENS) + ending
    assert len(read_chessfile(write(tmp_path, content))) == 5


def test_byte_order_mark_is_ignored(tmp_path):
    content = ("﻿" + "\n".join(SUBJECT_FENS)).encode("utf-8")
    assert len(read_chessfile(write(tmp_path, content))) == 5


def test_last_line_without_newline(tmp_path):
    assert len(read_chessfile(write(tmp_path, "\n".join(SUBJECT_FENS)))) == 5


def test_parse_line_keeps_its_number():
    board = parse_line(f"{SUBJECT_FENS[0]} Checkmate Black", 42)
    assert (board.line_number, board.label) == (42, "Checkmate Black")


@pytest.mark.parametrize("lines, require_labels, line_number, message", [
    (SUBJECT_FENS, True, 1, "missing expected label"),
    ([f"{SUBJECT_FENS[0]} Nothing", SUBJECT_FENS[1]], True, 2, "missing expected label"),
    ([SUBJECT_FENS[0], f"{SUBJECT_FENS[1]} Chek"], False, 2, "unknown label 'Chek'"),
    ([f"{SUBJECT_FENS[0]} check white"], False, 1, "unknown label 'check white'"),
    ([SUBJECT_FENS[0], "", "8/8/8/8/8/8/8/k1K5 w - -"], False, 3, "expected 6 fields"),
    (["8/8/8/8/8/8/8/k1K5 w - - 0 Nothing"], False, 1, "fullmove number"),
    (["8/8/8/8/8/8/k1K5 w - - 0 1"], False, 1, "7 ranks"),
])
def test_invalid_lines_raise_an_error_with_the_file_and_line(
    tmp_path, lines, require_labels, line_number, message
):
    path = write(tmp_path, "\n".join(lines) + "\n")
    with pytest.raises(AnalyzerError) as error:
        read_chessfile(path, require_labels=require_labels)
    assert str(error.value).startswith(f"{path}:{line_number}: ")
    assert message in str(error.value)


def test_missing_file_raises_an_error(tmp_path):
    with pytest.raises(AnalyzerError, match="no such file"):
        read_chessfile(tmp_path / "missing.txt")


def test_directory_raises_an_error(tmp_path):
    with pytest.raises(AnalyzerError, match="cannot read chessboards"):
        read_chessfile(tmp_path)


def test_binary_file_raises_an_error(tmp_path):
    with pytest.raises(AnalyzerError, match="not a text file"):
        read_chessfile(write(tmp_path, b"\xff\xfe\x00\x81"))


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
def test_unreadable_file_raises_an_error(tmp_path):
    path = write(tmp_path, "\n".join(SUBJECT_FENS))
    os.chmod(path, 0)
    try:
        with pytest.raises(AnalyzerError, match="Permission denied"):
            read_chessfile(path)
    finally:
        os.chmod(path, 0o600)
