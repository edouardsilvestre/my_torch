"""Tests of the command line of my_torch_analyzer."""

import os

import pytest

from analyzer.arguments import USAGE, Arguments, UsageError, check_files, parse_arguments
from analyzer.errors import AnalyzerError
from analyzer.main import main

NOT_ROOT = pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run in a directory holding a network file and a chessboard file."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "net.nn").write_text("network")
    (tmp_path / "boards.txt").write_text("chessboards")
    (tmp_path / "folder").mkdir()
    return tmp_path


def test_usage_is_the_one_of_the_subject():
    lines = USAGE.splitlines()
    assert lines[0] == "USAGE"
    assert lines[1] == (
        "        ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE"
    )
    assert lines[3] == "DESCRIPTION"
    for option in ("--train", "--predict", "--save", "LOADFILE", "CHESSFILE"):
        assert any(line.startswith(f"        {option} ") for line in lines)
    assert "Only works in train mode." in USAGE


def test_help_prints_the_usage(capsys):
    assert main(["--help"]) == 0
    captured = capsys.readouterr()
    assert captured.out == USAGE + "\n"
    assert captured.err == ""


@pytest.mark.parametrize("argv, expected", [
    (["--predict", "net.nn", "boards.txt"], Arguments("predict", "net.nn", "boards.txt")),
    (["--train", "net.nn", "boards.txt"], Arguments("train", "net.nn", "boards.txt")),
    (["--train", "--save", "out.nn", "net.nn", "boards.txt"],
     Arguments("train", "net.nn", "boards.txt", "out.nn")),
    (["net.nn", "boards.txt", "--train", "--save", "out.nn"],
     Arguments("train", "net.nn", "boards.txt", "out.nn")),
    (["--save", "out.nn", "--train", "net.nn", "boards.txt"],
     Arguments("train", "net.nn", "boards.txt", "out.nn")),
    (["net.nn", "--predict", "boards.txt"], Arguments("predict", "net.nn", "boards.txt")),
    (["--train", "--save", "net.nn", "net.nn", "boards.txt"],
     Arguments("train", "net.nn", "boards.txt", "net.nn")),
    (["--predict", "-", "boards.txt"], Arguments("predict", "-", "boards.txt")),
])
def test_valid_command_lines(argv, expected):
    assert parse_arguments(argv) == expected


@pytest.mark.parametrize("argv, message", [
    ([], "missing arguments"),
    (["net.nn", "boards.txt"], "missing mode: --predict or --train"),
    (["--predict"], "got 0 file(s)"),
    (["--predict", "net.nn"], "got 1 file(s)"),
    (["--predict", "a", "b", "c"], "got 3 file(s)"),
    (["--predict", "--train", "a", "b"], "--predict and --train cannot be used together"),
    (["--train", "a", "b", "--predict"], "--predict and --train cannot be used together"),
    (["--predict", "--predict", "a", "b"], "--predict is given twice"),
    (["--train", "--train", "a", "b"], "--train is given twice"),
    (["--train", "--save", "x", "--save", "y", "a", "b"], "--save is given twice"),
    (["--predict", "--save", "x", "a", "b"], "--save only works with --train"),
    (["--save", "x", "a", "b"], "missing mode"),
    (["--train", "a", "b", "--save"], "--save must be followed by a file name"),
    (["--train", "--save", "--predict", "a", "b"], "--save must be followed by a file name"),
    (["--pred", "a", "b"], "unknown option '--pred'"),
    (["--PREDICT", "a", "b"], "unknown option '--PREDICT'"),
    (["--save=x", "--train", "a", "b"], "unknown option '--save=x'"),
    (["-h"], "unknown option '-h'"),
    (["--predict", "-x", "a", "b"], "unknown option '-x'"),
    (["--help", "--predict", "a", "b"], "--help must be used alone"),
    (["--predict", "a", "b", "--help"], "--help must be used alone"),
    (["--help", "--help"], "--help must be used alone"),
])
def test_invalid_command_lines_return_84_with_the_usage(capsys, argv, message):
    with pytest.raises(UsageError):
        parse_arguments(argv)
    assert main(argv) == 84
    captured = capsys.readouterr()
    assert captured.out == ""
    first_line, rest = captured.err.split("\n", 1)
    assert first_line.startswith("my_torch_analyzer: ")
    assert message in first_line
    assert rest == "\n" + USAGE + "\n"


@pytest.mark.parametrize("argv", [
    ["--predict", "net.nn", "boards.txt"],
    ["--train", "net.nn", "boards.txt"],
    ["--train", "--save", "new.nn", "net.nn", "boards.txt"],
    ["--train", "--save", "folder/new.nn", "net.nn", "boards.txt"],
])
def test_valid_files_are_accepted(workdir, argv):
    check_files(parse_arguments(argv))


@pytest.mark.parametrize("argv, message", [
    (["--predict", "missing.nn", "boards.txt"], "cannot read LOADFILE 'missing.nn': no such file"),
    (["--predict", "net.nn", "missing.txt"], "cannot read CHESSFILE 'missing.txt': no such file"),
    (["--predict", "folder", "boards.txt"], "cannot read LOADFILE 'folder': is a directory"),
    (["--predict", "net.nn", "folder"], "cannot read CHESSFILE 'folder': is a directory"),
    (["--predict", "broken.nn", "boards.txt"], "cannot read LOADFILE 'broken.nn': no such file"),
    (["--train", "--save", "folder", "net.nn", "boards.txt"],
     "cannot save the network to 'folder': is a directory"),
    (["--train", "--save", "nowhere/new.nn", "net.nn", "boards.txt"],
     "cannot save the network to 'nowhere/new.nn': no such directory"),
])
def test_file_problems_return_84_without_the_usage(workdir, capsys, argv, message):
    os.symlink("nowhere", "broken.nn")
    with pytest.raises(AnalyzerError, match=message):
        check_files(parse_arguments(argv))
    assert main(argv) == 84
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"my_torch_analyzer: {message}\n"


def test_save_file_is_not_checked_in_prediction(workdir):
    check_files(Arguments("predict", "net.nn", "boards.txt", "nowhere/new.nn"))


@NOT_ROOT
def test_unreadable_file(workdir):
    os.chmod("boards.txt", 0)
    try:
        with pytest.raises(AnalyzerError, match="CHESSFILE 'boards.txt': permission denied"):
            check_files(parse_arguments(["--predict", "net.nn", "boards.txt"]))
    finally:
        os.chmod("boards.txt", 0o600)


@NOT_ROOT
@pytest.mark.parametrize("argv, mode", [
    (["--train", "--save", "locked/new.nn", "net.nn", "boards.txt"], "train"),
    (["--train", "locked/net.nn", "boards.txt"], "train"),
    (["--predict", "locked/net.nn", "boards.txt"], "predict"),
])
def test_read_only_directory(workdir, argv, mode):
    (workdir / "locked").mkdir()
    (workdir / "locked" / "net.nn").write_text("network")
    os.chmod(workdir / "locked", 0o500)
    try:
        if mode == "train":
            with pytest.raises(AnalyzerError, match="permission denied"):
                check_files(parse_arguments(argv))
        else:
            check_files(parse_arguments(argv))
    finally:
        os.chmod(workdir / "locked", 0o700)


def test_training_is_not_available_yet(workdir, capsys):
    assert main(["--train", "net.nn", "boards.txt"]) == 84
    assert capsys.readouterr().err == "my_torch_analyzer: --train is not available yet\n"


def test_nothing_is_created(workdir):
    before = sorted(os.listdir(workdir))
    main(["--train", "--save", "new.nn", "net.nn", "boards.txt"])
    main(["--train", "--save", "nowhere/new.nn", "net.nn", "boards.txt"])
    assert sorted(os.listdir(workdir)) == before
