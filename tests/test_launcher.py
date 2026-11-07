"""Tests of the real executables, run as separate processes."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ANALYZER = ROOT / "my_torch_analyzer"
GENERATOR = ROOT / "my_torch_generator"


def run_binary(binary, cwd, *arguments):
    return subprocess.run([binary, *arguments], cwd=cwd, capture_output=True, text=True)


def test_binary_runs_from_another_directory(tmp_path):
    result = run_binary(ANALYZER, tmp_path, "--help")
    assert result.returncode == 0
    assert result.stdout.startswith("USAGE\n        ./my_torch_analyzer [--predict")
    assert result.stderr == ""


@pytest.mark.parametrize("binary, package", [
    (ANALYZER, "analyzer"),
    (GENERATOR, "my_torch"),
])
def test_failed_import_returns_84(tmp_path, binary, package):
    isolated_binary = tmp_path / binary.name
    shutil.copy(binary, isolated_binary)
    result = run_binary(isolated_binary, tmp_path)
    assert result.returncode == 84
    assert result.stdout == ""
    assert f"No module named '{package}'" in result.stderr


def test_generator_help(tmp_path):
    result = run_binary(GENERATOR, tmp_path, "--help")
    assert result.returncode == 0
    assert result.stdout.startswith("USAGE\n        ./my_torch_generator config_file_1 nb_1")
    assert result.stderr == ""


def test_generator_example_of_the_subject(tmp_path):
    shutil.copy(ROOT / "configs" / "basic_network.conf", tmp_path)
    result = run_binary(GENERATOR, tmp_path, "basic_network.conf", "3")
    assert result.returncode == 0
    assert result.stdout == "" and result.stderr == ""
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "basic_network.conf", "basic_network_1.nn", "basic_network_2.nn", "basic_network_3.nn",
    ]


@pytest.mark.parametrize("arguments", [
    (),
    ("basic_network.conf",),
    ("basic_network.conf", "zero"),
    ("invalid.conf", "1"),
    ("missing.conf", "1"),
])
def test_generator_errors_return_84(tmp_path, arguments):
    (tmp_path / "invalid.conf").write_text("[network]\ninput_size = 0\n")
    shutil.copy(ROOT / "configs" / "basic_network.conf", tmp_path)
    result = run_binary(GENERATOR, tmp_path, *arguments)
    assert result.returncode == 84
    assert result.stdout == ""
    assert result.stderr.startswith("my_torch_generator: ")
    assert not any(path.suffix == ".nn" for path in tmp_path.iterdir())


def test_analyzer_help(tmp_path):
    result = run_binary(ANALYZER, tmp_path, "--help")
    assert result.returncode == 0
    assert result.stdout.startswith("USAGE\n        ./my_torch_analyzer [--predict | --train")
    assert result.stderr == ""


@pytest.mark.parametrize("arguments, message", [
    ((), "missing arguments"),
    (("net.nn", "boards.txt"), "missing mode"),
    (("--predict", "--train", "net.nn", "boards.txt"), "cannot be used together"),
    (("--predict", "--save", "out.nn", "net.nn", "boards.txt"), "only works with --train"),
    (("--predict", "missing.nn", "boards.txt"), "no such file"),
])
def test_analyzer_errors_return_84(tmp_path, arguments, message):
    (tmp_path / "net.nn").write_text("network")
    (tmp_path / "boards.txt").write_text("chessboards")
    result = run_binary(ANALYZER, tmp_path, *arguments)
    assert result.returncode == 84
    assert result.stdout == ""
    assert result.stderr.startswith("my_torch_analyzer: ")
    assert message in result.stderr.splitlines()[0]


SUBJECT_FENS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3",
    "8/8/8/8/8/8/8/k1K5 w - - 0 1",
]


@pytest.fixture
def generated_network(tmp_path):
    """Generate a blank chess network with my_torch_generator, and write chessboards."""
    shutil.copy(ROOT / "configs" / "basic_network.conf", tmp_path)
    assert run_binary(GENERATOR, tmp_path, "basic_network.conf", "1").returncode == 0
    (tmp_path / "chessboards.txt").write_text("\n".join(SUBJECT_FENS) + "\n")
    return tmp_path


def test_predict_example_of_the_subject(generated_network):
    result = run_binary(ANALYZER, generated_network, "--predict", "basic_network_1.nn",
                        "chessboards.txt")
    assert result.returncode == 0
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    assert len(lines) == 5
    assert set(lines) <= {"Nothing", "Check", "Checkmate"}


def test_predict_into_a_closed_pipe(generated_network):
    (generated_network / "chessboards.txt").write_text("\n".join(SUBJECT_FENS * 4000) + "\n")
    process = subprocess.Popen(
        [ANALYZER, "--predict", "basic_network_1.nn", "chessboards.txt"],
        cwd=generated_network, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    lines = [process.stdout.readline() for _ in range(5)]
    process.stdout.close()
    error_output = process.stderr.read()
    process.stderr.close()
    assert process.wait() == 0
    assert error_output == ""
    assert all(line.strip() in {"Nothing", "Check", "Checkmate"} for line in lines)


def test_predict_error_returns_84(generated_network):
    (generated_network / "chessboards.txt").write_text("8/8/8 w - - 0 1\n")
    result = run_binary(ANALYZER, generated_network, "--predict", "basic_network_1.nn",
                        "chessboards.txt")
    assert result.returncode == 84
    assert result.stdout == ""
    assert result.stderr.startswith("my_torch_analyzer: chessboards.txt:1: invalid FEN")


def test_predict_into_a_pipe_closed_before_any_output(generated_network):
    process = subprocess.Popen(
        [ANALYZER, "--predict", "basic_network_1.nn", "chessboards.txt"],
        cwd=generated_network, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    process.stdout.close()
    error_output = process.stderr.read()
    process.stderr.close()
    assert process.wait() == 0
    assert error_output == ""
