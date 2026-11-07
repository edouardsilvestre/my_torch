"""Tests of the real executables, run as separate processes."""

import select
import shutil
import signal
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


TRAINING_CONFIG = """[network]
input_size = 769
layers = [{ size = 32, activation = "relu" }, { size = 5, activation = "identity" }]
seed = 0

[training]
loss = "cross_entropy"
epochs = EPOCHS
batch_size = 5
seed = 0

[optimizer]
name = "adam"
learning_rate = 0.01
"""
SUBJECT_LABELS = ["Checkmate Black", "Nothing", "Nothing", "Check White", "Nothing"]


def prepare_training(directory, epochs):
    (directory / "colors.conf").write_text(TRAINING_CONFIG.replace("EPOCHS", str(epochs)))
    assert run_binary(GENERATOR, directory, "colors.conf", "1").returncode == 0
    lines = [f"{fen} {label}" for fen, label in zip(SUBJECT_FENS, SUBJECT_LABELS)]
    (directory / "train.txt").write_text("\n".join(lines) + "\n")
    (directory / "boards.txt").write_text("\n".join(SUBJECT_FENS) + "\n")


def test_generate_train_then_predict(tmp_path):
    prepare_training(tmp_path, 100)
    trained = run_binary(ANALYZER, tmp_path, "--train", "--save", "trained.nn", "colors_1.nn",
                         "train.txt")
    assert trained.returncode == 0
    assert trained.stderr == ""
    assert len(trained.stdout.splitlines()) == 100
    predicted = run_binary(ANALYZER, tmp_path, "--predict", "trained.nn", "boards.txt")
    assert predicted.returncode == 0
    assert predicted.stdout.splitlines() == SUBJECT_LABELS


@pytest.mark.parametrize("arguments", [
    ("--train", "colors_1.nn", "train.txt"),
    ("--train", "--save", "trained.nn", "colors_1.nn", "train.txt"),
])
def test_ctrl_c_during_training(tmp_path, arguments):
    prepare_training(tmp_path, 1000000)
    original = (tmp_path / "colors_1.nn").read_bytes()
    process = subprocess.Popen([ANALYZER, *arguments], cwd=tmp_path, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        ready, _, _ = select.select([process.stdout], [], [], 30)
        assert ready, "no progress line within 30 seconds"
        assert process.stdout.readline().startswith("epoch")
        process.send_signal(signal.SIGINT)
        _, error_output = process.communicate(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode == 84
    assert error_output == "my_torch_analyzer: interrupted, nothing was saved\n"
    assert (tmp_path / "colors_1.nn").read_bytes() == original
    assert not (tmp_path / "trained.nn").exists()
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]
