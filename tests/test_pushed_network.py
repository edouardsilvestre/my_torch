"""Tests of the pre-trained network pushed in the repository, which the autograder loads."""

import shutil
import subprocess
from pathlib import Path

from analyzer.labels import CLASS_SETS, LABELS
from analyzer.predict import load_chess_network
from my_torch import load_config, load_network
from my_torch.config import check_training_settings

ROOT = Path(__file__).resolve().parent.parent
NETWORK = ROOT / "my_torch_network.nn"
CONFIG = ROOT / "configs" / "my_torch_network.conf"
ANALYZER = ROOT / "my_torch_analyzer"
SUBJECT_FENS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3",
    "8/8/8/8/8/8/8/k1K5 w - - 0 1",
]
SUBJECT_FULL_OUTPUT = ["Checkmate Black", "Nothing", "Nothing", "Check White", "Nothing"]
TRAINING_POSITIONS = 572449


def test_the_pushed_network_has_the_name_required_by_the_subject():
    assert [path.name for path in ROOT.glob("my_torch_network*.nn")] == ["my_torch_network.nn"]


def test_the_pushed_network_reads_chessboards_and_predicts_colors():
    network, classes = load_chess_network(NETWORK)
    assert network.input_size == 769
    assert classes == CLASS_SETS[5]


def test_the_pushed_network_follows_its_configuration():
    config = load_config(CONFIG)
    saved = load_network(NETWORK)
    assert saved.network.architecture == config.layers
    assert check_training_settings(saved.settings) == config.training_settings()
    assert saved.optimizer.hyperparameters() == config.create_optimizer().hyperparameters()
    assert saved.optimizer.steps == config.epochs * -(-TRAINING_POSITIONS // config.batch_size)


def test_the_real_analyzer_answers_the_example_of_the_subject(tmp_path):
    (tmp_path / "chessboards.txt").write_text("\n".join(SUBJECT_FENS) + "\n")
    result = subprocess.run([ANALYZER, "--predict", NETWORK, "chessboards.txt"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stderr == ""
    answers = result.stdout.splitlines()
    assert len(answers) == len(SUBJECT_FENS)
    assert all(answer in LABELS and answer in CLASS_SETS[5] for answer in answers)


def test_the_pushed_network_can_be_trained_further(tmp_path):
    shutil.copy(NETWORK, tmp_path / "copy.nn")
    lines = [f"{fen} {label}" for fen, label in zip(SUBJECT_FENS, SUBJECT_FULL_OUTPUT)]
    (tmp_path / "train.txt").write_text("\n".join(lines) + "\n")
    result = subprocess.run([ANALYZER, "--train", "--save", "more.nn", "copy.nn", "train.txt"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    steps = load_network(NETWORK).optimizer.steps
    one_batch_per_epoch = load_config(CONFIG).epochs
    assert load_network(tmp_path / "more.nn").optimizer.steps == steps + one_batch_per_epoch
    assert (tmp_path / "copy.nn").read_bytes() == NETWORK.read_bytes()
