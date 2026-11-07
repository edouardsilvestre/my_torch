"""Tests of the training mode of my_torch_analyzer."""

import os

import pytest

import analyzer.train as training
from analyzer.errors import AnalyzerError
from analyzer.main import main
from analyzer.train import load_training_network, read_training_data
from my_torch import Network, get_loss, get_optimizer, load_network, save_network

FENS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3",
    "8/8/8/8/8/8/8/k1K5 w - - 0 1",
]
LABELS = ["Checkmate Black", "Nothing", "Nothing", "Check White", "Nothing"]
EXPECTED = {
    5: LABELS,
    3: ["Checkmate", "Nothing", "Nothing", "Check", "Nothing"],
    2: ["Check", "Nothing", "Nothing", "Check", "Nothing"],
}
SETTINGS = {"epochs": 100, "batch_size": 5, "shuffle": True, "seed": 0}
CHECKED_SETTINGS = {**SETTINGS, "l2": 0.0, "patience": None, "monitor": "val_loss",
                    "validation_split": 0.0}


def save_trainable(path, outputs=5, settings=SETTINGS):
    network = Network(769, [(32, "relu"), (outputs, "identity")], seed=0)
    save_network(path, network, get_loss("cross_entropy"),
                 get_optimizer("adam", learning_rate=0.01), settings)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run in a directory holding trainable networks, labelled and unlabelled chessboards."""
    monkeypatch.chdir(tmp_path)
    for outputs in (2, 3, 5):
        save_trainable(tmp_path / f"net{outputs}.nn", outputs)
    lines = [f"{fen} {label}" for fen, label in zip(FENS, LABELS)]
    (tmp_path / "train.txt").write_text("\n".join(lines) + "\n")
    (tmp_path / "boards.txt").write_text("\n".join(FENS) + "\n")
    return tmp_path


def predictions(capsys, network_file):
    capsys.readouterr()
    assert main(["--predict", network_file, "boards.txt"]) == 0
    return capsys.readouterr().out.splitlines()


@pytest.mark.parametrize("outputs", [2, 3, 5])
def test_trained_network_predicts_its_labels(workdir, capsys, outputs):
    assert main(["--train", f"net{outputs}.nn", "train.txt"]) == 0
    assert capsys.readouterr().err == ""
    assert predictions(capsys, f"net{outputs}.nn") == EXPECTED[outputs]


def test_progress_is_printed_once_per_epoch(workdir, capsys):
    main(["--train", "net5.nn", "train.txt"])
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 100
    assert lines[0].startswith("epoch   1/100 - loss ")
    assert lines[-1].startswith("epoch 100/100 - loss ")
    assert "accuracy" in lines[-1]
    assert captured.err == ""


def test_save_keeps_the_loaded_file(workdir, capsys):
    original = (workdir / "net5.nn").read_bytes()
    assert main(["--train", "--save", "trained.nn", "net5.nn", "train.txt"]) == 0
    assert (workdir / "net5.nn").read_bytes() == original
    assert predictions(capsys, "trained.nn") == LABELS


def test_without_save_the_loaded_file_is_replaced(workdir):
    original = (workdir / "net5.nn").read_bytes()
    assert main(["--train", "net5.nn", "train.txt"]) == 0
    assert (workdir / "net5.nn").read_bytes() != original
    assert not [name for name in os.listdir(workdir) if name.startswith(".")]


def test_saved_network_keeps_what_is_needed_to_train_again(workdir):
    main(["--train", "--save", "trained.nn", "net5.nn", "train.txt"])
    saved = load_network("trained.nn")
    assert saved.optimizer.steps == 100
    assert saved.settings == CHECKED_SETTINGS
    assert saved.optimizer.hyperparameters()["learning_rate"] == 0.01
    main(["--train", "trained.nn", "train.txt"])
    assert load_network("trained.nn").optimizer.steps == 200


def test_training_follows_the_saved_settings(workdir):
    save_trainable(workdir / "short.nn", settings={"epochs": 3, "batch_size": 2})
    main(["--train", "short.nn", "train.txt"])
    assert load_network("short.nn").optimizer.steps == 3 * 3


def test_training_data_by_hand(workdir):
    saved, classes, settings = load_training_network("net3.nn")
    inputs, targets = read_training_data("train.txt", classes)
    assert inputs.shape == (5, 769)
    assert targets.tolist() == [[0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]]
    assert settings == CHECKED_SETTINGS


@pytest.mark.parametrize("parts, message", [
    ({"loss": None, "optimizer": None, "settings": None},
     "has no loss, optimizer, training settings: it cannot be trained"),
    ({"optimizer": None}, "has no optimizer"),
    ({"settings": {"epochs": 0, "batch_size": 4}}, "settings.epochs must be a positive integer"),
])
def test_network_without_valid_training_information(workdir, capsys, parts, message):
    values = {"loss": get_loss("cross_entropy"), "optimizer": get_optimizer("adam"),
              "settings": SETTINGS}
    values.update(parts)
    save_network(workdir / "incomplete.nn", Network(769, [(3, "identity")]), **values)
    original = (workdir / "incomplete.nn").read_bytes()
    assert main(["--train", "incomplete.nn", "train.txt"]) == 84
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err
    assert (workdir / "incomplete.nn").read_bytes() == original


@pytest.mark.parametrize("content, network, message", [
    (f"{FENS[0]} Nothing\n{FENS[1]}\n", "net5.nn", "train.txt:2: missing expected label"),
    (f"{FENS[0]} Nothing\n\n{FENS[1]} Check\n", "net5.nn",
     "train.txt:3: label 'Check' does not tell"),
    (f"{FENS[0]} Chek\n", "net3.nn", "train.txt:1: unknown label 'Chek'"),
    ("\n\n", "net3.nn", "no labelled chessboard in 'train.txt'"),
])
def test_invalid_training_files_change_nothing(workdir, capsys, content, network, message):
    (workdir / "train.txt").write_text(content)
    original = (workdir / network).read_bytes()
    assert main(["--train", network, "train.txt"]) == 84
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err
    assert (workdir / network).read_bytes() == original


def test_unsuitable_network_is_refused(workdir, capsys):
    save_network(workdir / "small.nn", Network(768, [(3, "identity")]), get_loss("mse"),
                 get_optimizer("sgd"), SETTINGS)
    assert main(["--train", "small.nn", "train.txt"]) == 84
    assert "network expects 768 inputs" in capsys.readouterr().err


def test_interruption_saves_nothing(workdir, capsys, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    original = (workdir / "net5.nn").read_bytes()
    monkeypatch.setattr(training, "train", interrupted)
    assert main(["--train", "--save", "trained.nn", "net5.nn", "train.txt"]) == 84
    assert main(["--train", "net5.nn", "train.txt"]) == 84
    captured = capsys.readouterr()
    assert captured.err == "my_torch_analyzer: interrupted, nothing was saved\n" * 2
    assert (workdir / "net5.nn").read_bytes() == original
    assert not (workdir / "trained.nn").exists()


def test_interruption_during_prediction(workdir, capsys, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("analyzer.main.predict", interrupted)
    assert main(["--predict", "net5.nn", "boards.txt"]) == 84
    assert capsys.readouterr().err == "my_torch_analyzer: interrupted, nothing was saved\n"


def test_errors_are_raised_as_analyzer_errors(workdir):
    with pytest.raises(AnalyzerError):
        read_training_data("boards.txt", ("Nothing", "Check"))


def test_closed_output_during_training_is_an_error(workdir, capsys, monkeypatch):
    def closed_output(*args, **kwargs):
        raise BrokenPipeError(32, "Broken pipe")

    original = (workdir / "net5.nn").read_bytes()
    monkeypatch.setattr(training, "train", closed_output)
    monkeypatch.setattr("analyzer.main.silence_standard_output", lambda: None)
    assert main(["--train", "net5.nn", "train.txt"]) == 84
    assert capsys.readouterr().err == (
        "my_torch_analyzer: the standard output was closed during training, nothing was saved\n"
    )
    assert (workdir / "net5.nn").read_bytes() == original
