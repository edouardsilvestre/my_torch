"""Tests of the evaluation of a chess network on labelled chessboards."""

import json

import numpy as np
import pytest

from analyzer.errors import AnalyzerError
from analyzer.evaluate import evaluate, reduce_classes
from analyzer.labels import CLASS_SETS
from analyzer.predict import predict_classes
from my_torch import Network, get_loss, get_optimizer, save_network, train
from analyzer.train import read_training_data
from scripts.evaluate import main as evaluate_main

FENS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3",
    "8/8/8/8/8/8/8/k1K5 w - - 0 1",
]
LABELS = ["Checkmate Black", "Nothing", "Nothing", "Check White", "Nothing"]


@pytest.fixture
def files(tmp_path):
    """A labelled file, a network that learned it, and one that predicts Nothing everywhere."""
    chessboards = tmp_path / "test.txt"
    chessboards.write_text("\n".join(f"{fen} {label}" for fen, label in zip(FENS, LABELS)) + "\n")
    inputs, targets = read_training_data(chessboards, CLASS_SETS[5])
    learned = Network(769, [(32, "relu"), (5, "identity")], seed=0)
    train(learned, get_loss("cross_entropy"), get_optimizer("adam", learning_rate=0.01),
          inputs, targets, 100, 5, seed=0)
    save_network(tmp_path / "learned.nn", learned)
    nothing = Network(769, [(5, "identity")])
    nothing.layers[0].biases = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
    save_network(tmp_path / "nothing.nn", nothing)
    return tmp_path


@pytest.mark.parametrize("classes, fewer, expected", [
    (5, 3, [0, 1, 1, 2, 2]),
    (5, 2, [0, 1, 1, 1, 1]),
    (3, 2, [0, 1, 1]),
    (5, 5, [0, 1, 2, 3, 4]),
])
def test_reduce_classes(classes, fewer, expected):
    indices = np.arange(classes)
    assert reduce_classes(indices, CLASS_SETS[classes], CLASS_SETS[fewer]).tolist() == expected


def test_learned_network_is_perfect_at_every_level(files):
    levels = evaluate(files / "learned.nn", files / "test.txt")
    assert [len(level.classes) for level in levels] == [5, 3, 2]
    assert all(level.metrics.accuracy == 1.0 for level in levels)


def test_network_that_always_says_nothing(files):
    five, three, two = evaluate(files / "nothing.nn", files / "test.txt")
    assert five.metrics.accuracy == pytest.approx(3 / 5)
    assert five.metrics.confusion[:, 0].tolist() == [3, 1, 0, 0, 1]
    assert five.metrics.per_class[0].recall == 1.0
    assert five.metrics.per_class[0].precision == pytest.approx(3 / 5)
    assert three.metrics.confusion.tolist() == [[3, 0, 0], [1, 0, 0], [1, 0, 0]]
    assert two.metrics.confusion.tolist() == [[3, 0], [2, 0]]


def test_a_wrong_color_is_only_wrong_at_the_level_with_colors(files):
    (files / "black.txt").write_text("4k3/8/8/8/8/8/8/4K2r w - - 0 1 Check Black\n")
    white = Network(769, [(5, "identity")])
    white.layers[0].biases = np.array([0.0, 5.0, 0.0, 0.0, 0.0])
    save_network(files / "white.nn", white)
    five, three, two = evaluate(files / "white.nn", files / "black.txt")
    assert five.metrics.confusion[2].tolist() == [0, 1, 0, 0, 0]
    assert (five.metrics.accuracy, three.metrics.accuracy, two.metrics.accuracy) == (0, 1, 1)
    assert three.metrics.confusion.tolist() == [[0, 0, 0], [0, 1, 0], [0, 0, 0]]


def test_a_network_without_colors_has_fewer_levels(files):
    save_network(files / "three.nn", Network(769, [(3, "identity")]))
    levels = evaluate(files / "three.nn", files / "test.txt")
    assert [level.classes for level in levels] == [CLASS_SETS[3], CLASS_SETS[2]]


def test_labels_without_colors_cannot_evaluate_a_network_with_colors(files):
    (files / "plain.txt").write_text(f"{FENS[3]} Check\n")
    with pytest.raises(AnalyzerError, match="plain.txt:1: label 'Check' does not tell"):
        evaluate(files / "learned.nn", files / "plain.txt")


@pytest.mark.parametrize("content, message", [
    ("\n", "no labelled chessboard"),
    (f"{FENS[0]}\n", "missing expected label"),
])
def test_invalid_files(files, content, message):
    (files / "bad.txt").write_text(content)
    with pytest.raises(AnalyzerError, match=message):
        evaluate(files / "learned.nn", files / "bad.txt")


def test_unsuitable_network(files):
    save_network(files / "small.nn", Network(768, [(3, "identity")]))
    with pytest.raises(AnalyzerError, match="768 inputs"):
        evaluate(files / "small.nn", files / "test.txt")


def test_report(files, capsys):
    evaluate_main(["--network", str(files / "nothing.nn"), "--chessboards",
                   str(files / "test.txt")])
    report = capsys.readouterr().out
    assert "5 classes: accuracy 0.6000 (3 / 5)" in report
    assert "3 classes: accuracy 0.6000 (3 / 5)" in report
    assert "2 classes: accuracy 0.6000 (3 / 5)" in report
    assert "    Nothing              0.6000    1.0000    0.7500         3" in report
    assert "Confusion matrix (rows: expected, columns: predicted)" in report
    assert f"    {'Check':17}{2:17d}{0:17d}" in report.splitlines()


def test_json_report(files):
    output = files / "report.json"
    evaluate_main(["--network", str(files / "nothing.nn"), "--chessboards",
                   str(files / "test.txt"), "--json", str(output)])
    report = json.loads(output.read_text())
    assert [level["accuracy"] for level in report["levels"]] == [0.6, 0.6, 0.6]
    first = report["levels"][0]
    assert first["classes"][0] == "Nothing"
    assert first["per_class"]["Nothing"] == pytest.approx(
        {"precision": 0.6, "recall": 1.0, "f1": 0.75, "support": 3}
    )
    assert first["confusion"][1] == [1, 0, 0, 0, 0]


def test_script_error_exits_with_a_message(files):
    with pytest.raises(SystemExit) as error:
        evaluate_main(["--network", str(files / "missing.nn"), "--chessboards",
                       str(files / "test.txt")])
    assert "evaluate: cannot load network: no such file" in str(error.value.code)


def test_mc_dropout_report(files, capsys):
    inputs, targets = read_training_data(files / "test.txt", CLASS_SETS[5])
    network = Network(769, [(64, "relu"), (5, "identity")], seed=2, dropout=[0.8, 0.0])
    save_network(files / "dropout.nn", network)
    expected = predict_classes(network, inputs, 3)
    # Chosen so that MC dropout and a single pass without dropout disagree here.
    assert not np.array_equal(expected, predict_classes(network, inputs))
    output = files / "report.json"
    evaluate_main(["--network", str(files / "dropout.nn"), "--chessboards",
                   str(files / "test.txt"), "--json", str(output), "--mc-passes", "3"])
    assert "dropout.nn on " in capsys.readouterr().out.splitlines()[0]
    report = json.loads(output.read_text())
    assert report["mc_passes"] == 3
    accuracy = np.mean(expected == np.argmax(targets, axis=1))
    assert report["levels"][0]["accuracy"] == pytest.approx(accuracy)


def test_mc_dropout_report_header(files, capsys):
    network = Network(769, [(8, "relu"), (5, "identity")], seed=0, dropout=[0.5, 0.0])
    save_network(files / "dropout.nn", network)
    evaluate_main(["--network", str(files / "dropout.nn"), "--chessboards",
                   str(files / "test.txt"), "--mc-passes", "4"])
    assert capsys.readouterr().out.splitlines()[0].endswith(" with MC dropout (4 passes)")


@pytest.mark.parametrize("passes, message", [
    ("3", "MC dropout needs a network with dropout"),
    ("-1", "passes must be a positive integer"),
])
def test_invalid_mc_dropout_evaluation(files, passes, message):
    with pytest.raises(SystemExit) as error:
        evaluate_main(["--network", str(files / "learned.nn"), "--chessboards",
                       str(files / "test.txt"), "--mc-passes", passes])
    assert message in str(error.value.code)
