"""Tests of the prediction mode of my_torch_analyzer."""

import numpy as np
import pytest

import analyzer.predict as prediction
from analyzer.chessfile import read_chessfile
from analyzer.encoding import encode_positions
from analyzer.errors import AnalyzerError
from analyzer.labels import decode_predictions, get_classes
from analyzer.main import main
from analyzer.predict import load_chess_network, predict_classes
from my_torch import MyTorchError, Network, save_network

SUBJECT_FENS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3",
    "8/8/8/8/8/8/8/k1K5 w - - 0 1",
]
SUBJECT_LABELS = ["Checkmate Black", "Nothing", "Nothing", "Check White", "Nothing"]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run in a directory holding the chessboards of the subject and networks of each size."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "boards.txt").write_text("\n".join(SUBJECT_FENS) + "\n")
    for outputs in (2, 3, 5):
        network = Network(769, [(32, "relu"), (outputs, "identity")], seed=outputs)
        save_network(tmp_path / f"net{outputs}.nn", network)
    return tmp_path


def expected_predictions(path, network_path):
    network, classes = load_chess_network(network_path)
    inputs = encode_positions(board.position for board in read_chessfile(path))
    return decode_predictions(network.predict(inputs), classes)


@pytest.mark.parametrize("outputs", [2, 3, 5])
def test_networks_with_2_3_or_5_outputs_are_accepted(workdir, outputs):
    network, classes = load_chess_network(f"net{outputs}.nn")
    assert network.input_size == 769
    assert classes == get_classes(outputs)


@pytest.mark.parametrize("input_size, outputs, message", [
    (768, 3, "network expects 768 inputs, but chessboards are encoded with 769"),
    (2, 2, "network expects 2 inputs"),
    (769, 4, "4 outputs cannot predict chess labels"),
    (769, 1, "1 outputs cannot predict chess labels"),
])
def test_unsuitable_networks_are_refused(tmp_path, input_size, outputs, message):
    save_network(tmp_path / "net.nn", Network(input_size, [(outputs, "identity")]))
    with pytest.raises(AnalyzerError, match=message):
        load_chess_network(tmp_path / "net.nn")


def test_invalid_network_file_is_refused(tmp_path):
    (tmp_path / "net.nn").write_text("not a network")
    with pytest.raises(MyTorchError, match="invalid network file"):
        load_chess_network(tmp_path / "net.nn")


def test_batches_give_the_same_predictions(monkeypatch):
    rng = np.random.default_rng(0)
    inputs = rng.integers(0, 2, size=(2500, 769)).astype(np.uint8)
    network = Network(769, [(16, "relu"), (5, "identity")], seed=0)
    monkeypatch.setattr(prediction, "PREDICTION_BATCH_SIZE", 1000)
    assert np.array_equal(predict_classes(network, inputs), network.predict(inputs))


def test_no_inputs_give_no_predictions():
    network = Network(769, [(3, "identity")])
    assert predict_classes(network, np.zeros((0, 769), dtype=np.uint8)).shape == (0,)


@pytest.mark.parametrize("outputs", [2, 3, 5])
def test_one_prediction_per_chessboard_in_order(workdir, capsys, outputs):
    assert main(["--predict", f"net{outputs}.nn", "boards.txt"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.endswith("\n")
    lines = captured.out.splitlines()
    assert lines == expected_predictions("boards.txt", f"net{outputs}.nn")
    assert len(lines) == 5
    assert set(lines) <= set(get_classes(outputs))


def test_order_follows_the_file(workdir, capsys):
    main(["--predict", "net5.nn", "boards.txt"])
    forward = capsys.readouterr().out.splitlines()
    (workdir / "reversed.txt").write_text("\n".join(reversed(SUBJECT_FENS)) + "\n")
    main(["--predict", "net5.nn", "reversed.txt"])
    assert capsys.readouterr().out.splitlines() == list(reversed(forward))


def test_expected_labels_and_empty_lines_do_not_change_the_output(workdir, capsys):
    main(["--predict", "net3.nn", "boards.txt"])
    plain = capsys.readouterr().out
    lines = [f"{fen} {label}" for fen, label in zip(SUBJECT_FENS, SUBJECT_LABELS)]
    (workdir / "labelled.txt").write_text("\n" + "\n\n".join(lines) + "\n\n")
    assert main(["--predict", "net3.nn", "labelled.txt"]) == 0
    assert capsys.readouterr().out == plain


def test_file_without_chessboards_prints_nothing(workdir, capsys):
    (workdir / "empty.txt").write_text("\n\n")
    assert main(["--predict", "net3.nn", "empty.txt"]) == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("last_line, message", [
    ("8/8/8 w - - 0 1", "boards.txt:6: invalid FEN"),
    (f"{SUBJECT_FENS[0]} Chek", "boards.txt:6: unknown label 'Chek'"),
])
def test_invalid_line_prints_no_prediction(workdir, capsys, last_line, message):
    (workdir / "boards.txt").write_text("\n".join(SUBJECT_FENS + [last_line]) + "\n")
    assert main(["--predict", "net3.nn", "boards.txt"]) == 84
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"my_torch_analyzer: {message}")


def test_unsuitable_network_returns_84(workdir, capsys):
    save_network(workdir / "small.nn", Network(768, [(3, "identity")]))
    assert main(["--predict", "small.nn", "boards.txt"]) == 84
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "my_torch_analyzer: network expects 768 inputs, but chessboards are encoded with 769\n"
    )


def test_closed_output_is_not_an_error(monkeypatch, capsys):
    silenced = []

    def closed_output(argv):
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr("analyzer.main.run", closed_output)
    monkeypatch.setattr("analyzer.main.silence_standard_output", lambda: silenced.append(True))
    assert main(["--predict", "net.nn", "boards.txt"]) == 0
    assert capsys.readouterr().err == ""
    assert silenced == [True]
