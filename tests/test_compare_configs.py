"""Tests of the comparison of configurations on the validation set."""

import json

import pytest

from scripts.compare_configs import main as compare_main

FENS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3",
    "8/8/8/8/8/8/8/k1K5 w - - 0 1",
]
LABELS = ["Checkmate Black", "Nothing", "Nothing", "Check White", "Nothing"]
CONFIG = """[network]
input_size = 769
layers = [
    {{ size = {hidden}, activation = "relu" }},
    {{ size = {outputs}, activation = "identity" }},
]
seed = 0

[training]
loss = "cross_entropy"
epochs = 100
batch_size = 5
seed = 0

[optimizer]
name = "adam"
learning_rate = 0.01
"""


@pytest.fixture
def files(tmp_path):
    lines = "\n".join(f"{fen} {label}" for fen, label in zip(FENS, LABELS)) + "\n"
    (tmp_path / "train.txt").write_text(lines)
    (tmp_path / "validation.txt").write_text(lines)
    (tmp_path / "small.conf").write_text(CONFIG.format(hidden=32, outputs=5))
    (tmp_path / "plain.conf").write_text(CONFIG.format(hidden=16, outputs=3))
    return tmp_path


def run(files, *configs, json_name=None):
    arguments = ["--train", str(files / "train.txt"), "--validation",
                 str(files / "validation.txt")]
    if json_name:
        arguments += ["--json", str(files / json_name)]
    compare_main(arguments + [str(files / config) for config in configs])


def test_each_configuration_is_trained_and_measured(files, capsys):
    run(files, "small.conf", "plain.conf", json_name="results.json")
    results = json.loads((files / "results.json").read_text())
    assert [result["config"].rsplit("/", 1)[1] for result in results] == ["small.conf",
                                                                          "plain.conf"]
    small, plain = results
    assert small["validation_accuracy"] == {"5": 1.0, "3": 1.0, "2": 1.0}
    assert set(plain["validation_accuracy"]) == {"3", "2"}
    assert len(small["history"]["val_accuracy"]) == 100
    assert small["best_validation_accuracy"] == max(small["history"]["val_accuracy"])
    assert small["history"]["val_accuracy"][small["best_epoch"] - 1] == 1.0
    table = capsys.readouterr().out
    assert "| small | 1.0000 | 1.0000 | 1.0000 | 1.0000 |" in table
    assert "| plain |" in table and "| - |" in table


def test_invalid_configuration_stops_with_a_message(files):
    (files / "bad.conf").write_text(CONFIG.format(hidden=8, outputs=4))
    with pytest.raises(SystemExit) as error:
        run(files, "bad.conf")
    assert "4 outputs cannot predict chess labels" in str(error.value.code)


def test_early_stopping_of_a_configuration_is_used(files):
    stopping = 'seed = 0\npatience = 3\nmonitor = "val_accuracy"\n\n[optimizer]'
    early = CONFIG.format(hidden=32, outputs=5).replace("seed = 0\n\n[optimizer]", stopping)
    (files / "early.conf").write_text(early)
    run(files, "early.conf", json_name="results.json")
    result = json.loads((files / "results.json").read_text())[0]
    assert len(result["history"]["val_accuracy"]) == result["best_epoch"] + 3 < 100
