"""Tests of the benchmarks of the chess network and of their drawings."""

import json

import pytest

from scripts.benchmark import main as benchmark_main
from scripts.plot_benchmark import best, main as plot_main
from scripts.search_hyperparameters import configuration

FENS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate Black",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3 Check White",
    "8/8/8/8/8/8/8/k1K5 w - - 0 1 Nothing",
]
BASE = """[network]
input_size = 769
layers = [{ size = 8, activation = "relu" }, { size = 5, activation = "identity" }]
seed = 0

[training]
loss = "cross_entropy"
epochs = 50
batch_size = 4
seed = 0
patience = 2
monitor = "val_accuracy"

[optimizer]
name = "adam"
learning_rate = 0.01
"""


@pytest.fixture
def files(tmp_path):
    (tmp_path / "boards.txt").write_text("\n".join(FENS) + "\n")
    (tmp_path / "base.conf").write_text(BASE)
    (tmp_path / "bench.toml").write_text(
        'title = "Test"\n[variants]\nsmall = { hidden = [4] }\n'
        'sgd = { optimizer = { name = "sgd", learning_rate = 0.1 } }\n'
        'tanh = { activation = "tanh", initializer = "xavier_uniform" }\n')
    return tmp_path


def run(files, *extra):
    benchmark_main(["--benchmark", str(files / "bench.toml"), "--base", str(files / "base.conf"),
                    "--train", str(files / "boards.txt"),
                    "--validation", str(files / "boards.txt"), "--epochs", "3",
                    "--json", str(files / "results.json"), *extra])
    return json.loads((files / "results.json").read_text())


def test_each_variant_is_trained_the_same_number_of_epochs(files, capsys):
    data = run(files)
    assert data["title"] == "Test"
    results = data["results"]
    assert [result["config"] for result in results] == ["small", "sgd", "tanh"]
    # Early stopping of the base is not used: every curve has the same length.
    assert all(len(result["history"]["loss"]) == result["kept_epoch"] == 3
               for result in results)
    assert results[0]["parameters"] == 769 * 4 + 4 + 4 * 5 + 5
    assert results[1]["learning_rate"] == 0.1 and results[0]["learning_rate"] == 0.01
    for result in results:
        assert set(result["validation_accuracy"]) == {"5", "3", "2"}
        assert len(result["confusion"]) == 5 and sum(map(sum, result["confusion"])) == 4
        assert result["train_accuracy"] == result["history"]["accuracy"][-1]
        assert result["seconds_per_epoch"] == pytest.approx(result["seconds"] / 3)
        assert result["inference_microseconds"] > 0
    output = capsys.readouterr().out
    assert "small:\nepoch 1/3 - loss " in output
    assert "| Variant | Parameters | Train acc. |" in output and "| tanh | " in output


def test_the_variants_change_the_base(files):
    results = run(files)["results"]
    assert results[0]["params"] == {"hidden": [4]}
    data = configuration({"network": {"layers": [{"size": 8, "activation": "relu"},
                                                 {"size": 5, "activation": "identity"}],
                                      "initializer": "he"},
                          "training": {}, "optimizer": {"name": "adam"}},
                         {"initializer": "lecun", "optimizer": {"name": "sgd"},
                          "schedule": {"name": "step"}, "learning_rate": 0.2}, 3, None)
    assert data["network"]["initializer"] == "lecun"
    assert data["optimizer"] == {"name": "sgd", "learning_rate": 0.2}
    assert data["training"]["schedule"] == {"name": "step"}
    assert "patience" not in data["training"]


def test_the_early_stopping_of_the_base_is_not_used(files):
    (files / "bench.toml").write_text('[variants]\nbase = {}\n')
    history = run(files, "--epochs", "40")["results"][0]["history"]
    # The tiny set is learned within a few epochs: early stopping would stop long before 40.
    assert history["val_accuracy"][-1] == 1.0
    assert len(history["loss"]) == 40


def test_only_a_part_of_the_data_is_used(files):
    result = run(files, "--train-size", "2", "--validation-size", "3")["results"][0]
    assert sum(map(sum, result["confusion"])) == 3


def test_the_data_parts_are_the_same_for_every_variant(files):
    first = run(files, "--train-size", "3", "--seed", "1")["results"]
    second = run(files, "--train-size", "3", "--seed", "1")["results"]
    assert [result["history"]["loss"] for result in first] == [
        result["history"]["loss"] for result in second]


@pytest.mark.parametrize("content, message", [
    ('title = "x"\n', "needs a [variants] table"),
    ("[variants]\nbad = 3\n", "variant 'bad' must be a table"),
    ('[variants]\nbad = { momentum = 0.9 }\n', "unknown hyperparameter 'momentum'"),
    ('[variants]\nbad = { activation = "swish" }\n', "activation must be one of"),
])
def test_invalid_benchmarks_stop_with_a_message(files, content, message):
    (files / "bench.toml").write_text(content)
    with pytest.raises(SystemExit) as error:
        run(files)
    assert message in str(error.value.code)


def test_the_benchmark_is_drawn(files, capsys):
    pytest.importorskip("matplotlib")
    run(files)
    capsys.readouterr()
    plot_main(["--json", str(files / "results.json"), "--output",
               str(files / "images" / "test")])
    names = ["test_curves.png", "test_summary.png", "test_confusion.png"]
    assert capsys.readouterr().out.split() == [str(files / "images" / name) for name in names]
    for name in names:
        assert (files / "images" / name).read_bytes().startswith(b"\x89PNG")


def test_the_best_variant_has_the_best_validation_accuracy():
    results = [{"config": "a", "validation_accuracy": {"5": 0.5}},
               {"config": "b", "validation_accuracy": {"5": 0.7}},
               {"config": "c", "validation_accuracy": {"5": 0.7}}]
    assert best(results)["config"] == "b"


@pytest.mark.parametrize("content, message", [
    ("[]", "no benchmark results"),
    ('{"results": []}', "no benchmark results"),
    ('{"results": [{"config": "a"}]}', "every variant needs"),
    ("not json", "Expecting value"),
])
def test_invalid_results_are_refused(tmp_path, content, message):
    (tmp_path / "results.json").write_text(content)
    with pytest.raises(SystemExit) as error:
        plot_main(["--json", str(tmp_path / "results.json"), "--output", str(tmp_path / "x")])
    assert message in str(error.value.code)
