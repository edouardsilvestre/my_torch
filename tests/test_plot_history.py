"""Tests of the drawing of learning curves from the results of compare_configs."""

import json

import pytest

from scripts.plot_history import COLORS, kept_epoch, main as plot_main

HISTORY = {"loss": [0.9, 0.5, 0.4], "accuracy": [0.4, 0.7, 0.8],
           "val_loss": [0.95, 0.6, 0.7], "val_accuracy": [0.35, 0.65, 0.6]}


def write_results(path, results):
    path.write_text(json.dumps(results))
    return path


def test_curves_are_drawn_into_an_image(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    results = [{"config": "configs/a.conf", "history": HISTORY, "kept_epoch": 2},
               {"config": "configs/b.conf", "history": {**HISTORY, "loss": [1.0, 0.8, 0.2]}}]
    source = write_results(tmp_path / "results.json", results)
    output = tmp_path / "images" / "curves.png"
    plot_main(["--json", str(source), "--output", str(output), "--title", "Test"])
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert capsys.readouterr().out == f"{output}: 2 configurations\n"


def test_the_kept_epoch_is_the_last_one_without_early_stopping():
    assert kept_epoch({"history": HISTORY}) == 3
    assert kept_epoch({"history": HISTORY, "kept_epoch": 2}) == 2


@pytest.mark.parametrize("content, message", [
    (None, "No such file or directory"),
    ("not json", "Expecting value"),
    ([], "no configuration"),
    ([{"config": "a.conf", "history": HISTORY}] * (len(COLORS) + 1), "at most 8 configurations"),
    ([{"config": "a.conf"}], "training and validation loss and accuracy"),
    ([{"config": "a.conf", "history": {**HISTORY, "val_loss": []}}], "validation loss"),
    (["a.conf"], "training and validation loss and accuracy"),
])
def test_invalid_results_stop_with_a_message(tmp_path, content, message):
    source = tmp_path / "results.json"
    if content is not None:
        source.write_text(content if isinstance(content, str) else json.dumps(content))
    with pytest.raises(SystemExit) as error:
        plot_main(["--json", str(source), "--output", str(tmp_path / "curves.png")])
    assert message in str(error.value.code)
    assert not (tmp_path / "curves.png").exists()
