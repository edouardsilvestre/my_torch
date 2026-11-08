"""Tests of the measures of NumPy speed and of the signal through deep networks."""

import numpy as np
import pytest

from scripts.numpy_speed import main as speed_main, measure, naive_matmul
from scripts.signal_by_initializer import main as signal_main, variances


def test_the_naive_product_is_a_matrix_product():
    left, right = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]]
    assert naive_matmul(left, right) == (np.array(left) @ np.array(right)).tolist()


def test_numpy_is_faster_than_the_naive_product(capsys):
    result = measure(8, 50, 20, repeats=1)
    assert result["operations"] == 2 * 8 * 50 * 20
    assert result["speedup"] == result["naive"] / result["numpy"] > 1
    speed_main(["--batch", "4", "--inputs", "30", "--neurons", "10", "--repeats", "1"])
    output = capsys.readouterr().out
    assert output.startswith("(4 x 30) @ (30 x 10): 2,400 operations\n")
    assert "times faster" in output


def test_he_keeps_the_signal_of_a_deep_relu_network():
    he = variances("he", 10, 200, 300, seed=0)
    uniform = variances("uniform", 10, 200, 300, seed=0)
    lecun = variances("lecun", 10, 200, 300, seed=0)
    assert len(he) == 10
    assert 0.5 < he[-1] / he[0] < 2
    assert lecun[-1] < lecun[0] / 100
    assert uniform[-1] < lecun[-1]


def test_the_measure_is_reproducible():
    assert variances("normal", 3, 50, 40, seed=1) == variances("normal", 3, 50, 40, seed=1)


def test_the_signal_table_and_image(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    signal_main(["--layers", "4", "--width", "30", "--examples", "20",
                 "--output", str(tmp_path / "signal.png")])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "| Initializer | Layer 1 | Layer 2 | Layer 4 |"
    assert len(lines) == 2 + 6 and lines[-1].startswith("| he | ")
    assert (tmp_path / "signal.png").read_bytes().startswith(b"\x89PNG")
