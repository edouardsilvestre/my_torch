"""Tests of the loss functions."""

import numpy as np
import pytest

from my_torch import MyTorchError, get_loss

ALL_LOSSES = ["mse", "cross_entropy"]


def numerical_backward(loss, outputs, targets, step=1e-6):
    """Estimate backward with central finite differences."""
    estimate = np.zeros_like(outputs)
    for index in np.ndindex(outputs.shape):
        shift = np.zeros_like(outputs)
        shift[index] = step
        upper = loss.forward(outputs + shift, targets)
        lower = loss.forward(outputs - shift, targets)
        estimate[index] = (upper - lower) / (2 * step)
    return estimate


@pytest.mark.parametrize("name, outputs, targets, expected", [
    ("mse", [[1.0, 2.0], [3.0, 4.0]], [[1.0, 0.0], [3.0, 0.0]], 5.0),
    ("cross_entropy", [[0.0, 1.0, 2.0]], [[0.0, 0.0, 1.0]], 0.40760596),
    ("cross_entropy", [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], 1.40760596),
])
def test_forward_matches_known_values(name, outputs, targets, expected):
    loss = get_loss(name).forward(np.array(outputs), np.array(targets))
    assert loss == pytest.approx(expected)


@pytest.mark.parametrize("probability", [0.90, 0.50, 0.01])
def test_cross_entropy_is_the_negative_log_of_the_right_probability(probability):
    other = (1.0 - probability) / 2
    scores = np.log(np.array([[probability, other, other]]))
    targets = np.array([[1.0, 0.0, 0.0]])
    assert get_loss("cross_entropy").forward(scores, targets) == pytest.approx(-np.log(probability))


@pytest.mark.parametrize("name", ALL_LOSSES)
def test_backward_matches_finite_differences(name):
    rng = np.random.default_rng(0)
    outputs = rng.normal(size=(4, 3))
    targets = np.eye(3)[rng.integers(0, 3, size=4)]
    loss = get_loss(name)
    backward = loss.backward(outputs, targets)
    assert backward.shape == outputs.shape
    assert np.allclose(backward, numerical_backward(loss, outputs, targets))


def test_mse_is_zero_when_outputs_equal_targets():
    targets = np.array([[1.0, 0.0], [0.0, 1.0]])
    mse = get_loss("mse")
    assert mse.forward(targets, targets) == 0.0
    assert not mse.backward(targets, targets).any()


@pytest.mark.filterwarnings("error")
def test_cross_entropy_stays_exact_with_extreme_scores():
    scores = np.array([[-1000.0, 1000.0], [1000.0, -1000.0]])
    targets = np.array([[1.0, 0.0], [1.0, 0.0]])
    loss = get_loss("cross_entropy")
    assert loss.forward(scores, targets) == pytest.approx(1000.0)
    assert np.allclose(loss.backward(scores, targets), [[-0.5, 0.5], [0.0, 0.0]])


@pytest.mark.parametrize("name", ALL_LOSSES)
@pytest.mark.parametrize("method", ["forward", "backward"])
def test_different_shapes_raise(name, method):
    with pytest.raises(MyTorchError, match="same shape"):
        getattr(get_loss(name), method)(np.zeros((32, 3)), np.zeros((32, 1)))


@pytest.mark.parametrize("name", ["MSE", "hinge"])
def test_unknown_loss_raises(name):
    with pytest.raises(MyTorchError, match="unknown loss"):
        get_loss(name)
