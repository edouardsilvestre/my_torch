"""Tests of the activation functions."""

import numpy as np
import pytest

from my_torch import MyTorchError, get_activation

ALL_ACTIVATIONS = ["identity", "sigmoid", "tanh", "relu", "leaky_relu", "softmax"]


def numerical_backward(activation, z, gradient, step=1e-6):
    """Estimate backward with central finite differences."""
    estimate = np.zeros_like(z)
    for index in np.ndindex(z.shape):
        shift = np.zeros_like(z)
        shift[index] = step
        upper = np.sum(activation.forward(z + shift) * gradient)
        lower = np.sum(activation.forward(z - shift) * gradient)
        estimate[index] = (upper - lower) / (2 * step)
    return estimate


@pytest.mark.parametrize("name, z, expected", [
    ("identity", [[-2.0, 0.0, 3.0]], [[-2.0, 0.0, 3.0]]),
    ("sigmoid", [[0.0, 2.0]], [[0.5, 0.88079708]]),
    ("tanh", [[0.0, 1.0]], [[0.0, 0.76159416]]),
    ("relu", [[-2.0, 0.0, 3.0]], [[0.0, 0.0, 3.0]]),
    ("leaky_relu", [[-2.0, 0.0, 3.0]], [[-0.02, 0.0, 3.0]]),
    ("softmax", [[0.0, 1.0, 2.0]], [[0.09003057, 0.24472847, 0.66524096]]),
])
def test_forward_matches_known_values(name, z, expected):
    assert np.allclose(get_activation(name).forward(np.array(z)), expected)


@pytest.mark.parametrize("name", ALL_ACTIVATIONS)
def test_backward_matches_finite_differences(name):
    rng = np.random.default_rng(0)
    z = rng.normal(size=(4, 3))
    gradient = rng.normal(size=(4, 3))
    activation = get_activation(name)
    backward = activation.backward(z, gradient)
    assert backward.shape == z.shape
    assert np.allclose(backward, numerical_backward(activation, z, gradient))


@pytest.mark.parametrize("name, slope", [("relu", 0.0), ("leaky_relu", 0.01)])
def test_derivative_at_zero_uses_the_left_slope(name, slope):
    assert get_activation(name).derivative(np.array([[0.0]])).item() == slope


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("name, z, expected", [
    ("sigmoid", [[-1000.0, 1000.0]], [[0.0, 1.0]]),
    ("softmax", [[1000.0, 1000.0, 0.0]], [[0.5, 0.5, 0.0]]),
])
def test_extreme_values_stay_finite_without_warning(name, z, expected):
    assert np.allclose(get_activation(name).forward(np.array(z)), expected)


def test_softmax_outputs_sum_to_one_for_each_example():
    z = np.array([[0.0, 1.0, 2.0], [5.0, 5.0, 5.0], [-3.0, 10.0, 0.0]])
    assert np.allclose(get_activation("softmax").forward(z).sum(axis=1), 1.0)


def test_softmax_computes_each_example_independently():
    z = np.array([[0.0, 1.0, 2.0], [-3.0, 10.0, 0.0]])
    softmax = get_activation("softmax")
    assert np.allclose(softmax.forward(z)[0], softmax.forward(z[:1])[0])


@pytest.mark.parametrize("name", ["Relu", "swish"])
def test_unknown_activation_raises(name):
    with pytest.raises(MyTorchError, match="unknown activation"):
        get_activation(name)
