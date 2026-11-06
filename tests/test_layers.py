"""Tests of the Dense layer."""

import numpy as np
import pytest

from my_torch import Dense, MyTorchError

INPUTS = np.array([[1.0, 2.0], [3.0, 4.0]])
EXPECTED_OUTPUTS = np.array([[2.0, 3.0, 1.0], [4.0, 5.0, 3.0]])


@pytest.fixture
def layer():
    """Layer of 2 inputs and 3 neurons with hand-picked parameters."""
    layer = Dense(2, 3)
    layer.weights = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
    layer.biases = np.array([1.0, 1.0, 1.0])
    return layer


def test_parameters_have_expected_shapes():
    layer = Dense(768, 256)
    assert layer.weights.shape == (768, 256)
    assert layer.biases.shape == (256,)


@pytest.mark.parametrize("sizes", [(0, 3), (3, 0), (-1, 3)])
def test_invalid_sizes_raise(sizes):
    with pytest.raises(MyTorchError):
        Dense(*sizes)


def test_forward_matches_hand_computed_values(layer):
    assert np.allclose(layer.forward(INPUTS), EXPECTED_OUTPUTS)


def test_batch_gives_same_result_as_each_example_alone(layer):
    batch_outputs = layer.forward(INPUTS)
    for i, example in enumerate(INPUTS):
        assert np.allclose(layer.forward(example[np.newaxis]), batch_outputs[i])


@pytest.mark.parametrize("batch_size", [1, 32])
def test_forward_output_shape(batch_size):
    outputs = Dense(768, 256).forward(np.zeros((batch_size, 768)))
    assert outputs.shape == (batch_size, 256)


def test_forward_remembers_inputs_and_outputs(layer):
    outputs = layer.forward(INPUTS)
    assert np.array_equal(layer.inputs, INPUTS)
    assert np.array_equal(layer.outputs, outputs)


@pytest.mark.parametrize("shape", [(2,), (2, 3), (1, 2, 2)])
def test_forward_rejects_wrong_shape(layer, shape):
    with pytest.raises(MyTorchError):
        layer.forward(np.zeros(shape))


def test_backward_matches_hand_computed_values(layer):
    layer.forward(INPUTS)
    input_gradient = layer.backward(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert np.allclose(layer.weights_gradient, [[1.0, 3.0, 0.0], [2.0, 4.0, 0.0]])
    assert np.allclose(layer.biases_gradient, [1.0, 1.0, 0.0])
    assert np.allclose(input_gradient, [[1.0, 0.0], [0.0, 1.0]])


def test_backward_output_shapes():
    layer = Dense(768, 256)
    layer.forward(np.zeros((5, 768)))
    input_gradient = layer.backward(np.zeros((5, 256)))
    assert layer.weights_gradient.shape == (768, 256)
    assert layer.biases_gradient.shape == (256,)
    assert input_gradient.shape == (5, 768)


def test_backward_before_forward_raises():
    with pytest.raises(MyTorchError, match="forward must be called"):
        Dense(2, 3).backward(np.zeros((1, 3)))


def test_backward_rejects_wrong_gradient_shape(layer):
    layer.forward(INPUTS)
    with pytest.raises(MyTorchError, match="expected a gradient"):
        layer.backward(np.zeros((2, 2)))
