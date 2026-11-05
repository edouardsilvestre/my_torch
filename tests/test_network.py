"""Tests of the Network class."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from my_torch import MyTorchError, Network

ROOT = Path(__file__).resolve().parent.parent
INPUTS = np.array([[1.0, 2.0], [3.0, 0.0]])


@pytest.fixture
def network():
    """Network of 2 layers with hand-picked parameters."""
    network = Network(2, [(2, "relu"), (1, "identity")])
    network.layers[0].weights = np.array([[1.0, -1.0], [0.0, 2.0]])
    network.layers[0].biases = np.array([0.0, -1.0])
    network.layers[1].weights = np.array([[2.0], [3.0]])
    network.layers[1].biases = np.array([1.0])
    return network


def test_architecture_chains_layer_sizes():
    architecture = [(256, "relu"), (64, "relu"), (3, "identity")]
    network = Network(768, architecture)
    assert [layer.weights.shape for layer in network.layers] == [(768, 256), (256, 64), (64, 3)]
    assert [type(activation).__name__ for activation in network.activations] == ["ReLU", "ReLU", "Identity"]
    assert network.architecture == architecture


def test_architecture_is_copied():
    architecture = [(2, "relu")]
    network = Network(4, architecture)
    architecture.append((3, "tanh"))
    assert network.architecture == [(2, "relu")]


@pytest.mark.parametrize("input_size, architecture", [
    (4, []),
    (4, [(0, "relu")]),
    (0, [(2, "relu")]),
    (4, [(2, "swish")]),
])
def test_invalid_architecture_raises(input_size, architecture):
    with pytest.raises(MyTorchError):
        Network(input_size, architecture)


def test_forward_matches_hand_computed_values(network):
    assert np.allclose(network.forward(INPUTS), [[9.0], [7.0]])


def test_forward_keeps_values_before_and_after_the_activation(network):
    network.forward(INPUTS)
    assert np.allclose(network.layers[0].outputs, [[1.0, 2.0], [3.0, -4.0]])
    assert np.allclose(network.layers[1].inputs, [[1.0, 2.0], [3.0, 0.0]])


def test_forward_output_shape():
    network = Network(768, [(256, "relu"), (64, "relu"), (3, "identity")])
    assert network.forward(np.zeros((5, 768))).shape == (5, 3)


def test_forward_rejects_wrong_number_of_inputs(network):
    with pytest.raises(MyTorchError):
        network.forward(np.zeros((2, 3)))


@pytest.mark.parametrize("activation", ["identity", "softmax"])
def test_predict_returns_the_index_of_the_highest_score(activation):
    network = Network(3, [(3, activation)])
    network.layers[0].weights = np.eye(3)
    scores = np.array([[0.1, 2.0, -1.0], [3.0, 0.5, 0.5], [4.0, 0.0, 1.0]])
    assert network.predict(scores).tolist() == [1, 0, 0]


def test_predict_returns_one_class_per_example():
    network = Network(4, [(3, "identity")])
    assert network.predict(np.zeros((5, 4))).shape == (5,)


def test_module_does_not_depend_on_the_analyzer():
    code = "import sys, my_torch; sys.exit('analyzer' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0
