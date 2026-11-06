"""Tests of the ability of a small network to learn."""

import numpy as np
import pytest

from my_torch import Network, get_loss

XOR_INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_LABELS = np.array([0, 1, 1, 0])
XOR_TARGETS = np.eye(2)[XOR_LABELS]


def train_on_xor(network, learning_rate=0.5, epochs=2000):
    """Run a plain gradient descent, then return the final loss."""
    loss = get_loss("cross_entropy")
    for _ in range(epochs):
        network.backward(loss.backward(network.forward(XOR_INPUTS), XOR_TARGETS))
        for layer in network.layers:
            layer.weights -= learning_rate * layer.weights_gradient
            layer.biases -= learning_rate * layer.biases_gradient
    return loss.forward(network.forward(XOR_INPUTS), XOR_TARGETS)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("architecture, initializer", [
    ([(4, "tanh"), (2, "identity")], "xavier_normal"),
    ([(8, "relu"), (2, "identity")], "he"),
])
def test_network_learns_xor(architecture, initializer, seed):
    network = Network(2, architecture, initializer, seed=seed)
    assert train_on_xor(network) < 0.01
    assert network.predict(XOR_INPUTS).tolist() == XOR_LABELS.tolist()


def test_single_linear_layer_cannot_learn_xor():
    network = Network(2, [(2, "identity")], seed=0)
    train_on_xor(network)
    assert network.predict(XOR_INPUTS).tolist() != XOR_LABELS.tolist()
