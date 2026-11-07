"""Tests of the ability of a small network to learn."""

import numpy as np
import pytest

from my_torch import Network, get_loss, get_optimizer, train

XOR_INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_LABELS = np.array([0, 1, 1, 0])
XOR_TARGETS = np.eye(2)[XOR_LABELS]

REGRESSION_INPUTS = np.array(
    [[a, b] for a in np.linspace(-1.0, 1.0, 5) for b in np.linspace(-1.0, 1.0, 5)]
)
REGRESSION_WEIGHTS = np.array([[2.0], [-3.0]])
REGRESSION_BIAS = 1.0
REGRESSION_TARGETS = REGRESSION_INPUTS @ REGRESSION_WEIGHTS + REGRESSION_BIAS

SEEDS = [0, 1, 2]


def train_full_batch(network, optimizer, loss, inputs, targets, epochs=2000):
    """Train a network on the whole dataset in a fixed order, then return the final loss."""
    history = train(network, loss, optimizer, inputs, targets, epochs, len(inputs), shuffle=False)
    return history["loss"][-1]


def train_on_xor(network, optimizer):
    return train_full_batch(
        network, optimizer, get_loss("cross_entropy"), XOR_INPUTS, XOR_TARGETS
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("architecture, initializer", [
    ([(4, "tanh"), (2, "identity")], "xavier_normal"),
    ([(8, "relu"), (2, "identity")], "he"),
])
def test_network_learns_xor(architecture, initializer, seed):
    network = Network(2, architecture, initializer, seed=seed)
    assert train_on_xor(network, get_optimizer("sgd", learning_rate=0.5)) < 0.01
    assert network.predict(XOR_INPUTS).tolist() == XOR_LABELS.tolist()


def test_single_linear_layer_cannot_learn_xor():
    network = Network(2, [(2, "identity")], seed=0)
    train_on_xor(network, get_optimizer("sgd", learning_rate=0.5))
    assert network.predict(XOR_INPUTS).tolist() != XOR_LABELS.tolist()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name, params", [
    ("sgd", {"learning_rate": 0.5}),
    ("momentum", {"learning_rate": 0.05}),
    ("momentum", {"learning_rate": 0.05, "nesterov": True}),
    ("rmsprop", {"learning_rate": 0.1}),
    ("adam", {"learning_rate": 0.01}),
])
def test_optimizer_learns_xor(name, params, seed):
    network = Network(2, [(4, "tanh"), (2, "identity")], "xavier_normal", seed=seed)
    assert train_on_xor(network, get_optimizer(name, **params)) < 0.01
    assert network.predict(XOR_INPUTS).tolist() == XOR_LABELS.tolist()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name, params", [
    ("sgd", {"learning_rate": 0.1}),
    ("momentum", {"learning_rate": 0.05}),
    ("momentum", {"learning_rate": 0.05, "nesterov": True}),
    ("rmsprop", {"learning_rate": 0.01}),
    ("adam", {"learning_rate": 0.01}),
])
def test_optimizer_finds_the_solution_of_a_linear_regression(name, params, seed):
    network = Network(2, [(1, "identity")], seed=seed)
    train_full_batch(
        network, get_optimizer(name, **params), get_loss("mse"),
        REGRESSION_INPUTS, REGRESSION_TARGETS,
    )
    layer = network.layers[0]
    assert layer.weights == pytest.approx(REGRESSION_WEIGHTS, abs=0.01)
    assert layer.biases == pytest.approx(np.array([REGRESSION_BIAS]), abs=0.01)
