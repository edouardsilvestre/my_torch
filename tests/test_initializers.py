"""Tests of the initialization strategies."""

import numpy as np
import pytest

from my_torch import MyTorchError, Network, get_initializer

SHAPE = (2000, 500)
EXPECTED_VARIANCES = {
    "uniform": 0.05 ** 2 / 3,
    "normal": 0.05 ** 2,
    "lecun": 1 / 2000,
    "xavier_uniform": 2 / 2500,
    "xavier_normal": 2 / 2500,
    "he": 2 / 2000,
}
ARCHITECTURE = [(500, "relu"), (500, "relu"), (3, "identity")]


@pytest.mark.parametrize("name, variance", EXPECTED_VARIANCES.items())
def test_weights_have_the_expected_statistics(name, variance):
    weights = get_initializer(name)(SHAPE, np.random.default_rng(0))
    assert weights.shape == SHAPE
    assert weights.mean() == pytest.approx(0.0, abs=0.001)
    assert weights.var() == pytest.approx(variance, rel=0.02)


@pytest.mark.parametrize("name, limit", [("uniform", 0.05), ("xavier_uniform", np.sqrt(6 / 2500))])
def test_uniform_strategies_stay_within_their_limit(name, limit):
    weights = get_initializer(name)(SHAPE, np.random.default_rng(0))
    assert np.abs(weights).max() <= limit


@pytest.mark.parametrize("name", EXPECTED_VARIANCES)
def test_same_seed_gives_the_same_weights(name):
    initialize = get_initializer(name)
    first = initialize((3, 2), np.random.default_rng(42))
    second = initialize((3, 2), np.random.default_rng(42))
    assert np.array_equal(first, second)


@pytest.mark.parametrize("name", ["HE", "orthogonal"])
def test_unknown_initializer_raises(name):
    with pytest.raises(MyTorchError, match="unknown initializer"):
        get_initializer(name)


def test_network_uses_he_by_default():
    network = Network(2000, ARCHITECTURE, seed=0)
    assert network.layers[0].weights.var() == pytest.approx(2 / 2000, rel=0.02)


def test_network_uses_the_requested_initializer():
    network = Network(2000, ARCHITECTURE, "xavier_normal", seed=0)
    assert network.layers[0].weights.var() == pytest.approx(2 / 2500, rel=0.02)


def test_network_rejects_an_unknown_initializer():
    with pytest.raises(MyTorchError, match="unknown initializer"):
        Network(2000, ARCHITECTURE, "orthogonal")


def test_networks_with_the_same_seed_have_the_same_weights():
    first = Network(10, ARCHITECTURE, seed=42)
    second = Network(10, ARCHITECTURE, seed=42)
    for first_layer, second_layer in zip(first.layers, second.layers):
        assert np.array_equal(first_layer.weights, second_layer.weights)


def test_networks_without_seed_have_different_weights():
    first = Network(10, ARCHITECTURE)
    second = Network(10, ARCHITECTURE)
    assert not np.array_equal(first.layers[0].weights, second.layers[0].weights)


def test_layers_of_the_same_size_have_different_weights():
    network = Network(500, ARCHITECTURE, seed=0)
    assert not np.array_equal(network.layers[0].weights, network.layers[1].weights)


def test_biases_start_at_zero():
    network = Network(10, ARCHITECTURE, seed=0)
    assert not any(layer.biases.any() for layer in network.layers)
