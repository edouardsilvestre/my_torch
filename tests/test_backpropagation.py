"""Tests of the backpropagation, with gradient checking on small networks."""

import numpy as np
import pytest

from my_torch import MyTorchError, Network, get_loss

STEP = 1e-6


def numerical_gradient(parameter, compute_loss):
    """Estimate the gradient of a parameter with central finite differences."""
    estimate = np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        saved = parameter[index]
        parameter[index] = saved + STEP
        upper = compute_loss()
        parameter[index] = saved - STEP
        lower = compute_loss()
        parameter[index] = saved
        estimate[index] = (upper - lower) / (2 * STEP)
    return estimate


@pytest.mark.parametrize("hidden", ["identity", "sigmoid", "tanh", "relu", "leaky_relu", "softmax"])
@pytest.mark.parametrize("last, loss_name", [
    ("identity", "cross_entropy"),
    ("identity", "mse"),
    ("softmax", "mse"),
    ("sigmoid", "mse"),
])
def test_gradients_match_finite_differences(hidden, last, loss_name):
    rng = np.random.default_rng(0)
    network = Network(4, [(5, hidden), (4, hidden), (3, last)], seed=1)
    inputs = rng.normal(size=(6, 4))
    targets = np.eye(3)[rng.integers(0, 3, size=6)]
    loss = get_loss(loss_name)

    def compute_loss():
        return loss.forward(network.forward(inputs), targets)

    network.backward(loss.backward(network.forward(inputs), targets))
    for layer in network.layers:
        assert np.allclose(layer.weights_gradient, numerical_gradient(layer.weights, compute_loss), atol=1e-6)
        assert np.allclose(layer.biases_gradient, numerical_gradient(layer.biases, compute_loss), atol=1e-6)


def hand_picked_network():
    network = Network(2, [(2, "relu"), (1, "identity")])
    network.layers[0].weights = np.array([[1.0, -1.0], [0.0, 2.0]])
    network.layers[0].biases = np.array([0.0, -1.0])
    network.layers[1].weights = np.array([[2.0], [3.0]])
    network.layers[1].biases = np.array([1.0])
    return network


def test_backward_matches_hand_computed_values():
    network = hand_picked_network()
    network.forward(np.array([[1.0, 2.0], [3.0, 0.0]]))
    network.backward(np.array([[1.0], [-1.0]]))
    assert np.allclose(network.layers[1].weights_gradient, [[-2.0], [2.0]])
    assert np.allclose(network.layers[1].biases_gradient, [0.0])
    assert np.allclose(network.layers[0].weights_gradient, [[-4.0, 3.0], [4.0, 6.0]])
    assert np.allclose(network.layers[0].biases_gradient, [0.0, 3.0])


@pytest.mark.parametrize("loss_name, last", [("mse", "identity"), ("cross_entropy", "identity")])
def test_gradient_does_not_depend_on_the_batch_size(loss_name, last):
    network = Network(2, [(3, "relu"), (2, last)], seed=0)
    example = np.array([[1.0, 2.0]])
    target = np.array([[1.0, 0.0]])
    loss = get_loss(loss_name)
    network.backward(loss.backward(network.forward(example), target))
    single = network.layers[0].weights_gradient
    doubled_example, doubled_target = np.vstack([example, example]), np.vstack([target, target])
    network.backward(loss.backward(network.forward(doubled_example), doubled_target))
    assert np.allclose(network.layers[0].weights_gradient, single)


def test_backward_does_not_change_the_parameters():
    network = hand_picked_network()
    network.forward(np.array([[1.0, 2.0]]))
    network.backward(np.array([[1.0]]))
    assert np.array_equal(network.layers[0].weights, [[1.0, -1.0], [0.0, 2.0]])
    assert np.array_equal(network.layers[1].biases, [1.0])


def test_network_backward_before_forward_raises():
    with pytest.raises(MyTorchError, match="forward must be called"):
        hand_picked_network().backward(np.array([[1.0]]))


def test_network_backward_rejects_wrong_gradient_shape():
    network = Network(2, [(3, "identity")], seed=0)
    network.forward(np.zeros((2, 2)))
    with pytest.raises(MyTorchError, match="expected a gradient"):
        network.backward(np.zeros((2, 1)))
