"""Tests of the optimizers."""

import numpy as np
import pytest

from my_torch import Dense, MyTorchError, Network, Optimizer, get_optimizer
from my_torch.optimizers import SGD, Adam, Momentum, RMSProp

ALL_OPTIMIZERS = [
    ("sgd", {}),
    ("momentum", {}),
    ("momentum", {"nesterov": True}),
    ("rmsprop", {}),
    ("adam", {}),
]
GRADIENTS = [2.0, -1.0, 3.0, 0.5, -2.0, 1.0]


def run_steps(optimizer, gradients):
    """Apply successive gradients to a weight starting at 0, return its positions."""
    layer = Dense(1, 1)
    positions = []
    for gradient in gradients:
        layer.weights_gradient = np.array([[float(gradient)]])
        layer.biases_gradient = np.array([0.0])
        optimizer.step([layer])
        positions.append(float(layer.weights[0, 0]))
    return positions


@pytest.mark.parametrize("name, params, gradients, expected", [
    ("sgd", {}, [1, 1, 1], [-0.1, -0.2, -0.3]),
    ("momentum", {"momentum": 0.9}, [1, 1, 1], [-0.1, -0.29, -0.561]),
    ("momentum", {"momentum": 0.9, "nesterov": True}, [1, 1, 1], [-0.19, -0.461, -0.8049]),
    ("rmsprop", {"decay": 0.9}, [2, -1], [-0.31623, -0.16879]),
    ("adam", {"beta1": 0.9, "beta2": 0.999}, [2, -1], [-0.1, -0.12663]),
])
def test_positions_computed_by_hand(name, params, gradients, expected):
    optimizer = get_optimizer(name, learning_rate=0.1, **params)
    assert run_steps(optimizer, gradients) == pytest.approx(expected, abs=1e-5)


def test_sgd_updates_weights_and_biases():
    layer = Dense(2, 1)
    layer.weights = np.array([[1.0], [2.0]])
    layer.biases = np.array([3.0])
    layer.weights_gradient = np.array([[0.5], [-1.0]])
    layer.biases_gradient = np.array([1.0])
    get_optimizer("sgd", learning_rate=0.1).step([layer])
    assert layer.weights == pytest.approx(np.array([[0.95], [2.1]]))
    assert layer.biases == pytest.approx(np.array([2.9]))


@pytest.mark.parametrize("name, params", ALL_OPTIMIZERS)
def test_parameters_keep_separate_memories(name, params):
    sequences = [[1.0, 2.0, -1.0], [-3.0, 0.5, 4.0], [0.1, 0.2, 0.3], [5.0, -5.0, 5.0]]
    layers = [Dense(1, 1), Dense(1, 1)]
    optimizer = get_optimizer(name, learning_rate=0.1, **params)
    for step in range(3):
        for index, layer in enumerate(layers):
            layer.weights_gradient = np.array([[sequences[2 * index][step]]])
            layer.biases_gradient = np.array([sequences[2 * index + 1][step]])
        optimizer.step(layers)
    positions = [
        layers[0].weights[0, 0],
        layers[0].biases[0],
        layers[1].weights[0, 0],
        layers[1].biases[0],
    ]
    expected = [
        run_steps(get_optimizer(name, learning_rate=0.1, **params), sequence)[-1]
        for sequence in sequences
    ]
    assert positions == pytest.approx(expected)


@pytest.mark.parametrize("nesterov", [False, True])
def test_momentum_at_zero_is_sgd(nesterov):
    optimizer = get_optimizer("momentum", learning_rate=0.1, momentum=0.0, nesterov=nesterov)
    assert run_steps(optimizer, GRADIENTS) == run_steps(
        get_optimizer("sgd", learning_rate=0.1), GRADIENTS
    )


def test_nesterov_follows_the_original_formulation():
    """The stored weight of the rewritten form is the look-ahead point of the original."""
    learning_rate, momentum, curvature = 0.1, 0.9, 3.0
    position, velocity = 1.0, 0.0
    layer = Dense(1, 1)
    layer.weights = np.array([[position]])
    optimizer = get_optimizer(
        "momentum", learning_rate=learning_rate, momentum=momentum, nesterov=True
    )
    for _ in range(50):
        look_ahead = position - learning_rate * momentum * velocity
        assert layer.weights[0, 0] == pytest.approx(look_ahead)
        velocity = momentum * velocity + curvature * look_ahead
        position -= learning_rate * velocity
        layer.weights_gradient = curvature * layer.weights
        layer.biases_gradient = np.array([0.0])
        optimizer.step([layer])


@pytest.mark.parametrize("gradient", [2.0, -500.0, 0.001])
def test_adam_first_step_has_the_size_of_the_learning_rate(gradient):
    position = run_steps(get_optimizer("adam", learning_rate=0.1), [gradient])[0]
    assert position == pytest.approx(-0.1 * np.sign(gradient), abs=1e-5)


def test_adam_counts_steps_and_not_parameters():
    layers = [Dense(1, 1), Dense(1, 1)]
    for layer in layers:
        layer.weights_gradient = np.array([[1.0]])
        layer.biases_gradient = np.array([1.0])
    optimizer = get_optimizer("adam")
    optimizer.step(layers)
    optimizer.step(layers)
    assert optimizer.steps == 2


@pytest.mark.parametrize("name", ["rmsprop", "adam"])
def test_adaptive_steps_do_not_depend_on_the_scale_of_gradients(name):
    scaled = [1000.0 * gradient for gradient in GRADIENTS]
    assert run_steps(get_optimizer(name, learning_rate=0.1), scaled) == pytest.approx(
        run_steps(get_optimizer(name, learning_rate=0.1), GRADIENTS), abs=1e-6
    )


@pytest.mark.parametrize("name, params", ALL_OPTIMIZERS)
def test_null_gradient_does_not_move_the_parameter(name, params):
    assert run_steps(get_optimizer(name, **params), [0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("name, params", ALL_OPTIMIZERS)
def test_step_does_not_change_the_gradients(name, params):
    layer = Dense(1, 1)
    layer.weights_gradient = np.array([[1.5]])
    layer.biases_gradient = np.array([-2.5])
    get_optimizer(name, **params).step([layer])
    assert layer.weights_gradient.tolist() == [[1.5]]
    assert layer.biases_gradient.tolist() == [-2.5]


@pytest.mark.parametrize("name, params", ALL_OPTIMIZERS)
def test_step_before_backward_raises_an_error(name, params):
    network = Network(2, [(2, "identity")], seed=0)
    with pytest.raises(MyTorchError):
        get_optimizer(name, **params).step(network.layers)


@pytest.mark.parametrize("name, parameter, value, message", [
    ("sgd", "learning_rate", 0, "learning rate"),
    ("sgd", "learning_rate", -0.1, "learning rate"),
    ("sgd", "learning_rate", float("nan"), "learning rate"),
    ("momentum", "learning_rate", 0, "learning rate"),
    ("rmsprop", "learning_rate", 0, "learning rate"),
    ("adam", "learning_rate", 0, "learning rate"),
    ("momentum", "momentum", 1, "momentum"),
    ("momentum", "momentum", -0.1, "momentum"),
    ("momentum", "momentum", float("nan"), "momentum"),
    ("rmsprop", "decay", 1, "decay"),
    ("rmsprop", "decay", -0.1, "decay"),
    ("rmsprop", "epsilon", 0, "epsilon"),
    ("rmsprop", "epsilon", float("nan"), "epsilon"),
    ("adam", "beta1", 1, "beta1"),
    ("adam", "beta1", -0.1, "beta1"),
    ("adam", "beta2", 1, "beta2"),
    ("adam", "beta2", float("nan"), "beta2"),
    ("adam", "epsilon", 0, "epsilon"),
])
def test_invalid_values_raise_an_error(name, parameter, value, message):
    with pytest.raises(MyTorchError, match=message):
        get_optimizer(name, **{parameter: value})


@pytest.mark.parametrize("name, parameter", [
    ("momentum", "momentum"),
    ("rmsprop", "decay"),
    ("adam", "beta1"),
    ("adam", "beta2"),
])
def test_zero_is_a_valid_decay(name, parameter):
    assert getattr(get_optimizer(name, **{parameter: 0.0}), parameter) == 0.0


@pytest.mark.parametrize("name, expected", [
    ("sgd", SGD),
    ("momentum", Momentum),
    ("rmsprop", RMSProp),
    ("adam", Adam),
])
def test_get_optimizer_returns_the_right_class(name, expected):
    optimizer = get_optimizer(name)
    assert type(optimizer) is expected
    assert isinstance(optimizer, Optimizer)


def test_get_optimizer_forwards_the_parameters():
    momentum = get_optimizer("momentum", learning_rate=0.05, momentum=0.8, nesterov=True)
    assert (momentum.learning_rate, momentum.momentum, momentum.nesterov) == (0.05, 0.8, True)
    rmsprop = get_optimizer("rmsprop", learning_rate=0.2, decay=0.5, epsilon=1e-3)
    assert (rmsprop.learning_rate, rmsprop.decay, rmsprop.epsilon) == (0.2, 0.5, 1e-3)
    adam = get_optimizer("adam", learning_rate=0.3, beta1=0.5, beta2=0.6, epsilon=1e-4)
    assert (adam.learning_rate, adam.beta1, adam.beta2, adam.epsilon) == (0.3, 0.5, 0.6, 1e-4)


def test_default_values():
    assert get_optimizer("sgd").learning_rate == 0.01
    momentum = get_optimizer("momentum")
    assert (momentum.learning_rate, momentum.momentum, momentum.nesterov) == (0.01, 0.9, False)
    rmsprop = get_optimizer("rmsprop")
    assert (rmsprop.learning_rate, rmsprop.decay, rmsprop.epsilon) == (0.001, 0.9, 1e-8)
    adam = get_optimizer("adam")
    assert (adam.learning_rate, adam.beta1, adam.beta2, adam.epsilon) == (0.001, 0.9, 0.999, 1e-8)


def test_get_optimizer_returns_a_new_object():
    first, second = get_optimizer("momentum"), get_optimizer("momentum")
    assert first is not second
    run_steps(first, [1.0])
    assert first.velocities
    assert not second.velocities


@pytest.mark.parametrize("name", ["unknown", "SGD", ""])
def test_unknown_optimizer_raises_an_error(name):
    with pytest.raises(MyTorchError, match="unknown optimizer"):
        get_optimizer(name)


@pytest.mark.parametrize("name, parameter", [
    ("sgd", "momentum"),
    ("momentum", "beta1"),
    ("rmsprop", "beta1"),
    ("adam", "decay"),
])
def test_unknown_parameter_raises_an_error(name, parameter):
    with pytest.raises(MyTorchError, match=parameter):
        get_optimizer(name, **{parameter: 0.5})


@pytest.mark.parametrize("name, parameter, value", [
    ("sgd", "learning_rate", "abc"),
    ("sgd", "learning_rate", None),
    ("momentum", "momentum", "abc"),
    ("rmsprop", "epsilon", "abc"),
    ("adam", "beta1", "abc"),
])
def test_non_numeric_value_raises_an_error(name, parameter, value):
    with pytest.raises(MyTorchError):
        get_optimizer(name, **{parameter: value})


def test_base_class_does_not_compute_updates():
    with pytest.raises(NotImplementedError):
        Optimizer().compute_update((0, "weights"), np.zeros((1, 1)))
