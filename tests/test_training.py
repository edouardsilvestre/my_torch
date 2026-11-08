"""Tests of the training loop."""

import json
import re

import numpy as np
import pytest

from my_torch import MyTorchError, Network, Optimizer, get_loss, get_optimizer, train
from my_torch.training import evaluate, is_one_hot

XOR_INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_LABELS = np.array([0, 1, 1, 0])
XOR_TARGETS = np.eye(2)[XOR_LABELS]

# Each input is its own index, so that a batch tells which examples it holds.
INDEX_INPUTS = np.arange(10, dtype=float).reshape(10, 1)
INDEX_TARGETS = 10.0 * INDEX_INPUTS


class RecordingOptimizer(Optimizer):
    """Record the examples of each batch, without changing any parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.batches = []

    def step(self, layers) -> None:
        self.batches.append(layers[0].inputs[:, 0].astype(int).tolist())


def record_batches(epochs, batch_size, **options):
    optimizer = RecordingOptimizer()
    train(
        Network(1, [(1, "identity")], seed=0), get_loss("mse"), optimizer,
        INDEX_INPUTS, INDEX_TARGETS, epochs, batch_size, **options,
    )
    return optimizer.batches


def xor_network(seed=0):
    return Network(2, [(4, "tanh"), (2, "identity")], "xavier_normal", seed=seed)


@pytest.mark.parametrize("batch_size, sizes", [
    (3, [3, 3, 3, 1]),
    (1, [1] * 10),
    (10, [10]),
    (15, [10]),
])
def test_batches_have_the_requested_size(batch_size, sizes):
    batches = record_batches(2, batch_size, seed=0)
    assert [len(batch) for batch in batches] == sizes * 2


@pytest.mark.parametrize("batch_size", [1, 3, 4, 10])
def test_each_example_is_seen_once_per_epoch(batch_size):
    batches = record_batches(3, batch_size, seed=0)
    per_epoch = len(batches) // 3
    for epoch in range(3):
        examples = sum(batches[epoch * per_epoch:(epoch + 1) * per_epoch], [])
        assert sorted(examples) == list(range(10))


def test_without_shuffle_the_order_is_kept():
    assert record_batches(2, 4, shuffle=False) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]] * 2


def test_shuffle_changes_the_order_at_each_epoch():
    orders = record_batches(3, 10, seed=0)
    assert orders[0] != list(range(10))
    assert orders[0] != orders[1] and orders[1] != orders[2] and orders[0] != orders[2]


def test_same_seed_gives_the_same_order():
    assert record_batches(3, 4, seed=1) == record_batches(3, 4, seed=1)
    assert record_batches(3, 4, seed=1) != record_batches(3, 4, seed=2)


def test_targets_stay_paired_with_their_inputs():
    pairs = []

    class RecordingLoss(type(get_loss("mse"))):
        def backward(self, outputs, targets):
            pairs.append((self.inputs[:, 0].tolist(), targets[:, 0].tolist()))
            return super().backward(outputs, targets)

    loss = RecordingLoss()
    network = Network(1, [(1, "identity")], seed=0)
    original_forward = network.forward

    def forward(inputs):
        loss.inputs = inputs
        return original_forward(inputs)

    network.forward = forward
    train(network, loss, get_optimizer("sgd"), INDEX_INPUTS, INDEX_TARGETS, 2, 3, seed=0)
    assert len(pairs) == 8
    for inputs, targets in pairs:
        assert targets == [10.0 * value for value in inputs]


def test_training_does_not_modify_the_dataset():
    inputs, targets = INDEX_INPUTS.copy(), INDEX_TARGETS.copy()
    train(
        Network(1, [(1, "identity")], seed=0), get_loss("mse"), get_optimizer("sgd"),
        inputs, targets, 3, 3, seed=0,
    )
    assert np.array_equal(inputs, INDEX_INPUTS) and np.array_equal(targets, INDEX_TARGETS)


def test_full_batch_without_shuffle_is_a_plain_gradient_descent():
    expected, trained = xor_network(3), xor_network(3)
    loss, optimizer = get_loss("cross_entropy"), get_optimizer("sgd", learning_rate=0.5)
    for _ in range(100):
        expected.backward(loss.backward(expected.forward(XOR_INPUTS), XOR_TARGETS))
        optimizer.step(expected.layers)
    train(
        trained, loss, get_optimizer("sgd", learning_rate=0.5),
        XOR_INPUTS, XOR_TARGETS, 100, 4, shuffle=False,
    )
    for expected_layer, trained_layer in zip(expected.layers, trained.layers):
        assert np.array_equal(expected_layer.weights, trained_layer.weights)
        assert np.array_equal(expected_layer.biases, trained_layer.biases)


def test_history_holds_the_metrics_at_the_end_of_each_epoch():
    loss = get_loss("cross_entropy")
    history = train(
        xor_network(), loss, get_optimizer("sgd", learning_rate=0.5),
        XOR_INPUTS, XOR_TARGETS, 1, 4, shuffle=False,
    )
    reference = xor_network()
    optimizer = get_optimizer("sgd", learning_rate=0.5)
    reference.backward(loss.backward(reference.forward(XOR_INPUTS), XOR_TARGETS))
    optimizer.step(reference.layers)
    outputs = reference.forward(XOR_INPUTS)
    assert history["loss"] == [loss.forward(outputs, XOR_TARGETS)]
    assert history["accuracy"] == [np.mean(np.argmax(outputs, axis=1) == XOR_LABELS)]


def test_history_has_one_float_per_epoch():
    history = train(
        xor_network(), get_loss("cross_entropy"), get_optimizer("adam", learning_rate=0.01),
        XOR_INPUTS, XOR_TARGETS, 7, 2, seed=0, validation=(XOR_INPUTS, XOR_TARGETS),
    )
    assert list(history) == ["loss", "accuracy", "val_loss", "val_accuracy"]
    for values in history.values():
        assert len(values) == 7
        assert all(type(value) is float for value in values)
    json.dumps(history)


def test_validation_does_not_change_the_training():
    validation = (np.array([[0.2, 0.8], [0.9, 0.1]]), np.eye(2)[[1, 1]])
    loss = get_loss("cross_entropy")
    without, with_validation = xor_network(2), xor_network(2)
    history = train(
        without, loss, get_optimizer("adam", learning_rate=0.01),
        XOR_INPUTS, XOR_TARGETS, 30, 2, seed=5,
    )
    validated = train(
        with_validation, loss, get_optimizer("adam", learning_rate=0.01),
        XOR_INPUTS, XOR_TARGETS, 30, 2, seed=5, validation=validation,
    )
    for first, second in zip(without.layers, with_validation.layers):
        assert np.array_equal(first.weights, second.weights)
        assert np.array_equal(first.biases, second.biases)
    assert validated["loss"] == history["loss"]
    outputs = with_validation.forward(validation[0])
    assert validated["val_loss"][-1] == loss.forward(outputs, validation[1])
    assert validated["val_accuracy"][-1] == np.mean(with_validation.predict(validation[0]) == 1)


def test_regression_has_no_accuracy():
    history = train(
        Network(1, [(1, "identity")], seed=0), get_loss("mse"), get_optimizer("sgd"),
        INDEX_INPUTS, INDEX_TARGETS, 2, 5, seed=0, validation=(INDEX_INPUTS, INDEX_TARGETS),
    )
    assert list(history) == ["loss", "val_loss"]


def test_evaluate_computed_by_hand():
    network = Network(2, [(2, "identity")])
    network.layers[0].weights = np.eye(2)
    inputs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    targets = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    loss = get_loss("cross_entropy")
    metrics = evaluate(network, loss, inputs, targets)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["loss"] == pytest.approx(loss.forward(inputs, targets))


@pytest.mark.parametrize("loss_name, columns, one_hot", [
    ("cross_entropy", 3, True),
    ("mse", 2, False),
])
def test_evaluate_by_chunks_matches_the_whole_dataset(loss_name, columns, one_hot):
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(2500, 5))
    labels = rng.integers(0, columns, size=2500)
    targets = np.eye(columns)[labels] if one_hot else rng.normal(size=(2500, columns))
    network = Network(5, [(8, "relu"), (columns, "identity")], seed=1)
    loss = get_loss(loss_name)
    metrics = evaluate(network, loss, inputs, targets)
    outputs = network.forward(inputs)
    assert metrics["loss"] == pytest.approx(loss.forward(outputs, targets), abs=1e-12)
    if one_hot:
        assert metrics["accuracy"] == np.mean(np.argmax(outputs, axis=1) == labels)
    else:
        assert "accuracy" not in metrics


@pytest.mark.parametrize("targets, expected", [
    (np.eye(3), True),
    (np.array([[0.0, 1.0], [0.0, 1.0]]), True),
    (np.ones((3, 1)), False),
    (np.array([[0.5, 0.5]]), False),
    (np.array([[1.0, 1.0]]), False),
    (np.array([[0.0, 0.0]]), False),
])
def test_is_one_hot(targets, expected):
    assert is_one_hot(targets) is expected


def test_verbose_prints_one_line_per_epoch(capsys):
    history = train(
        xor_network(), get_loss("cross_entropy"), get_optimizer("adam", learning_rate=0.01),
        XOR_INPUTS, XOR_TARGETS, 12, 2, seed=0, validation=(XOR_INPUTS, XOR_TARGETS),
        verbose=True,
    )
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert captured.err == ""
    assert len(lines) == 12
    pattern = (
        r"epoch +(\d+)/12 - loss \d+\.\d{4} - accuracy \d\.\d{4}"
        r" - val_loss \d+\.\d{4} - val_accuracy \d\.\d{4}"
    )
    assert [int(re.fullmatch(pattern, line).group(1)) for line in lines] == list(range(1, 13))
    assert lines[0].startswith("epoch  1/12 - ")
    assert lines[-1] == "epoch 12/12 - " + " - ".join(
        f"{name} {values[-1]:.4f}" for name, values in history.items()
    )


def test_silent_by_default(capsys):
    train(
        xor_network(), get_loss("cross_entropy"), get_optimizer("sgd"),
        XOR_INPUTS, XOR_TARGETS, 3, 2, seed=0,
    )
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("batch_size, learning_rate", [(1, 0.1), (2, 0.5), (4, 0.5)])
def test_mini_batch_training_learns_xor(batch_size, learning_rate, seed):
    network = xor_network(seed)
    history = train(
        network, get_loss("cross_entropy"), get_optimizer("sgd", learning_rate=learning_rate),
        XOR_INPUTS, XOR_TARGETS, 1000, batch_size, seed=seed,
    )
    assert history["loss"][-1] < history["loss"][0]
    assert history["loss"][-1] < 0.01
    assert history["accuracy"][-1] == 1.0
    assert network.predict(XOR_INPUTS).tolist() == XOR_LABELS.tolist()


def test_mini_batch_training_finds_the_solution_of_a_linear_regression():
    inputs = np.array(
        [[a, b] for a in np.linspace(-1.0, 1.0, 5) for b in np.linspace(-1.0, 1.0, 5)]
    )
    weights = np.array([[2.0], [-3.0]])
    network = Network(2, [(1, "identity")], seed=0)
    train(
        network, get_loss("mse"), get_optimizer("sgd", learning_rate=0.1),
        inputs, inputs @ weights + 1.0, 500, 5, seed=0,
    )
    assert network.layers[0].weights == pytest.approx(weights, abs=1e-6)
    assert network.layers[0].biases == pytest.approx(np.array([1.0]), abs=1e-6)


@pytest.mark.parametrize("option, value", [
    ("epochs", 0),
    ("epochs", -1),
    ("epochs", 1.5),
    ("epochs", True),
    ("epochs", "3"),
    ("epochs", None),
    ("batch_size", 0),
    ("batch_size", 2.0),
])
def test_invalid_counts_raise_an_error(option, value):
    options = {"epochs": 1, "batch_size": 2, option: value}
    with pytest.raises(MyTorchError, match=option):
        train(
            Network(1, [(1, "identity")]), get_loss("mse"), get_optimizer("sgd"),
            INDEX_INPUTS, INDEX_TARGETS, **options,
        )


def test_numpy_integers_are_valid_counts():
    train(
        Network(1, [(1, "identity")]), get_loss("mse"), get_optimizer("sgd"),
        INDEX_INPUTS, INDEX_TARGETS, np.int64(2), np.int32(3),
    )


@pytest.mark.parametrize("inputs, targets, message", [
    (INDEX_INPUTS[:, 0], INDEX_TARGETS, "2D"),
    (INDEX_INPUTS.reshape(10, 1, 1), INDEX_TARGETS, "2D"),
    (INDEX_INPUTS, INDEX_TARGETS[:5], "same number of examples"),
    (INDEX_INPUTS[:0], INDEX_TARGETS[:0], "empty"),
])
@pytest.mark.parametrize("dataset", ["training", "validation"])
def test_invalid_datasets_raise_an_error(dataset, inputs, targets, message):
    if dataset == "training":
        arguments = (inputs, targets, 1, 2)
        options = {}
    else:
        arguments = (INDEX_INPUTS, INDEX_TARGETS, 1, 2)
        options = {"validation": (inputs, targets)}
    with pytest.raises(MyTorchError, match=f"{dataset}.*{message}"):
        train(
            Network(1, [(1, "identity")]), get_loss("mse"), get_optimizer("sgd"),
            *arguments, **options,
        )


def test_lists_are_accepted():
    history = train(
        Network(2, [(2, "identity")], seed=0), get_loss("cross_entropy"), get_optimizer("sgd"),
        XOR_INPUTS.tolist(), XOR_TARGETS.tolist(), 2, 2, seed=0,
    )
    assert len(history["loss"]) == 2


RANDOM = np.random.default_rng(0)
NOISY_INPUTS = RANDOM.normal(size=(60, 6))
NOISY_TARGETS = np.eye(3)[RANDOM.integers(0, 3, 60)]
UNRELATED_INPUTS = RANDOM.normal(size=(40, 6))
UNRELATED_TARGETS = np.eye(3)[RANDOM.integers(0, 3, 40)]


def penalised_loss(network, loss, inputs, targets, l2):
    data = loss.forward(network.forward(inputs), targets)
    return data + l2 / 2 * sum((layer.weights ** 2).sum() for layer in network.layers)


def test_l2_gradient_matches_finite_differences_of_the_penalised_loss():
    loss, l2, learning_rate = get_loss("cross_entropy"), 0.3, 0.1
    network = Network(6, [(5, "tanh"), (3, "identity")], seed=1)
    numeric = []
    for layer in network.layers:
        gradient = np.zeros_like(layer.weights)
        for index in np.ndindex(layer.weights.shape):
            layer.weights[index] += 1e-6
            up = penalised_loss(network, loss, NOISY_INPUTS, NOISY_TARGETS, l2)
            layer.weights[index] -= 2e-6
            down = penalised_loss(network, loss, NOISY_INPUTS, NOISY_TARGETS, l2)
            layer.weights[index] += 1e-6
            gradient[index] = (up - down) / 2e-6
        numeric.append(gradient)
    before = [layer.weights.copy() for layer in network.layers]
    train(network, loss, get_optimizer("sgd", learning_rate=learning_rate), NOISY_INPUTS,
          NOISY_TARGETS, 1, 60, shuffle=False, l2=l2)
    for old, layer, expected in zip(before, network.layers, numeric):
        assert (old - layer.weights) / learning_rate == pytest.approx(expected, abs=1e-7)


def train_one_step(l2):
    network = Network(6, [(3, "identity")], seed=2)
    # Biases start at zero, where a penalty on them would change nothing: start them elsewhere.
    network.layers[0].biases = np.array([[0.5, -1.0, 2.0]])
    train(network, get_loss("cross_entropy"), get_optimizer("sgd", learning_rate=0.1),
          NOISY_INPUTS, NOISY_TARGETS, 1, 60, shuffle=False, l2=l2)
    return network.layers[0]


def test_l2_does_not_touch_the_biases():
    plain, penalised = train_one_step(0.0), train_one_step(1.0)
    assert np.array_equal(plain.biases, penalised.biases)
    assert not np.array_equal(plain.weights, penalised.weights)


def test_l2_shrinks_the_weights():
    norms = []
    for l2 in (0.0, 0.05):
        network = Network(6, [(16, "relu"), (3, "identity")], seed=3)
        train(network, get_loss("cross_entropy"), get_optimizer("adam", learning_rate=0.01),
              NOISY_INPUTS, NOISY_TARGETS, 100, 10, seed=0, l2=l2)
        norms.append(sum((layer.weights ** 2).sum() for layer in network.layers))
    assert norms[1] < norms[0] / 5


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), "0.1", True])
def test_invalid_l2_raises_an_error(value):
    with pytest.raises(MyTorchError, match="l2"):
        train(Network(6, [(3, "identity")]), get_loss("mse"), get_optimizer("sgd"),
              NOISY_INPUTS, NOISY_TARGETS, 1, 10, l2=value)


def train_with_early_stopping(monitor, patience=5, epochs=200, verbose=False):
    network = Network(6, [(32, "relu"), (3, "identity")], seed=0)
    optimizer = get_optimizer("adam", learning_rate=0.01)
    history = train(network, get_loss("cross_entropy"), optimizer, NOISY_INPUTS, NOISY_TARGETS,
                    epochs, 10, seed=0, validation=(UNRELATED_INPUTS, UNRELATED_TARGETS),
                    patience=patience, monitor=monitor, verbose=verbose)
    return network, optimizer, history


@pytest.mark.parametrize("monitor, pick", [("val_loss", np.argmin), ("val_accuracy", np.argmax)])
def test_training_stops_patience_epochs_after_the_best_one(monitor, pick):
    network, _, history = train_with_early_stopping(monitor)
    best = int(pick(history[monitor])) + 1
    assert len(history[monitor]) == best + 5 < 200
    restored = evaluate(network, get_loss("cross_entropy"), UNRELATED_INPUTS, UNRELATED_TARGETS)
    assert restored[monitor[4:]] == pytest.approx(history[monitor][best - 1])


def test_the_best_network_and_optimizer_are_restored():
    network, optimizer, history = train_with_early_stopping("val_loss", patience=3, epochs=50)
    best = int(np.argmin(history["val_loss"])) + 1
    reference = Network(6, [(32, "relu"), (3, "identity")], seed=0)
    reference_optimizer = get_optimizer("adam", learning_rate=0.01)
    train(reference, get_loss("cross_entropy"), reference_optimizer, NOISY_INPUTS,
          NOISY_TARGETS, best, 10, seed=0)
    for layer, expected in zip(network.layers, reference.layers):
        assert np.array_equal(layer.weights, expected.weights)
        assert np.array_equal(layer.biases, expected.biases)
    assert optimizer.steps == reference_optimizer.steps
    for key, array in reference_optimizer.state().items():
        assert np.array_equal(optimizer.state()[key], array)


def test_the_best_epoch_is_restored_even_without_stopping():
    network, _, history = train_with_early_stopping("val_loss", patience=100, epochs=12)
    assert len(history["val_loss"]) == 12
    best = min(history["val_loss"])
    restored = evaluate(network, get_loss("cross_entropy"), UNRELATED_INPUTS, UNRELATED_TARGETS)
    assert restored["loss"] == pytest.approx(best)


def test_early_stopping_is_reported(capsys):
    _, _, history = train_with_early_stopping("val_accuracy", verbose=True)
    best = int(np.argmax(history["val_accuracy"])) + 1
    last = capsys.readouterr().out.splitlines()[-1]
    assert last == (f"early stopping: best val_accuracy {history['val_accuracy'][best - 1]:.4f} "
                    f"at epoch {best}")


def test_without_patience_the_monitor_changes_nothing():
    histories = []
    for monitor in ("val_loss", "val_accuracy"):
        network = Network(6, [(8, "relu"), (3, "identity")], seed=4)
        histories.append(train(network, get_loss("cross_entropy"), get_optimizer("sgd"),
                               NOISY_INPUTS, NOISY_TARGETS, 5, 10, seed=1,
                               validation=(UNRELATED_INPUTS, UNRELATED_TARGETS), monitor=monitor))
    assert histories[0] == histories[1]


@pytest.mark.parametrize("options, message", [
    ({"patience": 3}, "needs a validation set"),
    ({"patience": 0, "validation": (UNRELATED_INPUTS, UNRELATED_TARGETS)}, "patience"),
    ({"patience": True, "validation": (UNRELATED_INPUTS, UNRELATED_TARGETS)}, "patience"),
    ({"patience": 2, "validation": (UNRELATED_INPUTS, UNRELATED_TARGETS), "monitor": "loss"},
     "monitor must be one of"),
    ({"patience": 2, "validation": (UNRELATED_INPUTS, UNRELATED_INPUTS[:, :3]),
      "monitor": "val_accuracy"}, "one-hot"),
])
def test_invalid_early_stopping_raises_an_error(options, message):
    with pytest.raises(MyTorchError, match=message):
        train(Network(6, [(3, "identity")]), get_loss("cross_entropy"), get_optimizer("sgd"),
              NOISY_INPUTS, NOISY_TARGETS, 2, 10, **options)
