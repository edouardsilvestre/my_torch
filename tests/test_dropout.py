"""Tests of dropout and of MC dropout."""

import json
import tomllib

import numpy as np
import pytest

import analyzer.predict as prediction
from my_torch import (
    MyTorchError, Network, get_loss, get_optimizer, load_network, mc_dropout, save_network, train,
)
from my_torch.activations import Softmax
from my_torch.config import parse_config

RANDOM = np.random.default_rng(0)
INPUTS = RANDOM.normal(size=(12, 5))
TARGETS = np.eye(3)[RANDOM.integers(0, 3, 12)]


def make_network(dropout=(0.3, 0.5, 0.0), seed=1):
    return Network(5, [(6, "tanh"), (4, "relu"), (3, "identity")], seed=seed,
                   dropout=list(dropout))


def same_weights(network, dropout):
    """A copy of the network, with other dropout rates."""
    copy = make_network(dropout)
    for source, target in zip(network.layers, copy.layers):
        target.weights, target.biases = source.weights.copy(), source.biases.copy()
    return copy


# Network


def test_gradient_with_dropout_matches_finite_differences():
    network, loss = make_network(), get_loss("cross_entropy")

    def loss_with_fixed_masks():
        # The same generator state gives the same masks at each pass.
        return loss.forward(network.forward(INPUTS, np.random.default_rng(7)), TARGETS)

    outputs = network.forward(INPUTS, np.random.default_rng(7))
    network.backward(loss.backward(outputs, TARGETS))
    assert any(mask is not None and not mask.all() for mask in network.masks)
    for layer in network.layers:
        analytic = layer.weights_gradient.copy()
        numeric = np.zeros_like(layer.weights)
        for index in np.ndindex(layer.weights.shape):
            layer.weights[index] += 1e-6
            up = loss_with_fixed_masks()
            layer.weights[index] -= 2e-6
            down = loss_with_fixed_masks()
            layer.weights[index] += 1e-6
            numeric[index] = (up - down) / 2e-6
        assert analytic == pytest.approx(numeric, abs=1e-7)


def test_masks_drop_the_given_share_and_scale_the_others():
    network = Network(5, [(4000, "relu"), (2, "identity")], seed=0, dropout=[0.25, 0.0])
    network.forward(INPUTS, np.random.default_rng(0))
    mask = network.masks[0]
    assert set(np.unique(mask)) == {0.0, 1 / 0.75}
    assert np.mean(mask == 0) == pytest.approx(0.25, abs=0.01)
    # Inverted dropout: on average, each output keeps its value.
    assert mask.mean() == pytest.approx(1.0, abs=0.01)
    assert network.masks[1] is None


def test_dropped_outputs_are_zero_and_kept_ones_are_scaled():
    network = make_network((0.5, 0.0, 0.0))
    reference = same_weights(network, (0.0, 0.0, 0.0))
    network.forward(INPUTS, np.random.default_rng(3))
    mask = network.masks[0]
    hidden = reference.activations[0].forward(reference.layers[0].forward(INPUTS))
    assert np.array_equal(network.layers[1].inputs, hidden * mask)


def test_without_a_generator_every_neuron_is_used():
    network = make_network()
    reference = same_weights(network, (0.0, 0.0, 0.0))
    assert np.array_equal(network.forward(INPUTS), reference.forward(INPUTS))
    assert network.masks == [None, None, None]


def test_an_old_mask_is_not_reused_after_a_pass_without_dropout():
    network, loss = make_network(), get_loss("mse")
    network.forward(INPUTS, np.random.default_rng(0))
    outputs = network.forward(INPUTS)
    network.backward(loss.backward(outputs, TARGETS))
    reference = same_weights(network, (0.0, 0.0, 0.0))
    reference.backward(loss.backward(reference.forward(INPUTS), TARGETS))
    for layer, expected in zip(network.layers, reference.layers):
        assert np.array_equal(layer.weights_gradient, expected.weights_gradient)


@pytest.mark.parametrize("dropout, message", [
    ([0.1, 0.0], "expected 3 dropout rates"),
    ([0.1, 0.2, 0.0, 0.0], "expected 3 dropout rates"),
    ([1.0, 0.0, 0.0], "layer 0 must be at least 0 and below 1"),
    ([0.0, -0.1, 0.0], "layer 1 must be at least 0 and below 1"),
    ([True, 0.0, 0.0], "layer 0"),
    (["0.1", 0.0, 0.0], "layer 0"),
    ([0.0, 0.0, 0.5], "the last layer cannot have dropout"),
])
def test_invalid_dropout_rates(dropout, message):
    with pytest.raises(MyTorchError, match=message):
        make_network(dropout)


# Training


def train_network(dropout, seed=0, epochs=3):
    network = make_network(dropout)
    history = train(network, get_loss("cross_entropy"), get_optimizer("sgd", learning_rate=0.1),
                    INPUTS, TARGETS, epochs, 4, seed=seed)
    return network, history


def test_training_with_dropout_is_reproducible_and_differs_from_without():
    first, _ = train_network((0.5, 0.5, 0.0))
    again, _ = train_network((0.5, 0.5, 0.0))
    plain, _ = train_network((0.0, 0.0, 0.0))
    other_seed, _ = train_network((0.5, 0.5, 0.0), seed=1)
    for a, b, c, d in zip(first.layers, again.layers, plain.layers, other_seed.layers):
        assert np.array_equal(a.weights, b.weights)
        assert not np.array_equal(a.weights, c.weights)
        assert not np.array_equal(a.weights, d.weights)


def test_dropout_does_not_change_the_order_of_the_examples(monkeypatch):
    orders = {}
    for rates in ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0)):
        network, seen = make_network(rates), []
        original_forward = network.forward

        def forward(inputs, dropout_rng=None, seen=seen, original_forward=original_forward):
            seen.append(inputs[:, 0].tolist())
            return original_forward(inputs, dropout_rng)

        network.forward = forward
        train(network, get_loss("mse"), get_optimizer("sgd"), INPUTS, TARGETS, 2, 4, seed=0)
        orders[rates] = [batch for batch in seen if len(batch) == 4]
    assert orders[(0.0, 0.0, 0.0)] == orders[(0.5, 0.5, 0.0)]


def test_dropout_is_only_active_on_the_training_batches():
    network, history = train_network((0.5, 0.5, 0.0))
    loss = get_loss("cross_entropy").forward(network.forward(INPUTS), TARGETS)
    assert history["loss"][-1] == pytest.approx(loss)


def test_rates_of_zero_train_exactly_like_no_dropout():
    zero, _ = train_network((0.0, 0.0, 0.0))
    plain = Network(5, [(6, "tanh"), (4, "relu"), (3, "identity")], seed=1)
    train(plain, get_loss("cross_entropy"), get_optimizer("sgd", learning_rate=0.1),
          INPUTS, TARGETS, 3, 4, seed=0)
    for a, b in zip(zero.layers, plain.layers):
        assert np.array_equal(a.weights, b.weights)


# Configuration and saving


CONFIG = '''[network]
input_size = 5
layers = [
    {{ size = 6, activation = "tanh"{first} }},
    {{ size = 3, activation = "identity"{last} }},
]

[training]
loss = "cross_entropy"
epochs = 1
batch_size = 4

[optimizer]
name = "sgd"
'''


def config(first="", last=""):
    return parse_config(tomllib.loads(CONFIG.format(first=first, last=last)))


def test_dropout_is_read_from_the_configuration():
    assert config(", dropout = 0.25").create_network().dropout == [0.25, 0.0]
    assert config().create_network().dropout == [0.0, 0.0]
    assert config(", dropout = 0").create_network().dropout == [0.0, 0.0]


@pytest.mark.parametrize("first, last, message", [
    (", dropout = 1", "", r"network.layers\[0\].dropout must be in \[0, 1\[, got 1.0"),
    (", dropout = -0.5", "", r"network.layers\[0\].dropout must be in \[0, 1\["),
    (', dropout = "0.5"', "", r"network.layers\[0\].dropout must be a finite number"),
    ("", ", dropout = 0.1", r"network.layers\[1\].dropout: the last layer cannot have dropout"),
])
def test_invalid_dropout_in_the_configuration(first, last, message):
    with pytest.raises(MyTorchError, match=message):
        config(first, last)


def test_dropout_is_saved_with_the_network(tmp_path):
    network = make_network((0.2, 0.4, 0.0))
    save_network(tmp_path / "net.nn", network)
    assert load_network(tmp_path / "net.nn").network.dropout == [0.2, 0.4, 0.0]


def rewrite_metadata(path, change):
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    metadata = json.loads(str(arrays["metadata"]))
    change(metadata)
    arrays["metadata"] = np.array(json.dumps(metadata))
    # Through a file object: given a name, np.savez would add ".npz" to it.
    with open(path, "wb") as file:
        np.savez(file, **arrays)


def test_a_file_written_before_dropout_still_loads(tmp_path):
    path = tmp_path / "old.nn"
    save_network(path, make_network((0.0, 0.0, 0.0)))
    rewrite_metadata(path, lambda metadata: metadata.pop("dropout"))
    with np.load(path, allow_pickle=False) as archive:
        assert "dropout" not in json.loads(str(archive["metadata"]))
    assert load_network(path).network.dropout == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("value, message", [
    (0.5, "dropout must be a list of rates"),
    ([0.1, 0.0], "expected 3 dropout rates"),
    ([0.1, 1.5, 0.0], "layer 1 must be at least 0 and below 1"),
    ([0.1, 0.0, 0.3], "the last layer cannot have dropout"),
])
def test_invalid_dropout_in_a_file(tmp_path, value, message):
    path = tmp_path / "broken.nn"
    save_network(path, make_network())
    rewrite_metadata(path, lambda metadata: metadata.update(dropout=value))
    with pytest.raises(MyTorchError, match=f"invalid network file.*{message}"):
        load_network(path)


# MC dropout


def test_mc_dropout_averages_passes_with_dropout_active():
    network = make_network()
    mean, std = mc_dropout(network, INPUTS, 5, seed=4)
    rng = np.random.default_rng(4)
    passes = np.array([Softmax().forward(network.forward(INPUTS, rng)) for _ in range(5)])
    assert mean == pytest.approx(passes.mean(axis=0))
    assert std == pytest.approx(passes.std(axis=0))
    assert mean.sum(axis=1) == pytest.approx(np.ones(len(INPUTS)))
    assert (std > 0).any()


def test_mc_dropout_without_softmax_averages_the_raw_outputs():
    network = make_network()
    mean, _ = mc_dropout(network, INPUTS, 3, seed=2, softmax=False)
    rng = np.random.default_rng(2)
    assert mean == pytest.approx(np.mean([network.forward(INPUTS, rng) for _ in range(3)], axis=0))


def test_mc_dropout_is_reproducible_and_one_pass_has_no_spread():
    network = make_network()
    assert np.array_equal(mc_dropout(network, INPUTS, 4, seed=0)[0],
                          mc_dropout(network, INPUTS, 4, seed=0)[0])
    assert not np.array_equal(mc_dropout(network, INPUTS, 4, seed=0)[0],
                              mc_dropout(network, INPUTS, 4, seed=1)[0])
    assert np.array_equal(mc_dropout(network, INPUTS, 1, seed=0)[1], np.zeros((12, 3)))


def test_many_passes_get_close_to_the_network_without_dropout():
    network = Network(5, [(400, "relu"), (3, "identity")], seed=0, dropout=[0.1, 0.0])
    mean, _ = mc_dropout(network, INPUTS, 300, seed=0, softmax=False)
    assert mean == pytest.approx(network.forward(INPUTS), abs=0.1)


@pytest.mark.parametrize("network, passes, message", [
    (make_network(), 0, "passes must be a positive integer"),
    (make_network(), True, "passes must be a positive integer"),
    (make_network(), 2.0, "passes must be a positive integer"),
    (make_network((0.0, 0.0, 0.0)), 3, "MC dropout needs a network with dropout"),
])
def test_invalid_mc_dropout(network, passes, message):
    with pytest.raises(MyTorchError, match=message):
        mc_dropout(network, INPUTS, passes)


def test_predicted_classes_with_mc_dropout(monkeypatch):
    network = Network(5, [(8, "relu"), (5, "identity")], seed=0, dropout=[0.5, 0.0])
    inputs = np.random.default_rng(1).normal(size=(10, 5))
    monkeypatch.setattr(prediction, "PREDICTION_BATCH_SIZE", 4)
    predicted = prediction.predict_classes(network, inputs, mc_passes=6, seed=3)
    rng = np.random.default_rng(3)
    expected = np.concatenate([
        np.argmax(mc_dropout(network, inputs[start:start + 4], 6, rng)[0], axis=1)
        for start in range(0, 10, 4)
    ])
    assert np.array_equal(predicted, expected)
    assert np.array_equal(prediction.predict_classes(network, inputs),
                          np.argmax(network.forward(inputs), axis=1))
