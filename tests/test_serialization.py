"""Tests of the saving and loading of networks in .nn files."""

import json
import os
import stat
import zipfile

import numpy as np
import pytest

import my_torch.serialization as serialization
from my_torch import (
    MyTorchError, Network, SavedNetwork, get_loss, get_optimizer, load_network,
    save_network, train,
)
from my_torch.losses import loss_name

XOR_INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_TARGETS = np.eye(2)[[0, 1, 1, 0]]
ARCHITECTURE = [(4, "tanh"), (3, "relu"), (2, "identity")]
OPTIMIZERS = [
    ("sgd", {}),
    ("momentum", {}),
    ("momentum", {"nesterov": True}),
    ("rmsprop", {}),
    ("adam", {}),
]


def trained(name="adam", params=None, epochs=10, seed=0):
    network = Network(2, ARCHITECTURE, seed=seed)
    optimizer = get_optimizer(name, learning_rate=0.01, **(params or {}))
    loss = get_loss("cross_entropy")
    train(network, loss, optimizer, XOR_INPUTS, XOR_TARGETS, epochs, 4, shuffle=False)
    return network, loss, optimizer


def saved_file(tmp_path):
    network, loss, optimizer = trained()
    path = tmp_path / "network.nn"
    save_network(path, network, loss, optimizer, {"batch_size": 4, "epochs": 10})
    return path


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def write_arrays(path, arrays):
    with open(path, "wb") as file:
        np.savez(file, **arrays)
    return path


@pytest.mark.parametrize("activation", ["identity", "sigmoid", "tanh", "relu", "leaky_relu"])
def test_loaded_network_gives_the_same_predictions(tmp_path, activation):
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(200, 6))
    network = Network(6, [(8, activation), (5, activation), (3, "softmax")], seed=2)
    save_network(tmp_path / "network.nn", network)
    loaded = load_network(tmp_path / "network.nn").network
    assert np.array_equal(loaded.forward(inputs), network.forward(inputs))
    assert np.array_equal(loaded.predict(inputs), network.predict(inputs))
    assert loaded.input_size == 6 and loaded.architecture == network.architecture


def test_parameters_are_restored_exactly(tmp_path):
    network, _, _ = trained()
    save_network(tmp_path / "network.nn", network)
    loaded = load_network(tmp_path / "network.nn").network
    for expected, layer in zip(network.layers, loaded.layers):
        assert layer.weights.dtype == np.float64
        assert np.array_equal(expected.weights, layer.weights)
        assert np.array_equal(expected.biases, layer.biases)


def test_training_information_is_restored(tmp_path):
    saved = load_network(saved_file(tmp_path))
    _, loss, optimizer = trained()
    assert isinstance(saved, SavedNetwork)
    assert loss_name(saved.loss) == "cross_entropy"
    assert type(saved.optimizer) is type(optimizer)
    assert saved.optimizer.hyperparameters() == optimizer.hyperparameters()
    assert saved.optimizer.steps == optimizer.steps
    assert saved.settings == {"batch_size": 4, "epochs": 10}


def test_optional_parts_are_none_when_not_saved(tmp_path):
    network, _, _ = trained()
    save_network(tmp_path / "network.nn", network)
    saved = load_network(tmp_path / "network.nn")
    assert (saved.loss, saved.optimizer, saved.settings) == (None, None, None)


@pytest.mark.parametrize("name, params", OPTIMIZERS)
def test_training_resumes_exactly_after_loading(tmp_path, name, params):
    straight, _, _ = trained(name, params, epochs=20)
    network, loss, optimizer = trained(name, params, epochs=10)
    save_network(tmp_path / "network.nn", network, loss, optimizer)
    saved = load_network(tmp_path / "network.nn")
    train(saved.network, saved.loss, saved.optimizer, XOR_INPUTS, XOR_TARGETS, 10, 4,
          shuffle=False)
    for expected, layer in zip(straight.layers, saved.network.layers):
        assert np.array_equal(expected.weights, layer.weights)
        assert np.array_equal(expected.biases, layer.biases)


def test_file_holds_only_numpy_arrays_and_a_json_description(tmp_path):
    path = saved_file(tmp_path)
    with zipfile.ZipFile(path) as archive:
        assert all(name.endswith(".npy") for name in archive.namelist())
    arrays = read_arrays(path)
    metadata = json.loads(str(arrays["metadata"]))
    assert metadata["format"] == "my_torch" and metadata["version"] == 1
    assert metadata["input_size"] == 2
    assert metadata["architecture"] == [[4, "tanh"], [3, "relu"], [2, "identity"]]
    assert metadata["loss"] == "cross_entropy"
    assert metadata["optimizer"]["name"] == "adam"
    assert "layers/2/biases" in arrays and "optimizer/steps" in arrays


def test_saving_does_not_add_an_extension(tmp_path):
    network, _, _ = trained()
    save_network(tmp_path / "my_torch_network.nn", network)
    assert os.listdir(tmp_path) == ["my_torch_network.nn"]


def test_saved_file_follows_the_umask(tmp_path):
    network, _, _ = trained()
    save_network(tmp_path / "network.nn", network)
    mode = stat.S_IMODE(os.stat(tmp_path / "network.nn").st_mode)
    assert mode == 0o666 & ~serialization.current_umask()


def test_saving_replaces_an_existing_file(tmp_path):
    path = saved_file(tmp_path)
    network = Network(2, [(2, "identity")], seed=5)
    save_network(path, network)
    assert load_network(path).network.architecture == [(2, "identity")]
    assert os.listdir(tmp_path) == ["network.nn"]


def test_interrupted_save_keeps_the_previous_file(tmp_path, monkeypatch):
    path = saved_file(tmp_path)
    content = path.read_bytes()
    real_savez = np.savez

    def interrupted(*args, **kwargs):
        real_savez(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(serialization.np, "savez", interrupted)
    with pytest.raises(KeyboardInterrupt):
        save_network(path, Network(2, [(2, "identity")]))
    assert path.read_bytes() == content
    assert os.listdir(tmp_path) == ["network.nn"]


def broken_network(layer, parameter, value):
    network = Network(2, [(3, "relu"), (2, "identity")], seed=0)
    getattr(network.layers[layer], parameter)[0] = value
    return network


@pytest.mark.parametrize("arguments", [
    lambda: {"network": broken_network(0, "weights", np.nan)},
    lambda: {"network": broken_network(1, "biases", np.inf)},
    lambda: {"network": trained()[0], "settings": [1]},
    lambda: {"network": trained()[0], "settings": {"rate": float("nan")}},
    lambda: {"network": trained()[0], "settings": {"value": object()}},
    lambda: {"network": trained()[0], "loss": type("Custom", (type(get_loss("mse")),), {})()},
])
def test_invalid_saves_raise_an_error_and_keep_the_file(tmp_path, arguments):
    path = saved_file(tmp_path)
    content = path.read_bytes()
    with pytest.raises(MyTorchError):
        save_network(path, **arguments())
    assert path.read_bytes() == content
    assert os.listdir(tmp_path) == ["network.nn"]


def test_non_finite_optimizer_state_is_not_saved(tmp_path):
    network, loss, optimizer = trained("momentum")
    optimizer.velocities[(0, "weights")][0, 0] = np.nan
    with pytest.raises(MyTorchError, match="non-finite"):
        save_network(tmp_path / "network.nn", network, loss, optimizer)
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize("target", ["missing/network.nn", "."])
def test_unwritable_paths_raise_an_error(tmp_path, target):
    with pytest.raises(MyTorchError, match="cannot save network"):
        save_network(tmp_path / target, trained()[0])


def test_missing_file_raises_an_error(tmp_path):
    with pytest.raises(MyTorchError, match="no such file"):
        load_network(tmp_path / "missing.nn")


def test_directory_raises_an_error(tmp_path):
    with pytest.raises(MyTorchError, match="cannot load network"):
        load_network(tmp_path)


@pytest.mark.parametrize("content", [
    b"",
    b"hello\n",
    np.random.default_rng(0).bytes(5000),
])
def test_files_that_are_not_archives_raise_an_error(tmp_path, content):
    path = tmp_path / "network.nn"
    path.write_bytes(content)
    with pytest.raises(MyTorchError, match="invalid network file"):
        load_network(path)


def test_truncated_file_raises_an_error(tmp_path):
    path = saved_file(tmp_path)
    path.write_bytes(path.read_bytes()[:2000])
    with pytest.raises(MyTorchError, match="invalid network file"):
        load_network(path)


def test_corrupted_byte_raises_an_error(tmp_path):
    path = saved_file(tmp_path)
    content = bytearray(path.read_bytes())
    content[content.find(b"layers/0/weights") + 200] ^= 0xFF
    path.write_bytes(bytes(content))
    with pytest.raises(MyTorchError, match="invalid network file"):
        load_network(path)


def test_single_numpy_array_raises_an_error(tmp_path):
    with open(tmp_path / "network.nn", "wb") as file:
        np.save(file, np.arange(3.0))
    with pytest.raises(MyTorchError, match="invalid network file"):
        load_network(tmp_path / "network.nn")


def test_pickled_object_is_never_loaded(tmp_path):
    arrays = read_arrays(saved_file(tmp_path))
    with open(tmp_path / "evil.nn", "wb") as file:
        np.savez(file, **arrays, evil=np.array([{"code": 1}], dtype=object))
    with pytest.raises(MyTorchError, match="invalid network file"):
        load_network(tmp_path / "evil.nn")


def with_metadata(arrays, **changes):
    metadata = json.loads(str(arrays["metadata"]))
    for key, value in changes.items():
        if value is KeyError:
            del metadata[key]
        else:
            metadata[key] = value
    return {**arrays, "metadata": np.array(json.dumps(metadata))}


def without(arrays, key):
    return {name: value for name, value in arrays.items() if name != key}


def replaced(arrays, key, change):
    return {**arrays, key: change(arrays[key])}


@pytest.mark.parametrize("change, message", [
    (lambda a: without(a, "metadata"), "missing description"),
    (lambda a: {**a, "metadata": np.array(3)}, "missing description"),
    (lambda a: {**a, "metadata": np.array("{not json")}, "unreadable"),
    (lambda a: {**a, "metadata": np.array(str(a["metadata"]).replace("10}", "NaN}"))},
     "unreadable"),
    (lambda a: {**a, "metadata": np.array("[1, 2]")}, "not a my_torch network"),
    (lambda a: with_metadata(a, format="other"), "not a my_torch network"),
    (lambda a: with_metadata(a, version=2), "unsupported format version 2"),
    (lambda a: with_metadata(a, version=KeyError), "unsupported format version"),
    (lambda a: with_metadata(a, settings=KeyError), "description must hold exactly"),
    (lambda a: with_metadata(a, extra=1), "description must hold exactly"),
    (lambda a: with_metadata(a, input_size=0), "input size"),
    (lambda a: with_metadata(a, input_size=2.5), "input size"),
    (lambda a: with_metadata(a, input_size=True), "input size"),
    (lambda a: with_metadata(a, input_size=3), "layers/0/weights"),
    (lambda a: with_metadata(a, architecture=[]), "at least one layer"),
    (lambda a: with_metadata(a, architecture=[[4, "tanh"], [3, "relu"]], dropout=[0, 0]),
     "without a matching layer"),
    (lambda a: with_metadata(a, architecture=[[4, "tanh"], [3, "relu"], [2, "softplus"]]),
     "unknown activation"),
    (lambda a: with_metadata(a, architecture=[[4, "tanh"], [3, "relu"], [2]]), "pairs"),
    (lambda a: with_metadata(a, architecture=[[4, "tanh"], [3, "relu"], [-2, "identity"]]),
     "pairs"),
    (lambda a: with_metadata(a, loss="hinge"), "unknown loss"),
    (lambda a: with_metadata(a, loss=3), "loss must be a name"),
    (lambda a: with_metadata(a, optimizer={"name": "lbfgs", "hyperparameters": {}}),
     "unknown optimizer"),
    (lambda a: with_metadata(a, optimizer={"name": "adam", "hyperparameters": {"momentum": 1}}),
     "invalid parameters"),
    (lambda a: with_metadata(a, optimizer={"name": "adam",
                                           "hyperparameters": {"learning_rate": -1}}),
     "learning rate"),
    (lambda a: with_metadata(a, optimizer={"name": "adam"}), "name and hyperparameters"),
    (lambda a: with_metadata(a, optimizer={"name": "momentum", "hyperparameters": {}}),
     "unexpected optimizer state"),
    (lambda a: with_metadata(a, optimizer=None), "state without optimizer"),
    (lambda a: with_metadata(a, settings=[1]), "settings"),
    (lambda a: without(a, "layers/1/biases"), "layers/1/biases"),
    (lambda a: replaced(a, "layers/0/weights", lambda w: w[:1]), "layers/0/weights"),
    (lambda a: replaced(a, "layers/0/weights", lambda w: w.astype(int)), "layers/0/weights"),
    (lambda a: replaced(a, "layers/0/weights", lambda w: np.full_like(w, np.nan)),
     "non-finite"),
    (lambda a: replaced(a, "layers/0/biases", lambda b: np.full_like(b, np.inf)), "non-finite"),
    (lambda a: {**a, "layers/9/weights": np.zeros(2)}, "without a matching layer"),
    (lambda a: {**a, "notes": np.array("hi")}, "unexpected arrays"),
    (lambda a: without(a, "optimizer/steps"), "steps"),
    (lambda a: replaced(a, "optimizer/squares/0/weights", lambda s: -np.ones_like(s)),
     "invalid values"),
])
def test_inconsistent_files_raise_an_error(tmp_path, change, message):
    arrays = read_arrays(saved_file(tmp_path))
    path = write_arrays(tmp_path / "broken.nn", change(arrays))
    with pytest.raises(MyTorchError, match="invalid network file") as error:
        load_network(path)
    assert message in str(error.value)
