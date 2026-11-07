"""Tests of the configuration files."""

import numpy as np
import pytest

from my_torch import (
    Config, MyTorchError, Network, load_config, load_network, save_network, train,
)
from my_torch.config import check_training_settings, optimizer_parameters
from my_torch.losses import loss_name
from my_torch.optimizers import optimizer_name

XOR_INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_LABELS = np.array([0, 1, 1, 0])
XOR_TARGETS = np.eye(2)[XOR_LABELS]

LAYERS = '''layers = [
    { size = 4, activation = "tanh" },
    { size = 2, activation = "identity" },
]'''

CONFIG = f'''# a comment
[network]
input_size = 2
{LAYERS}
initializer = "xavier_normal"
seed = 0

[training]
loss = "cross_entropy"
epochs = 2000
batch_size = 4

[optimizer]
name = "sgd"
learning_rate = 0.5
'''


def write_config(tmp_path, text):
    path = tmp_path / "network.conf"
    path.write_text(text)
    return path


def config_with(tmp_path, old, new):
    assert old in CONFIG
    return write_config(tmp_path, CONFIG.replace(old, new, 1))


def test_valid_configuration_is_read(tmp_path):
    config = load_config(write_config(tmp_path, CONFIG))
    assert isinstance(config, Config)
    assert config.input_size == 2
    assert config.layers == [(4, "tanh"), (2, "identity")]
    assert (config.initializer, config.seed) == ("xavier_normal", 0)
    assert (config.loss, config.epochs, config.batch_size) == ("cross_entropy", 2000, 4)
    assert (config.optimizer, config.optimizer_params) == ("sgd", {"learning_rate": 0.5})


def test_optional_keys_have_defaults(tmp_path):
    text = CONFIG.replace('initializer = "xavier_normal"\nseed = 0\n', "")
    config = load_config(write_config(tmp_path, text.replace("learning_rate = 0.5\n", "")))
    assert (config.initializer, config.seed) == ("he", None)
    assert (config.shuffle, config.training_seed) == (True, None)
    assert config.optimizer_params == {}
    assert config.create_optimizer().learning_rate == 0.01


def test_integer_numbers_are_accepted_as_decimals(tmp_path):
    path = config_with(tmp_path, "learning_rate = 0.5", "learning_rate = 1")
    params = load_config(path).optimizer_params
    assert params == {"learning_rate": 1.0}
    assert type(params["learning_rate"]) is float


def test_every_optimizer_parameter_can_be_set(tmp_path):
    settings = {
        "momentum": "momentum = 0.8\nnesterov = true",
        "rmsprop": "decay = 0.5\nepsilon = 0.001",
        "adam": "beta1 = 0.5\nbeta2 = 0.6\nepsilon = 0.001",
    }
    for name, lines in settings.items():
        path = config_with(tmp_path, 'name = "sgd"', f'name = "{name}"\n{lines}')
        config = load_config(path)
        optimizer = config.create_optimizer()
        assert optimizer_name(optimizer) == name
        assert optimizer.hyperparameters() == {**optimizer.hyperparameters(),
                                               **config.optimizer_params}


@pytest.mark.parametrize("name, expected", [
    ("sgd", {"learning_rate": float}),
    ("momentum", {"learning_rate": float, "momentum": float, "nesterov": bool}),
    ("rmsprop", {"learning_rate": float, "decay": float, "epsilon": float}),
    ("adam", {"learning_rate": float, "beta1": float, "beta2": float, "epsilon": float}),
])
def test_optimizer_parameters_come_from_the_optimizers(name, expected):
    assert optimizer_parameters(name) == expected


def test_softmax_is_accepted_with_mse(tmp_path):
    text = CONFIG.replace('"cross_entropy"', '"mse"').replace(
        '{ size = 2, activation = "identity" }', '{ size = 2, activation = "softmax" }'
    )
    assert load_config(write_config(tmp_path, text)).layers[-1] == (2, "softmax")


@pytest.mark.parametrize("old, new, message", [
    ("input_size = 2", "input_size = ", "line 3"),
    ("[optimizer]\nname = \"sgd\"\nlearning_rate = 0.5\n", "", "missing key 'optimizer'"),
    ("[optimizer]", "[regularization]\nl2 = 0.1\n\n[optimizer]", "unknown key 'regularization'"),
    ("input_size = 2\n", "", "network: missing key 'input_size'"),
    ("input_size = 2", "input_size = 0", "network.input_size must be a positive integer"),
    ("input_size = 2", "input_size = 2.5", "network.input_size must be a positive integer"),
    ("input_size = 2", "input_size = true", "network.input_size must be a positive integer"),
    ("input_size = 2", 'input_size = "2"', "network.input_size must be a positive integer"),
    (LAYERS, "layers = []", "network.layers must be a non-empty list"),
    (LAYERS, "layers = 3", "network.layers must be a non-empty list"),
    ("size = 4,", "size = 0,", "network.layers[0].size must be a positive integer"),
    ("size = 4,", "size = 4.0,", "network.layers[0].size must be a positive integer"),
    ('{ size = 4, activation = "tanh" }', "{ size = 4 }",
     "network.layers[0]: missing key 'activation'"),
    ('{ size = 4, activation = "tanh" }', '{ size = 4, activation = "tanh", dropout = 0.5 }',
     "network.layers[0]: unknown key 'dropout'"),
    ('{ size = 4, activation = "tanh" }', "4", "network.layers[0] must be a table"),
    ('"tanh"', '"softplus"', "network.layers[0].activation must be one of"),
    ('"xavier_normal"', '"glorot"', "network.initializer must be one of"),
    ("seed = 0", "seed = -1", "network.seed must be a non-negative integer"),
    ("seed = 0", "seed = 1.5", "network.seed must be a non-negative integer"),
    ("seed = 0", "sed = 0", "network: unknown key 'sed'"),
    ('loss = "cross_entropy"\n', "", "training: missing key 'loss'"),
    ('"cross_entropy"', '"hinge"', "training.loss must be one of"),
    ("epochs = 2000", "epochs = 0", "training.epochs must be a positive integer"),
    ("epochs = 2000", 'epochs = "2000"', "training.epochs must be a positive integer"),
    ("batch_size = 4\n", "", "training: missing key 'batch_size'"),
    ("batch_size = 4", 'batch_size = 4\nshuffle = "yes"', "training.shuffle must be true or false"),
    ("batch_size = 4", "batch_size = 4\nseed = -3", "training.seed must be a non-negative integer"),
    ("batch_size = 4", "batch_size = 4\nepoch = 3", "training: unknown key 'epoch'"),
    ('name = "sgd"\n', "", "optimizer: missing key 'name'"),
    ('"sgd"', '"lbfgs"', "optimizer.name must be one of"),
    ("learning_rate = 0.5", "learning_rat = 0.5", "optimizer: unknown key 'learning_rat'"),
    ("learning_rate = 0.5", "learning_rate = 0.5\nbeta1 = 0.9", "optimizer: unknown key 'beta1'"),
    ("learning_rate = 0.5", 'learning_rate = "abc"', "optimizer.learning_rate must be a finite"),
    ("learning_rate = 0.5", "learning_rate = true", "optimizer.learning_rate must be a finite"),
    ("learning_rate = 0.5", "learning_rate = nan", "optimizer.learning_rate must be a finite"),
    ("learning_rate = 0.5", "learning_rate = inf", "optimizer.learning_rate must be a finite"),
    ("learning_rate = 0.5", "learning_rate = -0.5", "learning rate must be positive"),
    ('name = "sgd"\nlearning_rate = 0.5', 'name = "momentum"\nnesterov = 1',
     "optimizer.nesterov must be true or false"),
    ('name = "sgd"\nlearning_rate = 0.5', 'name = "momentum"\nmomentum = 1',
     "momentum must be in [0, 1["),
    ('{ size = 2, activation = "identity" }', '{ size = 2, activation = "softmax" }',
     "the last layer must use the identity activation"),
    ('{ size = 2, activation = "identity" }', '{ size = 2, activation = "sigmoid" }',
     "the last layer must use the identity activation"),
])
def test_invalid_configurations_raise_a_clear_error(tmp_path, old, new, message):
    path = config_with(tmp_path, old, new)
    with pytest.raises(MyTorchError, match="invalid configuration") as error:
        load_config(path)
    assert message in str(error.value)
    assert str(path) in str(error.value)


def test_section_that_is_not_a_table_raises_an_error(tmp_path):
    text = "network = 3\n" + CONFIG[CONFIG.index("[training]"):]
    with pytest.raises(MyTorchError, match="network must be a table"):
        load_config(write_config(tmp_path, text))


def test_missing_file_raises_an_error(tmp_path):
    with pytest.raises(MyTorchError, match="no such file"):
        load_config(tmp_path / "missing.conf")


def test_directory_raises_an_error(tmp_path):
    with pytest.raises(MyTorchError, match="cannot read configuration"):
        load_config(tmp_path)


def test_binary_file_raises_an_error(tmp_path):
    path = tmp_path / "network.conf"
    path.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(MyTorchError, match="invalid configuration"):
        load_config(path)


def test_network_follows_the_configuration(tmp_path):
    network = load_config(write_config(tmp_path, CONFIG)).create_network()
    expected = Network(2, [(4, "tanh"), (2, "identity")], "xavier_normal", seed=0)
    assert network.architecture == expected.architecture
    for layer, expected_layer in zip(network.layers, expected.layers):
        assert np.array_equal(layer.weights, expected_layer.weights)


def test_another_seed_gives_another_network(tmp_path):
    config = load_config(write_config(tmp_path, CONFIG))
    first, second = config.create_network(seed=5), config.create_network(seed=6)
    assert not np.array_equal(first.layers[0].weights, second.layers[0].weights)
    assert not np.array_equal(first.layers[0].weights, config.create_network().layers[0].weights)
    assert np.array_equal(first.layers[0].weights, config.create_network(seed=5).layers[0].weights)


def test_each_call_creates_a_new_optimizer(tmp_path):
    config = load_config(write_config(tmp_path, CONFIG))
    assert config.create_optimizer() is not config.create_optimizer()
    assert loss_name(config.create_loss()) == "cross_entropy"


def test_training_settings_are_saved_and_checked_back(tmp_path):
    config = load_config(config_with(tmp_path, "batch_size = 4", "batch_size = 4\nseed = 1"))
    settings = config.training_settings()
    assert settings == {"epochs": 2000, "batch_size": 4, "shuffle": True, "seed": 1,
                        "l2": 0.0, "patience": None, "monitor": "val_loss", "validation_split": 0.0}
    save_network(tmp_path / "network.nn", config.create_network(), config.create_loss(),
                 config.create_optimizer(), settings)
    assert check_training_settings(load_network(tmp_path / "network.nn").settings) == settings


@pytest.mark.parametrize("settings, message", [
    ({"epochs": 0, "batch_size": 4}, "settings.epochs"),
    ({"epochs": 3}, "missing key 'batch_size'"),
    ({"epochs": 3, "batch_size": 4, "loss": "mse"}, "unknown key 'loss'"),
    ([1], "settings must be a table"),
    ({"epochs": 3, "batch_size": 4, "shuffle": 1}, "settings.shuffle"),
])
def test_invalid_training_settings_raise_an_error(settings, message):
    with pytest.raises(MyTorchError, match=message):
        check_training_settings(settings)


def test_training_settings_get_their_defaults():
    assert check_training_settings({"epochs": 3, "batch_size": 4}) == {
        "epochs": 3, "batch_size": 4, "shuffle": True, "seed": None,
        "l2": 0.0, "patience": None, "monitor": "val_loss", "validation_split": 0.0,
    }


@pytest.mark.parametrize("path", ["configs/xor.conf", "configs/basic_network.conf"])
def test_example_configurations_give_valid_networks(path):
    config = load_config(path)
    network = config.create_network()
    inputs = np.zeros((3, config.input_size))
    assert network.forward(inputs).shape == (3, config.layers[-1][0])
    config.create_optimizer()


def test_xor_example_learns_xor():
    config = load_config("configs/xor.conf")
    network = config.create_network()
    history = train(
        network, config.create_loss(), config.create_optimizer(), XOR_INPUTS, XOR_TARGETS,
        config.epochs, config.batch_size, config.shuffle, config.training_seed,
    )
    assert history["loss"][-1] < 0.01
    assert network.predict(XOR_INPUTS).tolist() == XOR_LABELS.tolist()
