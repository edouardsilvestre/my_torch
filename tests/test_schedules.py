"""Tests of the learning rate schedules, of their use by train(), and of their configuration."""

import math

import numpy as np
import pytest

import analyzer.train as training
from analyzer.main import main
from my_torch import (
    MyTorchError, Network, get_loss, get_optimizer, get_schedule, load_config, load_network,
    save_network, train,
)
from my_torch.config import check_training_settings
from my_torch.schedules import SCHEDULES, Schedule, schedule_from_settings
from my_torch.training import format_epoch

XOR_INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_TARGETS = np.eye(2)[[0, 1, 1, 0]]
CONFIG = """[network]
input_size = 2
layers = [{ size = 4, activation = "tanh" }, { size = 2, activation = "identity" }]
seed = 0

[training]
loss = "cross_entropy"
epochs = 6
batch_size = 4
SCHEDULE

[optimizer]
name = "sgd"
learning_rate = 0.5
"""


def rates(schedule, epochs, history=None, base=1.0):
    return [schedule.learning_rate(base, epoch, epochs, history or {})
            for epoch in range(1, epochs + 1)]


def test_step_decay():
    assert rates(get_schedule("step", factor=0.5, every=2), 7) == [
        1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.125]


def test_exponential_decay():
    assert rates(get_schedule("exponential", factor=0.5), 4, base=0.1) == [
        0.1, 0.05, 0.025, 0.0125]


def test_cosine_annealing():
    values = rates(get_schedule("cosine", minimum=0.2), 10)
    assert values[0] == 1.0
    assert values[5] == pytest.approx(0.6)  # half way: the mean of the two bounds
    assert values[9] == pytest.approx(0.2 + 0.8 * (1 + math.cos(math.pi * 0.9)) / 2)
    assert all(later < earlier for earlier, later in zip(values, values[1:]))
    assert min(values) > 0.2


def test_cosine_minimum_above_the_learning_rate_is_refused():
    with pytest.raises(MyTorchError, match="minimum learning rate 2.0 is above"):
        get_schedule("cosine", minimum=2.0).learning_rate(1.0, 1, 5, {})


def test_plateau_lowers_the_rate_after_patience_epochs_without_progress():
    schedule = get_schedule("plateau", factor=0.5, patience=2)
    history = {"val_loss": [1.0, 0.9, 0.95, 0.97, 0.8, 0.85, 0.86, 0.87]}
    assert rates(schedule, 9, history) == [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.25, 0.25]


def test_plateau_waits_again_after_each_progress():
    schedule = get_schedule("plateau", factor=0.5, patience=2)
    history = {"val_loss": [1.0, 1.1, 0.9, 0.95, 0.8, 0.85]}
    assert rates(schedule, 7, history) == [1.0] * 7


def test_plateau_on_an_accuracy_wants_it_higher():
    schedule = get_schedule("plateau", factor=0.1, patience=1, monitor="accuracy")
    history = {"accuracy": [0.5, 0.6, 0.6, 0.7, 0.65]}
    assert rates(schedule, 6, history) == pytest.approx([1.0, 1.0, 1.0, 0.1, 0.1, 0.01])


def test_plateau_does_not_go_below_its_minimum():
    schedule = get_schedule("plateau", factor=0.1, patience=1, monitor="loss", minimum=0.05)
    history = {"loss": [1.0, 1.0, 1.0, 1.0]}
    assert rates(schedule, 5, history) == pytest.approx([1.0, 1.0, 0.1, 0.05, 0.05])


def test_plateau_does_not_raise_a_rate_already_below_its_minimum():
    schedule = get_schedule("plateau", factor=0.5, patience=1, monitor="loss", minimum=0.5)
    assert rates(schedule, 3, {"loss": [1.0, 1.0]}, base=0.1) == [0.1, 0.1, 0.1]


@pytest.mark.parametrize("name, parameters, message", [
    ("step", {"factor": 1}, "factor must be between 0 and 1 excluded"),
    ("step", {"factor": 0}, "factor must be between 0 and 1 excluded"),
    ("step", {"every": 0}, "every must be a positive integer"),
    ("step", {"every": 2.0}, "every must be a positive integer"),
    ("exponential", {"factor": True}, "factor must be between 0 and 1 excluded"),
    ("cosine", {"minimum": -0.1}, "minimum must be a non-negative number"),
    ("plateau", {"patience": 0}, "patience must be a positive integer"),
    ("plateau", {"monitor": "val"}, "monitor must be one of"),
    ("plateau", {"minimum": float("inf")}, "minimum must be a non-negative number"),
    ("warmup", {}, "unknown learning rate schedule 'warmup'"),
    ("step", {"gamma": 0.5}, "the step schedule has no parameter 'gamma'"),
])
def test_invalid_schedules_raise_an_error(name, parameters, message):
    with pytest.raises(MyTorchError, match=message):
        get_schedule(name, **parameters)


@pytest.mark.parametrize("name", list(SCHEDULES))
def test_settings_give_back_the_same_schedule(name):
    settings = get_schedule(name).settings()
    assert settings["name"] == name
    copy = schedule_from_settings(settings)
    assert type(copy) is SCHEDULES[name] and copy.settings() == settings
    history = {"val_loss": [1.0, 1.0, 1.0, 1.0, 1.0]}
    assert rates(copy, 6, history) == rates(get_schedule(name), 6, history)


def test_no_settings_give_no_schedule():
    assert schedule_from_settings(None) is None


def xor_training(schedule=None, epochs=4, **options):
    network = Network(2, [(4, "tanh"), (2, "identity")], seed=0)
    optimizer = get_optimizer("sgd", learning_rate=0.5)
    history = train(network, get_loss("cross_entropy"), optimizer, XOR_INPUTS, XOR_TARGETS,
                    epochs, 4, shuffle=False, schedule=schedule, **options)
    return network, optimizer, history


def test_the_optimizer_follows_the_schedule():
    network, _, history = xor_training(get_schedule("step", factor=0.5, every=1), epochs=3)
    assert history["learning_rate"] == [0.5, 0.25, 0.125]
    # The same steps, made by hand with the learning rate of each epoch.
    expected = Network(2, [(4, "tanh"), (2, "identity")], seed=0)
    loss = get_loss("cross_entropy")
    for rate in (0.5, 0.25, 0.125):
        train(expected, loss, get_optimizer("sgd", learning_rate=rate), XOR_INPUTS,
              XOR_TARGETS, 1, 4, shuffle=False)
    for layer, reference in zip(network.layers, expected.layers):
        assert np.array_equal(layer.weights, reference.weights)
        assert np.array_equal(layer.biases, reference.biases)


def test_the_optimizer_gets_its_learning_rate_back():
    _, optimizer, _ = xor_training(get_schedule("exponential", factor=0.5))
    assert optimizer.learning_rate == 0.5


def test_the_learning_rate_comes_back_after_an_error():
    optimizer = get_optimizer("sgd", learning_rate=0.5)
    network = Network(2, [(4, "tanh"), (2, "identity")], seed=0)

    class Failing(Schedule):
        def learning_rate(self, base, epoch, epochs, history):
            if epoch == 2:
                raise MyTorchError("stop")
            return 0.1

    with pytest.raises(MyTorchError, match="stop"):
        train(network, get_loss("cross_entropy"), optimizer, XOR_INPUTS, XOR_TARGETS, 3, 4,
              schedule=Failing())
    assert optimizer.learning_rate == 0.5


def test_without_schedule_the_history_has_no_learning_rate():
    assert "learning_rate" not in xor_training()[2]


def test_the_plateau_schedule_reads_the_validation_history():
    validation = (XOR_INPUTS, XOR_TARGETS)
    schedule = get_schedule("plateau", factor=0.5, patience=1, monitor="val_loss")
    _, _, history = xor_training(schedule, epochs=8, validation=validation)
    assert history["learning_rate"] == rates(schedule, 8, history, base=0.5)


def test_the_schedule_and_early_stopping_work_together():
    validation = (XOR_INPUTS, XOR_TARGETS)
    _, optimizer, history = xor_training(
        get_schedule("cosine"), epochs=50, validation=validation, patience=2,
        monitor="val_loss")
    assert history["learning_rate"] == rates(get_schedule("cosine"), 50, base=0.5)[
        :len(history["val_loss"])]
    assert optimizer.learning_rate == 0.5


@pytest.mark.parametrize("monitor, validation, message", [
    ("val_loss", False, "needs a validation set"),
    ("val_accuracy", False, "needs a validation set"),
    ("accuracy", True, "needs one-hot targets"),
])
def test_the_plateau_schedule_needs_its_metric(monitor, validation, message):
    network = Network(2, [(2, "identity")], seed=0)
    weights = network.layers[0].weights.copy()
    targets = XOR_TARGETS * 0.5
    with pytest.raises(MyTorchError, match=message):
        train(network, get_loss("mse"), get_optimizer("sgd"), XOR_INPUTS, targets, 2, 4,
              validation=(XOR_INPUTS, targets) if validation else None,
              schedule=get_schedule("plateau", monitor=monitor))
    assert np.array_equal(network.layers[0].weights, weights)


def test_the_learning_rate_is_shown_with_its_significant_digits():
    history = {"loss": [0.5], "learning_rate": [0.000125]}
    assert format_epoch(1, 10, history) == "epoch  1/10 - loss 0.5000 - learning_rate 0.000125"


def write_config(tmp_path, schedule):
    path = tmp_path / "network.conf"
    path.write_text(CONFIG.replace("SCHEDULE", schedule))
    return path


def test_a_schedule_is_read_from_the_configuration(tmp_path):
    config = load_config(write_config(tmp_path, 'schedule = { name = "step", every = 3 }'))
    assert config.schedule == {"name": "step", "factor": 0.5, "every": 3}
    assert config.training_settings()["schedule"] == config.schedule
    assert config.create_schedule().settings() == config.schedule


def test_a_schedule_can_be_a_sub_table(tmp_path):
    config = load_config(write_config(
        tmp_path, '[training.schedule]\nname = "plateau"\nmonitor = "loss"\npatience = 3'))
    assert config.schedule == {"name": "plateau", "factor": 0.5, "patience": 3,
                               "monitor": "loss", "minimum": 0.0}


def test_without_schedule_the_learning_rate_is_fixed(tmp_path):
    config = load_config(write_config(tmp_path, ""))
    assert config.schedule is None and config.create_schedule() is None
    assert config.training_settings()["schedule"] is None


@pytest.mark.parametrize("line, message", [
    ('schedule = "step"', "training.schedule must be a table"),
    ("schedule = { factor = 0.5 }", "training.schedule: missing key 'name'"),
    ('schedule = { name = "warmup" }', "training.schedule.name must be one of"),
    ('schedule = { name = "step", gamma = 0.5 }', "training.schedule: unknown key 'gamma'"),
    ('schedule = { name = "step", every = 2.5 }', "training.schedule.every must be a positive"),
    ('schedule = { name = "step", factor = "half" }', "training.schedule.factor must be a"),
    ('schedule = { name = "step", factor = 2 }', "training.schedule: factor must be between"),
    ('schedule = { name = "plateau", monitor = "val" }', "training.schedule.monitor must be"),
    ('schedule = { name = "cosine", minimum = -1 }', "training.schedule: minimum must be"),
])
def test_invalid_schedules_in_a_configuration(tmp_path, line, message):
    with pytest.raises(MyTorchError) as error:
        load_config(write_config(tmp_path, line))
    assert message in str(error.value)


def test_the_schedule_is_saved_with_the_network(tmp_path):
    config = load_config(write_config(
        tmp_path, 'schedule = { name = "exponential", factor = 0.8 }'))
    save_network(tmp_path / "network.nn", config.create_network(), config.create_loss(),
                 config.create_optimizer(), config.training_settings())
    settings = check_training_settings(load_network(tmp_path / "network.nn").settings)
    assert settings["schedule"] == {"name": "exponential", "factor": 0.8}


def test_old_settings_have_no_schedule():
    assert check_training_settings({"epochs": 3, "batch_size": 4})["schedule"] is None


def test_invalid_saved_schedules_are_refused():
    with pytest.raises(MyTorchError, match="settings.schedule.name must be one of"):
        check_training_settings({"epochs": 3, "batch_size": 4, "schedule": {"name": "x"}})


FENS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate Black",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3 Check White",
    "8/8/8/8/8/8/8/k1K5 w - - 0 1 Nothing",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1 Nothing",
]


def save_chess_network(path, schedule, validation_split=0.0):
    network = Network(769, [(8, "relu"), (5, "identity")], seed=0)
    settings = {"epochs": 4, "batch_size": 5, "shuffle": True, "seed": 0,
                "validation_split": validation_split, "schedule": schedule}
    save_network(path, network, get_loss("cross_entropy"),
                 get_optimizer("adam", learning_rate=0.01), settings)


def test_train_mode_follows_the_saved_schedule(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.txt").write_text("\n".join(FENS) + "\n")
    save_chess_network(tmp_path / "net.nn", {"name": "step", "factor": 0.5, "every": 2})
    calls = []
    real_train = training.train
    monkeypatch.setattr(training, "train", lambda *arguments, **options: (
        calls.append(options), real_train(*arguments, **options))[1])
    assert main(["--train", "net.nn", "train.txt"]) == 0
    assert calls[0]["schedule"].settings() == {"name": "step", "factor": 0.5, "every": 2}
    output = capsys.readouterr().out.splitlines()
    assert [line.rsplit(" ", 1)[1] for line in output] == ["0.01", "0.01", "0.005", "0.005"]
    saved = load_network("net.nn")
    assert saved.optimizer.learning_rate == 0.01
    assert saved.settings["schedule"] == {"name": "step", "factor": 0.5, "every": 2}


def test_train_mode_refuses_a_plateau_on_a_missing_validation(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.txt").write_text("\n".join(FENS) + "\n")
    save_chess_network(tmp_path / "net.nn", {"name": "plateau", "factor": 0.5, "patience": 1,
                                             "monitor": "val_loss", "minimum": 0.0})
    original = (tmp_path / "net.nn").read_bytes()
    assert main(["--train", "net.nn", "train.txt"]) == 84
    assert ("the plateau schedule on val_loss needs a validation set: set validation_split "
            "in the training settings") in capsys.readouterr().err
    assert (tmp_path / "net.nn").read_bytes() == original


def test_train_mode_runs_a_plateau_on_its_validation_share(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.txt").write_text("\n".join(FENS) + "\n")
    save_chess_network(tmp_path / "net.nn", {"name": "plateau", "factor": 0.5, "patience": 1,
                                             "monitor": "val_loss", "minimum": 0.0}, 0.4)
    assert main(["--train", "net.nn", "train.txt"]) == 0
    assert "val_loss" in capsys.readouterr().out
