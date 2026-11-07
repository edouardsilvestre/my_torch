"""Tests of my_torch_generator, called through its main function."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from my_torch import load_config, load_network
from my_torch.generator import USAGE, main, network_seed, output_name, parse_arguments

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run in an empty directory holding a copy of both example configurations."""
    monkeypatch.chdir(tmp_path)
    for name in ("basic_network.conf", "xor.conf"):
        shutil.copy(CONFIGS / name, tmp_path / name)
    return tmp_path


def files(directory):
    return sorted(os.listdir(directory))


def test_help_prints_the_usage(capsys):
    assert main(["--help"]) == 0
    captured = capsys.readouterr()
    assert captured.out == USAGE + "\n"
    assert captured.err == ""


def test_usage_is_the_one_of_the_subject():
    assert USAGE.splitlines()[:2] == [
        "USAGE",
        "        ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]",
    ]
    assert "config_file_i" in USAGE and "nb_i" in USAGE


def test_example_of_the_subject(workdir, capsys):
    assert main(["basic_network.conf", "3"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert files(workdir) == [
        "basic_network.conf", "basic_network_1.nn", "basic_network_2.nn",
        "basic_network_3.nn", "xor.conf",
    ]


def test_generated_networks_follow_the_configuration(workdir):
    main(["basic_network.conf", "2"])
    config = load_config("basic_network.conf")
    for index in (1, 2):
        saved = load_network(f"basic_network_{index}.nn")
        assert saved.network.input_size == config.input_size
        assert saved.network.architecture == config.layers
        assert type(saved.loss) is type(config.create_loss())
        assert saved.optimizer.hyperparameters() == config.create_optimizer().hyperparameters()
        assert saved.optimizer.state() == config.create_optimizer().state()
        assert saved.settings == config.training_settings()


def test_networks_without_seed_all_differ(workdir):
    main(["basic_network.conf", "3"])
    weights = [load_network(f"basic_network_{i}.nn").network.layers[0].weights for i in (1, 2, 3)]
    for first in range(3):
        for second in range(first + 1, 3):
            assert not np.array_equal(weights[first], weights[second])


def test_seeded_networks_differ_and_are_reproducible(workdir):
    config = load_config("xor.conf")
    main(["xor.conf", "3"])
    first_run = [Path(f"xor_{i}.nn").read_bytes() for i in (1, 2, 3)]
    main(["xor.conf", "3"])
    assert [Path(f"xor_{i}.nn").read_bytes() for i in (1, 2, 3)] == first_run
    for index in (1, 2, 3):
        expected = config.create_network(config.seed + index - 1)
        loaded = load_network(f"xor_{index}.nn").network
        assert np.array_equal(loaded.layers[0].weights, expected.layers[0].weights)
    assert not np.array_equal(
        load_network("xor_1.nn").network.layers[0].weights,
        load_network("xor_2.nn").network.layers[0].weights,
    )


def test_first_network_is_the_one_of_the_configuration(workdir):
    main(["xor.conf", "1"])
    expected = load_config("xor.conf").create_network()
    assert np.array_equal(load_network("xor_1.nn").network.layers[0].weights,
                          expected.layers[0].weights)


def test_networks_are_written_in_the_current_directory(workdir):
    other = workdir / "configs"
    other.mkdir()
    shutil.move("xor.conf", other / "xor.conf")
    assert main(["configs/xor.conf", "1"]) == 0
    assert (workdir / "xor_1.nn").exists()
    assert not (other / "xor_1.nn").exists()


def test_several_configurations_at_once(workdir):
    assert main(["xor.conf", "1", "basic_network.conf", "2"]) == 0
    assert {"xor_1.nn", "basic_network_1.nn", "basic_network_2.nn"} <= set(files(workdir))


@pytest.mark.parametrize("path, index, expected", [
    ("basic_network.conf", 1, "basic_network_1.nn"),
    ("configs/xor.conf", 12, "xor_12.nn"),
    ("network", 2, "network_2.nn"),
    ("net.v2.conf", 3, "net.v2_3.nn"),
])
def test_output_name(path, index, expected):
    assert output_name(path, index) == expected


@pytest.mark.parametrize("seed, index, expected", [(None, 1, None), (None, 3, None),
                                                    (7, 1, 7), (7, 3, 9)])
def test_network_seed(workdir, seed, index, expected):
    config = load_config("xor.conf")
    config.seed = seed
    assert network_seed(config, index) == expected


def test_pairs_are_parsed():
    assert parse_arguments(["a.conf", "3", "b.conf", "12", "c.conf", "007"]) == [
        ("a.conf", 3), ("b.conf", 12), ("c.conf", 7),
    ]


@pytest.mark.parametrize("argv, message", [
    ([], "missing arguments"),
    (["basic_network.conf"], "must be followed by a number"),
    (["basic_network.conf", "3", "xor.conf"], "must be followed by a number"),
    (["basic_network.conf", "0"], "invalid number of networks '0'"),
    (["basic_network.conf", "-1"], "invalid number of networks '-1'"),
    (["basic_network.conf", "2.5"], "invalid number of networks '2.5'"),
    (["basic_network.conf", "+3"], "invalid number of networks '+3'"),
    (["basic_network.conf", "abc"], "invalid number of networks 'abc'"),
    (["basic_network.conf", ""], "invalid number of networks ''"),
    (["basic_network.conf", " 3"], "invalid number of networks ' 3'"),
    (["basic_network.conf", "٣"], "invalid number of networks"),
])
def test_invalid_arguments_return_84_with_the_usage(workdir, capsys, argv, message):
    assert main(argv) == 84
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("my_torch_generator: ")
    assert message in captured.err.splitlines()[0]
    assert captured.err.endswith(USAGE + "\n")
    assert files(workdir) == ["basic_network.conf", "xor.conf"]


def write_invalid_config(directory):
    text = (CONFIGS / "xor.conf").read_text().replace("epochs = 2000", "epochs = 0")
    (directory / "bad.conf").write_text(text)


@pytest.mark.parametrize("argv, message", [
    (["missing.conf", "2"], "no such file 'missing.conf'"),
    (["xor.conf", "2", "bad.conf", "1"], "training.epochs"),
    (["xor.conf", "1", "xor.conf", "2"], "'xor_<i>.nn'"),
    (["xor.conf", "1", "configs/xor.conf", "1"], "'xor_<i>.nn'"),
    (["--help", "3"], "no such file '--help'"),
])
def test_invalid_requests_return_84_and_write_nothing(workdir, capsys, argv, message):
    write_invalid_config(workdir)
    (workdir / "configs").mkdir()
    shutil.copy(workdir / "xor.conf", workdir / "configs" / "xor.conf")
    before = files(workdir)
    assert main(argv) == 84
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("my_torch_generator: ")
    assert message in captured.err
    assert USAGE not in captured.err
    assert files(workdir) == before


def test_unwritable_directory_returns_84(workdir, capsys):
    os.chmod(workdir, 0o500)
    try:
        assert main(["xor.conf", "1"]) == 84
    finally:
        os.chmod(workdir, 0o700)
    assert "cannot save network" in capsys.readouterr().err


def test_unexpected_error_returns_84(monkeypatch, capsys):
    def failing(argv):
        raise ZeroDivisionError("division by zero")

    monkeypatch.setattr("my_torch.generator.run", failing)
    assert main(["xor.conf", "1"]) == 84
    assert capsys.readouterr().err == "my_torch_generator: unexpected error: division by zero\n"
