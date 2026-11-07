"""Configuration files describing a network and how to train it."""

import inspect
import math
import tomllib
from dataclasses import dataclass
from typing import Any

from my_torch.activations import ACTIVATIONS
from my_torch.errors import MyTorchError
from my_torch.initializers import INITIALIZERS
from my_torch.losses import LOSSES
from my_torch.optimizers import OPTIMIZERS, get_optimizer

DEFAULT_INITIALIZER = "he"
DEFAULT_SHUFFLE = True


@dataclass
class Config:
    """Validated content of a configuration file."""

    input_size: int
    layers: list[tuple[int, str]]
    initializer: str
    seed: int | None
    loss: str
    optimizer: str
    optimizer_params: dict[str, float | bool]
    epochs: int
    batch_size: int
    shuffle: bool
    training_seed: int | None


def check_keys(table: Any, where: str, required: set[str], optional: set[str]) -> None:
    if not isinstance(table, dict):
        raise MyTorchError(f"{where} must be a table")
    missing = sorted(required - set(table))
    if missing:
        raise MyTorchError(f"{where}: missing key '{missing[0]}'")
    unknown = sorted(set(table) - required - optional)
    if unknown:
        raise MyTorchError(f"{where}: unknown key '{unknown[0]}'")


def read_count(value: Any, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise MyTorchError(f"{where} must be a positive integer, got {value!r}")
    return value


def read_seed(value: Any, where: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MyTorchError(f"{where} must be a non-negative integer, got {value!r}")
    return value


def read_name(value: Any, where: str, choices: dict[str, Any]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise MyTorchError(
            f"{where} must be one of: {', '.join(choices)}, got {value!r}"
        )
    return value


def read_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise MyTorchError(f"{where} must be true or false, got {value!r}")
    return value


def read_number(value: Any, where: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise MyTorchError(f"{where} must be a finite number, got {value!r}")
    return float(value)


def optimizer_parameters(name: str) -> dict[str, type]:
    """Return the parameters accepted by an optimizer, with the type of their default."""
    signature = inspect.signature(OPTIMIZERS[name].__init__)
    return {
        parameter.name: type(parameter.default)
        for parameter in signature.parameters.values()
        if parameter.name != "self"
    }


def read_network(table: Any) -> tuple[int, list[tuple[int, str]], str, int | None]:
    check_keys(table, "network", {"input_size", "layers"}, {"initializer", "seed"})
    input_size = read_count(table["input_size"], "network.input_size")
    layers = table["layers"]
    if not isinstance(layers, list) or not layers:
        raise MyTorchError("network.layers must be a non-empty list of layers")
    architecture = []
    for index, layer in enumerate(layers):
        where = f"network.layers[{index}]"
        check_keys(layer, where, {"size", "activation"}, set())
        architecture.append((
            read_count(layer["size"], f"{where}.size"),
            read_name(layer["activation"], f"{where}.activation", ACTIVATIONS),
        ))
    initializer = read_name(
        table.get("initializer", DEFAULT_INITIALIZER), "network.initializer", INITIALIZERS
    )
    return input_size, architecture, initializer, read_seed(table.get("seed"), "network.seed")


def read_training(table: Any, where: str = "training", with_loss: bool = True) -> dict[str, Any]:
    required = {"epochs", "batch_size"} | ({"loss"} if with_loss else set())
    check_keys(table, where, required, {"shuffle", "seed"})
    training = {
        "epochs": read_count(table["epochs"], f"{where}.epochs"),
        "batch_size": read_count(table["batch_size"], f"{where}.batch_size"),
        "shuffle": read_bool(table.get("shuffle", DEFAULT_SHUFFLE), f"{where}.shuffle"),
        "seed": read_seed(table.get("seed"), f"{where}.seed"),
    }
    if with_loss:
        training["loss"] = read_name(table["loss"], f"{where}.loss", LOSSES)
    return training


def read_optimizer(table: Any) -> tuple[str, dict[str, float | bool]]:
    if not isinstance(table, dict):
        raise MyTorchError("optimizer must be a table")
    if "name" not in table:
        raise MyTorchError("optimizer: missing key 'name'")
    name = read_name(table["name"], "optimizer.name", OPTIMIZERS)
    accepted = optimizer_parameters(name)
    check_keys(table, "optimizer", {"name"}, set(accepted))
    params = {}
    for key, value in table.items():
        if key == "name":
            continue
        where = f"optimizer.{key}"
        read = read_bool if accepted[key] is bool else read_number
        params[key] = read(value, where)
    try:
        get_optimizer(name, **params)
    except MyTorchError as error:
        raise MyTorchError(f"optimizer: {error}") from error
    return name, params


def parse_config(data: Any) -> Config:
    """Check the content of a configuration and return it in a validated form."""
    check_keys(data, "configuration", {"network", "training", "optimizer"}, set())
    input_size, layers, initializer, seed = read_network(data["network"])
    training = read_training(data["training"])
    optimizer, optimizer_params = read_optimizer(data["optimizer"])
    if training["loss"] == "cross_entropy" and layers[-1][1] != "identity":
        raise MyTorchError(
            "with the cross_entropy loss, the last layer must use the identity activation, "
            f"got {layers[-1][1]!r}: the loss already applies softmax to the scores"
        )
    return Config(
        input_size, layers, initializer, seed, training["loss"], optimizer, optimizer_params,
        training["epochs"], training["batch_size"], training["shuffle"], training["seed"],
    )


def load_config(path: str) -> Config:
    """Read a TOML configuration file and check every value it holds."""
    try:
        with open(path, "rb") as file:
            data = tomllib.load(file)
    except FileNotFoundError as error:
        raise MyTorchError(f"cannot read configuration: no such file '{path}'") from error
    except OSError as error:
        raise MyTorchError(f"cannot read configuration '{path}': {error.strerror}") from error
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise MyTorchError(f"invalid configuration '{path}': {error}") from error
    try:
        return parse_config(data)
    except MyTorchError as error:
        raise MyTorchError(f"invalid configuration '{path}': {error}") from error
