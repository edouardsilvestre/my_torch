"""Saving and loading of networks in .nn files."""

import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Any

import numpy as np

from my_torch.errors import MyTorchError
from my_torch.losses import Loss, get_loss, loss_name
from my_torch.network import Network
from my_torch.optimizers import Optimizer, get_optimizer, optimizer_name

FORMAT_NAME = "my_torch"
FORMAT_VERSION = 1
METADATA_KEY = "metadata"
OPTIMIZER_PREFIX = "optimizer/"
METADATA_FIELDS = {
    "format", "version", "input_size", "architecture", "loss", "optimizer", "settings",
}
# Added after the first files were written: a file without it has no dropout.
OPTIONAL_METADATA_FIELDS = {"dropout"}


@dataclass
class SavedNetwork:
    """A network loaded from a file, with what was saved to resume its training."""

    network: Network
    loss: Loss | None = None
    optimizer: Optimizer | None = None
    settings: dict[str, Any] | None = None


def layer_key(index: int, parameter: str) -> str:
    return f"layers/{index}/{parameter}"


def current_umask() -> int:
    umask = os.umask(0)
    os.umask(umask)
    return umask


def save_network(
    path: str,
    network: Network,
    loss: Loss | None = None,
    optimizer: Optimizer | None = None,
    settings: dict[str, Any] | None = None,
) -> None:
    """Save a network, and optionally what is needed to resume its training, in a file."""
    if settings is not None and not isinstance(settings, dict):
        raise MyTorchError("settings must be a dictionary")
    metadata = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "input_size": network.input_size,
        "architecture": [[size, activation] for size, activation in network.architecture],
        "dropout": network.dropout,
        "loss": None if loss is None else loss_name(loss),
        "optimizer": None if optimizer is None else {
            "name": optimizer_name(optimizer),
            "hyperparameters": optimizer.hyperparameters(),
        },
        "settings": settings,
    }
    try:
        text = json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise MyTorchError(f"cannot save network: {error}") from error
    arrays = {}
    for index, layer in enumerate(network.layers):
        arrays[layer_key(index, "weights")] = layer.weights
        arrays[layer_key(index, "biases")] = layer.biases
    if optimizer is not None:
        for key, array in optimizer.state().items():
            arrays[OPTIMIZER_PREFIX + key] = array
    for key, array in arrays.items():
        if not np.all(np.isfinite(array)):
            raise MyTorchError(f"cannot save network: '{key}' holds non-finite values")
    write_archive(path, {METADATA_KEY: np.array(text), **arrays})


def write_archive(path: str, arrays: dict[str, np.ndarray]) -> None:
    """Write the arrays next to the file, then replace it, so that it is never half written."""
    directory = os.path.dirname(os.path.abspath(path))
    try:
        descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".", suffix=".tmp")
    except OSError as error:
        raise MyTorchError(f"cannot save network to '{path}': {error.strerror}") from error
    try:
        with os.fdopen(descriptor, "wb") as file:
            np.savez(file, **arrays)
        os.chmod(temporary, 0o666 & ~current_umask())
        os.replace(temporary, path)
    except OSError as error:
        os.unlink(temporary)
        raise MyTorchError(f"cannot save network to '{path}': {error.strerror}") from error
    except BaseException:
        os.unlink(temporary)
        raise


def load_network(path: str) -> SavedNetwork:
    """Load a network from a file, after checking that the file is consistent."""
    arrays = read_archive(path)
    try:
        metadata = read_metadata(arrays.pop(METADATA_KEY, None))
        architecture = [(size, activation) for size, activation in metadata["architecture"]]
        network = Network(metadata["input_size"], architecture, dropout=metadata.get("dropout"))
        read_layers(network, arrays)
        loss = None if metadata["loss"] is None else get_loss(metadata["loss"])
        optimizer = read_optimizer(metadata["optimizer"], network, arrays)
        if arrays:
            raise MyTorchError(f"unexpected arrays: {', '.join(sorted(arrays))}")
    except MyTorchError as error:
        raise MyTorchError(f"invalid network file '{path}': {error}") from error
    return SavedNetwork(network, loss, optimizer, metadata["settings"])


def read_archive(path: str) -> dict[str, np.ndarray]:
    """Read every array of a .nn file, without ever running code stored in it."""
    try:
        archive = np.load(path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("not an archive")
        with archive:
            return {name: archive[name] for name in archive.files}
    except FileNotFoundError as error:
        raise MyTorchError(f"cannot load network: no such file '{path}'") from error
    except (IsADirectoryError, PermissionError) as error:
        raise MyTorchError(f"cannot load network from '{path}': {error.strerror}") from error
    except (OSError, ValueError, EOFError, SyntaxError, zipfile.BadZipFile) as error:
        raise MyTorchError(f"invalid network file '{path}': not a my_torch network") from error


def is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def reject_constant(name: str) -> None:
    raise ValueError(f"{name} is not a valid number")


def read_metadata(array: np.ndarray | None) -> dict[str, Any]:
    """Decode and check the description of the network stored in the file."""
    if array is None or array.shape != () or array.dtype.kind != "U":
        raise MyTorchError("missing description of the network")
    try:
        metadata = json.loads(str(array), parse_constant=reject_constant)
    except ValueError as error:
        raise MyTorchError("unreadable description of the network") from error
    if not isinstance(metadata, dict) or metadata.get("format") != FORMAT_NAME:
        raise MyTorchError("not a my_torch network")
    if metadata.get("version") != FORMAT_VERSION:
        raise MyTorchError(
            f"unsupported format version {metadata.get('version')!r}, expected {FORMAT_VERSION}"
        )
    if not METADATA_FIELDS <= set(metadata) <= METADATA_FIELDS | OPTIONAL_METADATA_FIELDS:
        fields = ", ".join(sorted(METADATA_FIELDS))
        optional = ", ".join(sorted(OPTIONAL_METADATA_FIELDS))
        raise MyTorchError(f"description must hold exactly: {fields}, and optionally {optional}")
    if "dropout" in metadata and not isinstance(metadata["dropout"], list):
        raise MyTorchError("dropout must be a list of rates, one per layer")
    if not is_count(metadata["input_size"]):
        raise MyTorchError("input size must be a positive integer")
    architecture = metadata["architecture"]
    if not isinstance(architecture, list) or not all(
        isinstance(layer, list) and len(layer) == 2
        and is_count(layer[0]) and isinstance(layer[1], str)
        for layer in architecture
    ):
        raise MyTorchError("architecture must be a list of [size, activation] pairs")
    if metadata["loss"] is not None and not isinstance(metadata["loss"], str):
        raise MyTorchError("loss must be a name")
    optimizer = metadata["optimizer"]
    if optimizer is not None and not (
        isinstance(optimizer, dict)
        and set(optimizer) == {"name", "hyperparameters"}
        and isinstance(optimizer["name"], str)
        and isinstance(optimizer["hyperparameters"], dict)
    ):
        raise MyTorchError("optimizer must hold a name and hyperparameters")
    if metadata["settings"] is not None and not isinstance(metadata["settings"], dict):
        raise MyTorchError("settings must be a dictionary")
    return metadata


def read_layers(network: Network, arrays: dict[str, np.ndarray]) -> None:
    """Move the weights and biases of the file into the layers of the network."""
    for index, layer in enumerate(network.layers):
        for parameter in ("weights", "biases"):
            key = layer_key(index, parameter)
            array = arrays.pop(key, None)
            expected_shape = getattr(layer, parameter).shape
            if array is None or array.dtype.kind != "f" or array.shape != expected_shape:
                raise MyTorchError(f"'{key}' must be a float array of shape {expected_shape}")
            if not np.all(np.isfinite(array)):
                raise MyTorchError(f"'{key}' holds non-finite values")
            setattr(layer, parameter, array.astype(np.float64))
    extra = sorted(key for key in arrays if key.startswith("layers/"))
    if extra:
        raise MyTorchError(f"arrays without a matching layer: {', '.join(extra)}")


def read_optimizer(
    description: dict[str, Any] | None, network: Network, arrays: dict[str, np.ndarray]
) -> Optimizer | None:
    """Create the saved optimizer again, and restore its memory."""
    state = {
        key[len(OPTIMIZER_PREFIX):]: arrays.pop(key)
        for key in list(arrays)
        if key.startswith(OPTIMIZER_PREFIX)
    }
    if description is None:
        if state:
            raise MyTorchError("optimizer state without optimizer")
        return None
    optimizer = get_optimizer(description["name"], **description["hyperparameters"])
    optimizer.load_state(state, network.layers)
    return optimizer
