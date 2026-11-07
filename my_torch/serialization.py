"""Saving and loading of networks in .nn files."""

import json
import os
import tempfile
from typing import Any

import numpy as np

from my_torch.errors import MyTorchError
from my_torch.losses import Loss, loss_name
from my_torch.network import Network
from my_torch.optimizers import Optimizer, optimizer_name

FORMAT_NAME = "my_torch"
FORMAT_VERSION = 1
METADATA_KEY = "metadata"
OPTIMIZER_PREFIX = "optimizer/"


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
