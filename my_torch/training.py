"""Training loop of a network over epochs and mini-batches."""

import numpy as np

from my_torch.errors import MyTorchError
from my_torch.losses import Loss
from my_torch.network import Network
from my_torch.optimizers import Optimizer


def check_dataset(name: str, inputs: np.ndarray, targets: np.ndarray) -> None:
    if inputs.ndim != 2 or targets.ndim != 2:
        raise MyTorchError(
            f"{name} inputs and targets must be 2D arrays, got {inputs.ndim}D and {targets.ndim}D"
        )
    if len(inputs) != len(targets):
        raise MyTorchError(
            f"{name} inputs and targets must have the same number of examples, "
            f"got {len(inputs)} and {len(targets)}"
        )
    if len(inputs) == 0:
        raise MyTorchError(f"{name} dataset is empty")


def check_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise MyTorchError(f"{name} must be a positive integer, got {value!r}")


def train(
    network: Network,
    loss: Loss,
    optimizer: Optimizer,
    inputs: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> None:
    """Train a network on a dataset, with one optimizer step per batch."""
    inputs, targets = np.asarray(inputs), np.asarray(targets)
    check_dataset("training", inputs, targets)
    check_count("epochs", epochs)
    check_count("batch_size", batch_size)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(inputs)) if shuffle else np.arange(len(inputs))
        for start in range(0, len(inputs), batch_size):
            batch = order[start:start + batch_size]
            outputs = network.forward(inputs[batch])
            network.backward(loss.backward(outputs, targets[batch]))
            optimizer.step(network.layers)
