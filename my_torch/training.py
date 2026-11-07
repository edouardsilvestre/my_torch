"""Training loop of a network over epochs and mini-batches."""

import numpy as np

from my_torch.errors import MyTorchError
from my_torch.losses import Loss
from my_torch.network import Network
from my_torch.optimizers import Optimizer

EVALUATION_BATCH_SIZE = 1024


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


def is_one_hot(targets: np.ndarray) -> bool:
    """Tell whether each row of the targets holds a single 1 and zeros elsewhere."""
    return (
        targets.shape[1] > 1
        and bool(np.all((targets == 0) | (targets == 1)))
        and bool(np.all(targets.sum(axis=1) == 1))
    )


def evaluate(
    network: Network, loss: Loss, inputs: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    """Return the loss on a whole dataset, and the accuracy for one-hot targets."""
    inputs, targets = np.asarray(inputs), np.asarray(targets)
    check_dataset("evaluation", inputs, targets)
    total_loss = 0.0
    correct = 0
    for start in range(0, len(inputs), EVALUATION_BATCH_SIZE):
        batch_targets = targets[start:start + EVALUATION_BATCH_SIZE]
        outputs = network.forward(inputs[start:start + EVALUATION_BATCH_SIZE])
        total_loss += loss.forward(outputs, batch_targets) * len(batch_targets)
        correct += int(np.sum(np.argmax(outputs, axis=1) == np.argmax(batch_targets, axis=1)))
    metrics = {"loss": total_loss / len(inputs)}
    if is_one_hot(targets):
        metrics["accuracy"] = correct / len(inputs)
    return metrics


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
    validation: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, list[float]]:
    """Train a network on a dataset, with one optimizer step per batch.

    Return the history of the metrics measured at the end of each epoch.
    """
    inputs, targets = np.asarray(inputs), np.asarray(targets)
    check_dataset("training", inputs, targets)
    if validation is not None:
        validation = (np.asarray(validation[0]), np.asarray(validation[1]))
        check_dataset("validation", *validation)
    check_count("epochs", epochs)
    check_count("batch_size", batch_size)
    rng = np.random.default_rng(seed)
    history: dict[str, list[float]] = {}
    for _ in range(epochs):
        order = rng.permutation(len(inputs)) if shuffle else np.arange(len(inputs))
        for start in range(0, len(inputs), batch_size):
            batch = order[start:start + batch_size]
            outputs = network.forward(inputs[batch])
            network.backward(loss.backward(outputs, targets[batch]))
            optimizer.step(network.layers)
        metrics = evaluate(network, loss, inputs, targets)
        if validation is not None:
            for name, value in evaluate(network, loss, *validation).items():
                metrics[f"val_{name}"] = value
        for name, value in metrics.items():
            history.setdefault(name, []).append(value)
    return history
