"""Training loop of a network over epochs and mini-batches."""

import numpy as np

from my_torch.errors import MyTorchError
from my_torch.losses import Loss
from my_torch.network import Network
from my_torch.optimizers import Optimizer

EVALUATION_BATCH_SIZE = 1024
MONITORED = ("val_loss", "val_accuracy")


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


def check_l2(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < np.inf:
        raise MyTorchError(f"l2 must be a non-negative number, got {value!r}")


def check_early_stopping(
    patience: int | None, monitor: str, validation: tuple[np.ndarray, np.ndarray] | None
) -> None:
    if patience is None:
        return
    check_count("patience", patience)
    if validation is None:
        raise MyTorchError("early stopping needs a validation set")
    if monitor not in MONITORED:
        raise MyTorchError(f"monitor must be one of: {', '.join(MONITORED)}, got {monitor!r}")
    if monitor == "val_accuracy" and not is_one_hot(validation[1]):
        raise MyTorchError("val_accuracy needs one-hot validation targets")


def snapshot(network: Network, optimizer: Optimizer) -> tuple[list, dict[str, np.ndarray]]:
    """Copy the parameters of the network and the memory of the optimizer."""
    parameters = [(layer.weights.copy(), layer.biases.copy()) for layer in network.layers]
    state = {key: array.copy() for key, array in optimizer.state().items()}
    return parameters, state


def restore(network: Network, optimizer: Optimizer, saved: tuple[list, dict]) -> None:
    parameters, state = saved
    for layer, (weights, biases) in zip(network.layers, parameters):
        layer.weights, layer.biases = weights.copy(), biases.copy()
    optimizer.load_state(state, network.layers)


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


def format_epoch(epoch: int, epochs: int, history: dict[str, list[float]]) -> str:
    """Describe the metrics of the last epoch on a single line."""
    metrics = " - ".join(f"{name} {values[-1]:.4f}" for name, values in history.items())
    return f"epoch {epoch:>{len(str(epochs))}}/{epochs} - {metrics}"


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
    verbose: bool = False,
    l2: float = 0.0,
    patience: int | None = None,
    monitor: str = "val_loss",
) -> dict[str, list[float]]:
    """Train a network on a dataset, with one optimizer step per batch.

    With l2, the loss gets a penalty l2 / 2 times the sum of the squared weights, which
    pushes the weights towards 0: l2 times the weights is added to their gradient. The
    biases are not penalised, and the history reports the loss of the data alone.

    With patience, training stops once the monitored validation metric has not improved for
    that many epochs. The network and the optimizer then go back to the best epoch. The
    history keeps every epoch that was run.

    Dropout, when the network has some, is only active on the batches of training: the
    metrics are measured with every neuron.

    Return the history of the metrics measured at the end of each epoch.
    """
    inputs, targets = np.asarray(inputs), np.asarray(targets)
    check_dataset("training", inputs, targets)
    if validation is not None:
        validation = (np.asarray(validation[0]), np.asarray(validation[1]))
        check_dataset("validation", *validation)
    check_count("epochs", epochs)
    check_count("batch_size", batch_size)
    check_l2(l2)
    check_early_stopping(patience, monitor, validation)
    rng = np.random.default_rng(seed)
    # The masks of dropout get their own generator, so that the order of the examples
    # stays the same with or without dropout.
    dropout_rng = (
        np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
        if any(network.dropout) else None
    )
    history: dict[str, list[float]] = {}
    best_value, best_epoch, best = None, 0, None
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(inputs)) if shuffle else np.arange(len(inputs))
        for start in range(0, len(inputs), batch_size):
            batch = order[start:start + batch_size]
            outputs = network.forward(inputs[batch], dropout_rng)
            network.backward(loss.backward(outputs, targets[batch]))
            if l2:
                for layer in network.layers:
                    layer.weights_gradient += l2 * layer.weights
            optimizer.step(network.layers)
        metrics = evaluate(network, loss, inputs, targets)
        if validation is not None:
            for name, value in evaluate(network, loss, *validation).items():
                metrics[f"val_{name}"] = value
        for name, value in metrics.items():
            history.setdefault(name, []).append(value)
        if verbose:
            print(format_epoch(epoch, epochs, history), flush=True)
        if patience is None:
            continue
        value = metrics[monitor]
        if best_value is None or (
            value < best_value if monitor == "val_loss" else value > best_value
        ):
            best_value, best_epoch, best = value, epoch, snapshot(network, optimizer)
        elif epoch - best_epoch >= patience:
            break
    if best is not None:
        restore(network, optimizer, best)
        if verbose:
            print(f"early stopping: best {monitor} {best_value:.4f} at epoch {best_epoch}",
                  flush=True)
    return history
