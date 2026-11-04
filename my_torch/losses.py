"""Loss functions and their gradients."""

import numpy as np

from my_torch.activations import Softmax
from my_torch.errors import MyTorchError


def check_same_shape(outputs: np.ndarray, targets: np.ndarray) -> None:
    if outputs.shape != targets.shape:
        raise MyTorchError(
            f"outputs and targets must have the same shape, got {outputs.shape} and {targets.shape}"
        )


class Loss:
    """Measure of the error between the outputs of a network and the targets."""

    def forward(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        """Return the mean loss of the batch."""
        raise NotImplementedError

    def backward(self, outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Return the gradient of the mean loss with respect to the outputs."""
        raise NotImplementedError


class MSE(Loss):
    def forward(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        check_same_shape(outputs, targets)
        return float(np.mean((outputs - targets) ** 2))

    def backward(self, outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        check_same_shape(outputs, targets)
        return 2.0 * (outputs - targets) / outputs.size


class CrossEntropy(Loss):
    """Categorical cross-entropy computed from the raw scores of the last layer."""

    def forward(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        check_same_shape(outputs, targets)
        shifted = outputs - outputs.max(axis=1, keepdims=True)
        log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return float(-np.mean(np.sum(targets * log_probabilities, axis=1)))

    def backward(self, outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        check_same_shape(outputs, targets)
        return (Softmax().forward(outputs) - targets) / outputs.shape[0]


LOSSES = {
    "mse": MSE,
    "cross_entropy": CrossEntropy,
}


def get_loss(name: str) -> Loss:
    """Return a new loss from its name."""
    if name not in LOSSES:
        raise MyTorchError(
            f"unknown loss '{name}', expected one of: {', '.join(LOSSES)}"
        )
    return LOSSES[name]()
