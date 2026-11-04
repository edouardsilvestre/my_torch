"""Loss functions and their gradients."""

import numpy as np

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
