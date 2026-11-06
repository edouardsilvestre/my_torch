"""Optimizers updating the parameters of a network from their gradients."""

import numpy as np

from my_torch.errors import MyTorchError
from my_torch.layers import Dense


class Optimizer:
    """Update the parameters of the layers from their gradients."""

    def __init__(self, learning_rate: float = 0.01) -> None:
        if not learning_rate > 0:
            raise MyTorchError(f"learning rate must be positive, got {learning_rate}")
        self.learning_rate = learning_rate

    def step(self, layers: list[Dense]) -> None:
        """Apply one update to the weights and biases of every layer."""
        for index, layer in enumerate(layers):
            if layer.weights_gradient is None:
                raise MyTorchError("backward must be called before step")
            layer.weights -= self.compute_update((index, "weights"), layer.weights_gradient)
            layer.biases -= self.compute_update((index, "biases"), layer.biases_gradient)

    def compute_update(self, key: tuple[int, str], gradient: np.ndarray) -> np.ndarray:
        """Return the quantity to subtract from the parameter identified by key."""
        raise NotImplementedError


class SGD(Optimizer):
    """Plain gradient descent."""

    def compute_update(self, key: tuple[int, str], gradient: np.ndarray) -> np.ndarray:
        return self.learning_rate * gradient


class Momentum(Optimizer):
    """Gradient descent with a velocity, and an optional Nesterov look-ahead."""

    def __init__(
        self, learning_rate: float = 0.01, momentum: float = 0.9, nesterov: bool = False
    ) -> None:
        super().__init__(learning_rate)
        if not 0 <= momentum < 1:
            raise MyTorchError(f"momentum must be in [0, 1[, got {momentum}")
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities: dict[tuple[int, str], np.ndarray] = {}

    def compute_update(self, key: tuple[int, str], gradient: np.ndarray) -> np.ndarray:
        velocity = self.momentum * self.velocities.get(key, 0.0) + gradient
        self.velocities[key] = velocity
        if self.nesterov:
            return self.learning_rate * (gradient + self.momentum * velocity)
        return self.learning_rate * velocity
