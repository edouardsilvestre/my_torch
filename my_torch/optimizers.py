"""Optimizers updating the parameters of a network from their gradients."""

import numpy as np

from my_torch.errors import MyTorchError
from my_torch.layers import Dense


def check_positive(name: str, value: float) -> None:
    if not value > 0:
        raise MyTorchError(f"{name} must be positive, got {value}")


def check_decay(name: str, value: float) -> None:
    if not 0 <= value < 1:
        raise MyTorchError(f"{name} must be in [0, 1[, got {value}")


class Optimizer:
    """Update the parameters of the layers from their gradients."""

    def __init__(self, learning_rate: float = 0.01) -> None:
        check_positive("learning rate", learning_rate)
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
        check_decay("momentum", momentum)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities: dict[tuple[int, str], np.ndarray] = {}

    def compute_update(self, key: tuple[int, str], gradient: np.ndarray) -> np.ndarray:
        velocity = self.momentum * self.velocities.get(key, 0.0) + gradient
        self.velocities[key] = velocity
        if self.nesterov:
            return self.learning_rate * (gradient + self.momentum * velocity)
        return self.learning_rate * velocity


class RMSProp(Optimizer):
    """Gradient descent dividing each step by the recent size of its gradients."""

    def __init__(
        self, learning_rate: float = 0.001, decay: float = 0.9, epsilon: float = 1e-8
    ) -> None:
        super().__init__(learning_rate)
        check_decay("decay", decay)
        check_positive("epsilon", epsilon)
        self.decay = decay
        self.epsilon = epsilon
        self.squares: dict[tuple[int, str], np.ndarray] = {}

    def compute_update(self, key: tuple[int, str], gradient: np.ndarray) -> np.ndarray:
        square = self.decay * self.squares.get(key, 0.0) + (1 - self.decay) * gradient ** 2
        self.squares[key] = square
        return self.learning_rate * gradient / (np.sqrt(square) + self.epsilon)


class Adam(Optimizer):
    """Gradient descent using running means of the gradients and of their squares."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        check_decay("beta1", beta1)
        check_decay("beta2", beta2)
        check_positive("epsilon", epsilon)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.steps = 0
        self.means: dict[tuple[int, str], np.ndarray] = {}
        self.squares: dict[tuple[int, str], np.ndarray] = {}

    def step(self, layers: list[Dense]) -> None:
        self.steps += 1
        super().step(layers)

    def compute_update(self, key: tuple[int, str], gradient: np.ndarray) -> np.ndarray:
        mean = self.beta1 * self.means.get(key, 0.0) + (1 - self.beta1) * gradient
        square = self.beta2 * self.squares.get(key, 0.0) + (1 - self.beta2) * gradient ** 2
        self.means[key] = mean
        self.squares[key] = square
        corrected_mean = mean / (1 - self.beta1 ** self.steps)
        corrected_square = square / (1 - self.beta2 ** self.steps)
        return self.learning_rate * corrected_mean / (np.sqrt(corrected_square) + self.epsilon)
