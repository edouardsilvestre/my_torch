"""Activation functions and their backward pass."""

import numpy as np

MAX_EXPONENT = 500.0


class Activation:
    """Non-linear function applied to the outputs of a layer."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, z: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Propagate the gradient of the outputs back to the inputs z."""
        return gradient * self.derivative(z)


class Identity(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return z

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)


class Sigmoid(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -MAX_EXPONENT, MAX_EXPONENT)))

    def derivative(self, z: np.ndarray) -> np.ndarray:
        outputs = self.forward(z)
        return outputs * (1.0 - outputs)


class Tanh(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(z) ** 2
