"""Activation functions and their backward pass."""

import numpy as np

from my_torch.errors import MyTorchError

MAX_EXPONENT = 500.0
LEAKY_RELU_SLOPE = 0.01


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


class ReLU(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)


class LeakyReLU(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, z, LEAKY_RELU_SLOPE * z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1.0, LEAKY_RELU_SLOPE)


class Softmax(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        exponentials = np.exp(z - z.max(axis=1, keepdims=True))
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def backward(self, z: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        outputs = self.forward(z)
        weighted_sum = (outputs * gradient).sum(axis=1, keepdims=True)
        return outputs * (gradient - weighted_sum)


ACTIVATIONS = {
    "identity": Identity,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "softmax": Softmax,
}


def get_activation(name: str) -> Activation:
    """Return a new activation from its name."""
    if name not in ACTIVATIONS:
        raise MyTorchError(
            f"unknown activation '{name}', expected one of: {', '.join(ACTIVATIONS)}"
        )
    return ACTIVATIONS[name]()
