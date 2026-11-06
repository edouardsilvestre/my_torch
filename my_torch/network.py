"""Neural network assembled from dense layers and activations."""

import numpy as np

from my_torch.activations import get_activation
from my_torch.errors import MyTorchError
from my_torch.initializers import get_initializer
from my_torch.layers import Dense


class Network:
    """Stack of dense layers, each followed by its activation."""

    def __init__(
        self,
        input_size: int,
        architecture: list[tuple[int, str]],
        initializer: str = "he",
        seed: int | None = None,
    ) -> None:
        if not architecture:
            raise MyTorchError("a network needs at least one layer")
        initialize = get_initializer(initializer)
        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.architecture = list(architecture)
        self.layers = []
        self.activations = []
        size = input_size
        for output_size, activation_name in architecture:
            layer = Dense(size, output_size)
            layer.weights = initialize(layer.weights.shape, rng)
            self.layers.append(layer)
            self.activations.append(get_activation(activation_name))
            size = output_size

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute the outputs of the last layer for a batch of inputs."""
        outputs = inputs
        for layer, activation in zip(self.layers, self.activations):
            outputs = activation.forward(layer.forward(outputs))
        return outputs

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return the index of the highest score of each example of the batch."""
        return np.argmax(self.forward(inputs), axis=1)

    def backward(self, gradient: np.ndarray) -> None:
        """Propagate the gradient of the loss back through every layer."""
        last_outputs = self.layers[-1].outputs
        if last_outputs is None:
            raise MyTorchError("forward must be called before backward")
        if gradient.shape != last_outputs.shape:
            raise MyTorchError(
                f"expected a gradient of shape {last_outputs.shape}, got {gradient.shape}"
            )
        for layer, activation in zip(reversed(self.layers), reversed(self.activations)):
            gradient = layer.backward(activation.backward(layer.outputs, gradient))
