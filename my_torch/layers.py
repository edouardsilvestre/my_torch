"""Layers of a neural network."""

import numpy as np

from my_torch.errors import MyTorchError


class Dense:
    """Fully connected layer computing Z = X @ W + b."""

    def __init__(self, input_size: int, output_size: int) -> None:
        if input_size <= 0 or output_size <= 0:
            raise MyTorchError(
                f"layer sizes must be positive, got {input_size} and {output_size}"
            )
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros((input_size, output_size))
        self.biases = np.zeros(output_size)
        self.inputs = None
        self.outputs = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute the outputs of a batch of shape (batch_size, input_size)."""
        if inputs.ndim != 2 or inputs.shape[1] != self.input_size:
            raise MyTorchError(
                f"expected a batch of shape (batch_size, {self.input_size}), got {inputs.shape}"
            )
        self.inputs = inputs
        self.outputs = inputs @ self.weights + self.biases
        return self.outputs
