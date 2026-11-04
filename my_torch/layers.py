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
