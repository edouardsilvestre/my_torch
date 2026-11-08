"""Neural network assembled from dense layers and activations."""

import numpy as np

from my_torch.activations import get_activation
from my_torch.errors import MyTorchError
from my_torch.initializers import get_initializer
from my_torch.layers import Dense


def check_dropout(dropout: list[float], layer_count: int) -> list[float]:
    """Check the dropout rate of each layer: from 0 included to 1 excluded, 0 for the last."""
    dropout = list(dropout)
    if len(dropout) != layer_count:
        raise MyTorchError(f"expected {layer_count} dropout rates, one per layer, "
                           f"got {len(dropout)}")
    for index, rate in enumerate(dropout):
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 <= rate < 1:
            raise MyTorchError(
                f"dropout rate of layer {index} must be at least 0 and below 1, got {rate!r}"
            )
    if dropout[-1]:
        raise MyTorchError("the last layer cannot have dropout: its outputs are the scores")
    return [float(rate) for rate in dropout]


class Network:
    """Stack of dense layers, each followed by its activation, and optionally by dropout."""

    def __init__(
        self,
        input_size: int,
        architecture: list[tuple[int, str]],
        initializer: str = "he",
        seed: int | None = None,
        dropout: list[float] | None = None,
    ) -> None:
        if not architecture:
            raise MyTorchError("a network needs at least one layer")
        self.dropout = check_dropout(
            [0.0] * len(architecture) if dropout is None else dropout, len(architecture)
        )
        self.masks: list[np.ndarray | None] = [None] * len(architecture)
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

    def forward(
        self, inputs: np.ndarray, dropout_rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Compute the outputs of the last layer for a batch of inputs.

        Dropout is only active with a random generator, as during training: each output of a
        layer with a rate p is then set to 0 with probability p, and the others are divided
        by 1 - p, so that the expected value of each output does not change (inverted
        dropout). Without a generator, the network is deterministic and uses every neuron.
        """
        outputs = inputs
        for index, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            outputs = activation.forward(layer.forward(outputs))
            rate = self.dropout[index]
            if dropout_rng is not None and rate:
                mask = (dropout_rng.random(outputs.shape) >= rate) / (1 - rate)
                outputs = outputs * mask
                self.masks[index] = mask
            else:
                self.masks[index] = None
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
        for layer, activation, mask in zip(
            reversed(self.layers), reversed(self.activations), reversed(self.masks)
        ):
            if mask is not None:
                gradient = gradient * mask
            gradient = layer.backward(activation.backward(layer.outputs, gradient))
