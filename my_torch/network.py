"""Neural network assembled from dense layers and activations."""

from my_torch.activations import get_activation
from my_torch.errors import MyTorchError
from my_torch.layers import Dense


class Network:
    """Stack of dense layers, each followed by its activation."""

    def __init__(self, input_size: int, architecture: list[tuple[int, str]]) -> None:
        if not architecture:
            raise MyTorchError("a network needs at least one layer")
        self.input_size = input_size
        self.architecture = list(architecture)
        self.layers = []
        self.activations = []
        size = input_size
        for output_size, activation_name in architecture:
            self.layers.append(Dense(size, output_size))
            self.activations.append(get_activation(activation_name))
            size = output_size
