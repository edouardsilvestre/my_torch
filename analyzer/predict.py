"""Prediction mode of my_torch_analyzer."""

from analyzer.encoding import INPUT_SIZE
from analyzer.errors import AnalyzerError
from analyzer.labels import get_classes
from my_torch import Network, load_network


def load_chess_network(path: str) -> tuple[Network, tuple[str, ...]]:
    """Load a network, and check that it reads encoded chessboards and predicts labels."""
    network = load_network(path).network
    if network.input_size != INPUT_SIZE:
        raise AnalyzerError(
            f"network expects {network.input_size} inputs, "
            f"but chessboards are encoded with {INPUT_SIZE}"
        )
    return network, get_classes(network.architecture[-1][0])
