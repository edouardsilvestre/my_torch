"""Prediction mode of my_torch_analyzer."""

import sys

import numpy as np

from analyzer.arguments import Arguments
from analyzer.chessfile import read_chessfile
from analyzer.encoding import INPUT_SIZE, encode_positions
from analyzer.errors import AnalyzerError
from analyzer.labels import decode_predictions, get_classes
from my_torch import Network, load_network, mc_dropout

PREDICTION_BATCH_SIZE = 1024


def check_chess_network(network: Network) -> tuple[str, ...]:
    """Check that a network reads encoded chessboards, and return the classes it predicts."""
    if network.input_size != INPUT_SIZE:
        raise AnalyzerError(
            f"network expects {network.input_size} inputs, "
            f"but chessboards are encoded with {INPUT_SIZE}"
        )
    return get_classes(network.architecture[-1][0])


def load_chess_network(path: str) -> tuple[Network, tuple[str, ...]]:
    """Load a network, and check that it reads encoded chessboards and predicts labels."""
    network = load_network(path).network
    return network, check_chess_network(network)


def predict_classes(
    network: Network, inputs: np.ndarray, mc_passes: int = 0, seed: int | None = 0
) -> np.ndarray:
    """Return the predicted class of each row, computed by batches to bound the memory.

    With mc_passes, each class is the one with the highest probability on average over that
    many passes with dropout active (MC dropout), instead of a single pass without dropout.
    """
    if mc_passes:
        rng = np.random.default_rng(seed)
        softmax = network.architecture[-1][1] == "identity"
        predictions = [
            np.argmax(mc_dropout(network, inputs[start:start + PREDICTION_BATCH_SIZE],
                                 mc_passes, rng, softmax)[0], axis=1)
            for start in range(0, len(inputs), PREDICTION_BATCH_SIZE)
        ]
        return np.concatenate(predictions) if predictions else np.zeros(0, dtype=int)
    predictions = [
        network.predict(inputs[start:start + PREDICTION_BATCH_SIZE])
        for start in range(0, len(inputs), PREDICTION_BATCH_SIZE)
    ]
    return np.concatenate(predictions) if predictions else np.zeros(0, dtype=int)


def predict(arguments: Arguments) -> None:
    """Print the predicted label of each chessboard, one per line, in the order of the file."""
    network, classes = load_chess_network(arguments.load_file)
    chessboards = read_chessfile(arguments.chess_file)
    inputs = encode_positions(chessboard.position for chessboard in chessboards)
    labels = decode_predictions(predict_classes(network, inputs), classes)
    if labels:
        sys.stdout.write("\n".join(labels) + "\n")
    sys.stdout.flush()
