"""Training mode of my_torch_analyzer."""

from typing import Any

import numpy as np

from analyzer.chessfile import read_chessfile
from analyzer.encoding import encode_positions
from analyzer.errors import AnalyzerError
from analyzer.labels import encode_labels, to_class
from analyzer.predict import check_chess_network
from my_torch import MyTorchError, SavedNetwork, load_network
from my_torch.config import check_training_settings


def load_training_network(path: str) -> tuple[SavedNetwork, tuple[str, ...], dict[str, Any]]:
    """Load a network with what is needed to train it: loss, optimizer and settings."""
    saved = load_network(path)
    classes = check_chess_network(saved.network)
    missing = [
        name for name, value in (
            ("loss", saved.loss), ("optimizer", saved.optimizer),
            ("training settings", saved.settings),
        )
        if value is None
    ]
    if missing:
        raise AnalyzerError(
            f"network file '{path}' has no {', '.join(missing)}: it cannot be trained; "
            "create it from a configuration with my_torch_generator"
        )
    try:
        settings = check_training_settings(saved.settings)
    except MyTorchError as error:
        raise AnalyzerError(f"network file '{path}': {error}") from error
    return saved, classes, settings


def read_training_data(path: str, classes: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Read the labelled chessboards, and return the inputs and targets of the network."""
    chessboards = read_chessfile(path, require_labels=True)
    if not chessboards:
        raise AnalyzerError(f"no chessboard to train on in '{path}'")
    for chessboard in chessboards:
        try:
            to_class(chessboard.label, classes)
        except AnalyzerError as error:
            raise AnalyzerError(f"{path}:{chessboard.line_number}: {error}") from error
    inputs = encode_positions(chessboard.position for chessboard in chessboards)
    targets = encode_labels((chessboard.label for chessboard in chessboards), classes)
    return inputs, targets
