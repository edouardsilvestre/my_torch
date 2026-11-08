"""Training mode of my_torch_analyzer."""

from typing import Any

import numpy as np

from analyzer.arguments import Arguments
from analyzer.chessfile import read_chessfile
from analyzer.encoding import encode_positions
from analyzer.errors import AnalyzerError
from analyzer.labels import encode_labels, to_class
from analyzer.predict import check_chess_network
from my_torch import MyTorchError, SavedNetwork, load_network, save_network, train
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
        raise AnalyzerError(f"no labelled chessboard in '{path}'")
    for chessboard in chessboards:
        try:
            to_class(chessboard.label, classes)
        except AnalyzerError as error:
            raise AnalyzerError(f"{path}:{chessboard.line_number}: {error}") from error
    inputs = encode_positions(chessboard.position for chessboard in chessboards)
    targets = encode_labels((chessboard.label for chessboard in chessboards), classes)
    return inputs, targets


def hold_out(
    inputs: np.ndarray, targets: np.ndarray, share: float, seed: int | None
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Keep a random share of the chessboards aside, as a validation set."""
    count = round(len(inputs) * share)
    if not 0 < count < len(inputs):
        raise AnalyzerError(
            f"validation_split {share} keeps {count} of the {len(inputs)} chessboards aside: "
            "both the training and the validation sets need at least one"
        )
    order = np.random.default_rng(seed).permutation(len(inputs))
    kept, aside = order[count:], order[:count]
    return inputs[kept], targets[kept], (inputs[aside], targets[aside])


def run_training(arguments: Arguments) -> None:
    """Train the network on the labelled chessboards, then save it once training is over."""
    saved, classes, settings = load_training_network(arguments.load_file)
    if settings["patience"] is not None and not settings["validation_split"]:
        raise AnalyzerError(
            "early stopping needs a validation set: set validation_split in the training "
            "settings"
        )
    inputs, targets = read_training_data(arguments.chess_file, classes)
    validation = None
    if settings["validation_split"]:
        inputs, targets, validation = hold_out(
            inputs, targets, settings["validation_split"], settings["seed"]
        )
    train(
        saved.network, saved.loss, saved.optimizer, inputs, targets,
        settings["epochs"], settings["batch_size"], settings["shuffle"], settings["seed"],
        validation=validation, verbose=True, l2=settings["l2"], patience=settings["patience"],
        monitor=settings["monitor"],
    )
    save_network(
        arguments.save_file or arguments.load_file, saved.network, saved.loss,
        saved.optimizer, settings,
    )
