"""Evaluation of a chess network on a file of labelled chessboards."""

from dataclasses import dataclass

import numpy as np

from analyzer.labels import CLASS_SETS, to_class
from analyzer.predict import load_chess_network, predict_classes
from analyzer.train import read_training_data
from my_torch.metrics import Metrics, compute_metrics, confusion_matrix


@dataclass(frozen=True)
class LevelMetrics:
    """Metrics of the predictions once expressed with a given set of classes."""

    classes: tuple[str, ...]
    metrics: Metrics


def reduce_classes(
    indices: np.ndarray, classes: tuple[str, ...], fewer_classes: tuple[str, ...]
) -> np.ndarray:
    """Express class indices with fewer classes: without colors, then with checkmates as checks."""
    mapping = [fewer_classes.index(to_class(label, fewer_classes)) for label in classes]
    return np.array(mapping, dtype=np.int64)[indices]


def evaluate(network_path: str, chess_path: str) -> list[LevelMetrics]:
    """Compare the predictions of a network with the labels of a file.

    The metrics are given with the classes of the network, then with each simpler set of
    classes: a network with colors is also evaluated as if it had 3, then 2 classes.
    """
    network, classes = load_chess_network(network_path)
    inputs, targets = read_training_data(chess_path, classes)
    expected = np.argmax(targets, axis=1)
    predicted = predict_classes(network, inputs)
    levels = []
    for size in sorted(CLASS_SETS, reverse=True):
        if size > len(classes):
            continue
        fewer = CLASS_SETS[size]
        confusion = confusion_matrix(
            reduce_classes(expected, classes, fewer), reduce_classes(predicted, classes, fewer),
            len(fewer),
        )
        levels.append(LevelMetrics(fewer, compute_metrics(confusion)))
    return levels
