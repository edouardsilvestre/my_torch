"""Metrics comparing the classes predicted by a network with the expected ones."""

from dataclasses import dataclass

import numpy as np

from my_torch.errors import MyTorchError


@dataclass(frozen=True)
class ClassMetrics:
    """Precision, recall and F1 of one class, and its number of expected examples."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class Metrics:
    """Every metric computed from a confusion matrix."""

    confusion: np.ndarray
    accuracy: float
    per_class: list[ClassMetrics]
    macro: ClassMetrics


def confusion_matrix(expected: np.ndarray, predicted: np.ndarray, classes: int) -> np.ndarray:
    """Count the examples of each expected class (row) given each predicted class (column)."""
    expected, predicted = np.asarray(expected), np.asarray(predicted)
    if expected.ndim != 1 or expected.shape != predicted.shape:
        raise MyTorchError(
            f"expected and predicted classes must be two lists of the same length, "
            f"got shapes {expected.shape} and {predicted.shape}"
        )
    for name, values in (("expected", expected), ("predicted", predicted)):
        if values.size and values.dtype.kind not in "iu":
            raise MyTorchError(f"{name} classes must be integers")
        if values.size and (values.min() < 0 or values.max() >= classes):
            raise MyTorchError(f"{name} classes must be between 0 and {classes - 1}")
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (expected.astype(np.int64), predicted.astype(np.int64)), 1)
    return matrix


def divide(numerator: float, denominator: float) -> float:
    """Divide, giving 0 when the denominator is 0, as for a class that is never predicted."""
    return float(numerator / denominator) if denominator else 0.0


def compute_metrics(confusion: np.ndarray) -> Metrics:
    """Compute the accuracy, the metrics of each class and their macro average.

    The macro average gives the same weight to each class that is expected or predicted
    at least once, whatever its number of examples.
    """
    total = int(confusion.sum())
    if total == 0:
        raise MyTorchError("no example to evaluate")
    per_class = []
    for index in range(len(confusion)):
        true_positives = int(confusion[index, index])
        precision = divide(true_positives, confusion[:, index].sum())
        recall = divide(true_positives, confusion[index, :].sum())
        f1 = divide(2 * precision * recall, precision + recall)
        per_class.append(ClassMetrics(precision, recall, f1, int(confusion[index, :].sum())))
    present = [
        metrics for index, metrics in enumerate(per_class)
        if confusion[index, :].sum() or confusion[:, index].sum()
    ]
    macro = ClassMetrics(
        float(np.mean([metrics.precision for metrics in present])),
        float(np.mean([metrics.recall for metrics in present])),
        float(np.mean([metrics.f1 for metrics in present])),
        total,
    )
    return Metrics(confusion, divide(np.trace(confusion), total), per_class, macro)
