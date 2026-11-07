"""Tests of the metrics comparing predicted and expected classes."""

import numpy as np
import pytest

from my_torch import MyTorchError
from my_torch.metrics import ClassMetrics, compute_metrics, confusion_matrix

EXPECTED = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
PREDICTED = [0, 0, 1, 2, 1, 1, 0, 2, 2, 2]


def test_confusion_matrix_by_hand():
    assert confusion_matrix(EXPECTED, PREDICTED, 3).tolist() == [
        [2, 1, 1],
        [1, 2, 0],
        [0, 0, 3],
    ]


def test_metrics_by_hand():
    metrics = compute_metrics(confusion_matrix(EXPECTED, PREDICTED, 3))
    assert metrics.accuracy == pytest.approx(0.7)
    expected = [(2 / 3, 1 / 2, 4 / 7, 4), (2 / 3, 2 / 3, 2 / 3, 3), (3 / 4, 1.0, 6 / 7, 3)]
    for values, (precision, recall, f1, support) in zip(metrics.per_class, expected):
        assert values.precision == pytest.approx(precision)
        assert values.recall == pytest.approx(recall)
        assert values.f1 == pytest.approx(f1)
        assert values.support == support
    assert metrics.macro.precision == pytest.approx((2 / 3 + 2 / 3 + 3 / 4) / 3)
    assert metrics.macro.recall == pytest.approx((1 / 2 + 2 / 3 + 1) / 3)
    assert metrics.macro.f1 == pytest.approx((4 / 7 + 2 / 3 + 6 / 7) / 3)
    assert metrics.macro.support == 10


def test_perfect_predictions():
    metrics = compute_metrics(confusion_matrix([0, 1, 2, 1], [0, 1, 2, 1], 3))
    assert metrics.accuracy == 1.0
    assert all(values.f1 == 1.0 for values in metrics.per_class)


def test_always_the_same_class_is_seen_by_the_macro_average():
    metrics = compute_metrics(confusion_matrix([0, 1, 2, 0, 1, 2], [0] * 6, 3))
    assert metrics.accuracy == pytest.approx(1 / 3)
    assert metrics.per_class[0] == ClassMetrics(pytest.approx(1 / 3), 1.0, pytest.approx(0.5), 2)
    assert metrics.per_class[1] == ClassMetrics(0.0, 0.0, 0.0, 2)
    assert metrics.macro.f1 == pytest.approx(0.5 / 3)


def test_macro_average_ignores_classes_never_expected_nor_predicted():
    metrics = compute_metrics(confusion_matrix([0, 1, 0, 1], [0, 1, 1, 1], 5))
    assert metrics.macro.precision == pytest.approx((1.0 + 2 / 3) / 2)
    assert metrics.macro.recall == pytest.approx((1 / 2 + 1.0) / 2)


def test_class_predicted_but_never_expected_counts_in_the_macro_average():
    metrics = compute_metrics(confusion_matrix([0, 0], [0, 1], 2))
    assert metrics.per_class[1] == ClassMetrics(0.0, 0.0, 0.0, 0)
    assert metrics.macro.precision == pytest.approx(0.5)


def test_accepted_integer_types():
    matrix = confusion_matrix(np.array([0, 1], dtype=np.uint8), np.array([1, 1]), 2)
    assert matrix.tolist() == [[0, 1], [0, 1]]
    assert confusion_matrix([], [], 3).tolist() == [[0] * 3] * 3


@pytest.mark.parametrize("expected, predicted, classes, message", [
    ([0, 1], [0], 2, "same length"),
    ([[0]], [[0]], 2, "same length"),
    ([0, 3], [0, 1], 3, "expected classes must be between 0 and 2"),
    ([0, 1], [0, -1], 2, "predicted classes must be between 0 and 1"),
    ([0.5, 1], [0, 1], 2, "expected classes must be integers"),
])
def test_invalid_classes_raise_an_error(expected, predicted, classes, message):
    with pytest.raises(MyTorchError, match=message):
        confusion_matrix(expected, predicted, classes)


def test_no_example_raises_an_error():
    with pytest.raises(MyTorchError, match="no example to evaluate"):
        compute_metrics(confusion_matrix([], [], 3))
