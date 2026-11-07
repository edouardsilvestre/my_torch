"""Tests of the conversion of labels to and from the classes of a network."""

import numpy as np
import pytest

from analyzer.errors import AnalyzerError
from analyzer.labels import (
    CLASS_SETS, LABELS, check_label, decode_predictions, encode_labels, get_classes, to_class,
)
from my_torch.training import is_one_hot

COLOR_CLASSES = ("Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black")


def test_labels_of_the_subject():
    assert LABELS == ("Nothing", "Check", "Checkmate", "Check White", "Check Black",
                      "Checkmate White", "Checkmate Black")


@pytest.mark.parametrize("label", LABELS)
def test_known_labels_are_accepted(label):
    assert check_label(label) == label


@pytest.mark.parametrize("text", [
    "", "check", "CHECK", "Chek", "Stalemate", "Nothing White", "Check Red", "Check  White",
    "White Check", " Check",
])
def test_unknown_labels_are_refused(text):
    with pytest.raises(AnalyzerError, match="unknown label"):
        check_label(text)


@pytest.mark.parametrize("output_size, classes", [
    (2, ("Nothing", "Check")),
    (3, ("Nothing", "Check", "Checkmate")),
    (5, COLOR_CLASSES),
])
def test_classes_follow_the_number_of_outputs(output_size, classes):
    assert get_classes(output_size) == classes


@pytest.mark.parametrize("output_size", [0, 1, 4, 6, 769])
def test_other_numbers_of_outputs_are_refused(output_size):
    with pytest.raises(AnalyzerError, match="expected 2, 3 or 5 outputs"):
        get_classes(output_size)


@pytest.mark.parametrize("label, two, three, five", [
    ("Nothing", "Nothing", "Nothing", "Nothing"),
    ("Check", "Check", "Check", None),
    ("Checkmate", "Check", "Checkmate", None),
    ("Check White", "Check", "Check", "Check White"),
    ("Check Black", "Check", "Check", "Check Black"),
    ("Checkmate White", "Check", "Checkmate", "Checkmate White"),
    ("Checkmate Black", "Check", "Checkmate", "Checkmate Black"),
])
def test_labels_are_expressed_with_the_classes_of_the_network(label, two, three, five):
    assert to_class(label, CLASS_SETS[2]) == two
    assert to_class(label, CLASS_SETS[3]) == three
    if five is None:
        with pytest.raises(AnalyzerError, match="does not tell"):
            to_class(label, CLASS_SETS[5])
    else:
        assert to_class(label, CLASS_SETS[5]) == five


def test_targets_are_one_hot_by_hand():
    targets = encode_labels(["Nothing", "Checkmate Black", "Check White", "Nothing"],
                            COLOR_CLASSES)
    assert targets.dtype == np.float64
    assert np.array_equal(targets, [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1],
                                    [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])
    assert is_one_hot(targets)


def test_targets_use_the_reduced_labels():
    targets = encode_labels(["Checkmate White", "Check Black", "Nothing"], CLASS_SETS[3])
    assert np.array_equal(targets, [[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    targets = encode_labels(["Checkmate White", "Check Black", "Nothing"], CLASS_SETS[2])
    assert np.array_equal(targets, [[0, 1], [0, 1], [1, 0]])


def test_no_labels_give_no_targets():
    assert encode_labels([], CLASS_SETS[3]).shape == (0, 3)


def test_unconvertible_label_raises_an_error():
    with pytest.raises(AnalyzerError, match="'Check'"):
        encode_labels(["Nothing", "Check"], COLOR_CLASSES)


@pytest.mark.parametrize("output_size", [2, 3, 5])
def test_decoding_gives_back_the_labels(output_size):
    classes = get_classes(output_size)
    labels = list(classes) * 2
    targets = encode_labels(labels, classes)
    assert decode_predictions(np.argmax(targets, axis=1), classes) == labels


def test_decoding_by_hand():
    assert decode_predictions(np.array([2, 0, 1]), CLASS_SETS[3]) == [
        "Checkmate", "Nothing", "Check",
    ]
    assert decode_predictions([4, 3], COLOR_CLASSES) == ["Checkmate Black", "Checkmate White"]
