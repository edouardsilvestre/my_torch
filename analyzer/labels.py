"""Labels of the chessboards, and their conversion to and from the classes of a network."""

from collections.abc import Iterable, Sequence

import numpy as np

from analyzer.errors import AnalyzerError

NOTHING = "Nothing"
CHECK = "Check"
CHECKMATE = "Checkmate"
LABELS = (
    NOTHING, CHECK, CHECKMATE,
    "Check White", "Check Black", "Checkmate White", "Checkmate Black",
)
CLASS_SETS = {
    2: (NOTHING, CHECK),
    3: (NOTHING, CHECK, CHECKMATE),
    5: (NOTHING, "Check White", "Check Black", "Checkmate White", "Checkmate Black"),
}


def check_label(text: str) -> str:
    """Return the label if it is one of those of the subject."""
    if text not in LABELS:
        raise AnalyzerError(f"unknown label '{text}', expected one of: {', '.join(LABELS)}")
    return text


def get_classes(output_size: int) -> tuple[str, ...]:
    """Return the classes predicted by a network, from its number of outputs."""
    if output_size not in CLASS_SETS:
        sizes = list(map(str, CLASS_SETS))
        sizes = f"{', '.join(sizes[:-1])} or {sizes[-1]}"
        raise AnalyzerError(
            f"a network with {output_size} outputs cannot predict chess labels: "
            f"expected {sizes} outputs"
        )
    return CLASS_SETS[output_size]


def to_class(label: str, classes: Sequence[str]) -> str:
    """Express a label with the given classes, dropping what they do not distinguish.

    The color is dropped when the classes have none, and a checkmate counts as a check
    when the classes do not hold checkmates. A label that tells less than the classes,
    such as 'Check' for classes with colors, cannot be converted.
    """
    if label in classes:
        return label
    state = label.split()[0]
    if state in classes:
        return state
    if state == CHECKMATE and CHECK in classes:
        return CHECK
    raise AnalyzerError(
        f"label '{label}' does not tell which of these classes it is: {', '.join(classes)}"
    )


def encode_labels(labels: Iterable[str], classes: Sequence[str]) -> np.ndarray:
    """Return one-hot targets, one row per label and one column per class."""
    indices = [classes.index(to_class(label, classes)) for label in labels]
    targets = np.zeros((len(indices), len(classes)))
    targets[np.arange(len(indices)), indices] = 1.0
    return targets


def decode_predictions(indices: Iterable[int], classes: Sequence[str]) -> list[str]:
    """Return the label of each predicted class index."""
    return [classes[index] for index in indices]
