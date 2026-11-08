"""Draw the results of a benchmark: learning curves, a summary and a confusion matrix.

Run from the root of the repository, for example:
    python3 -m scripts.plot_benchmark --json datasets/activations.json \\
        --output docs/images/benchmarks/activations
writes activations_curves.png, activations_summary.png and activations_confusion.png.
Needs matplotlib, which is only used by the drawing scripts (see requirements-dev.txt).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scripts.plot_history import COLORS, GRID, INK, MUTED, SURFACE, plot as plot_curves

SUMMARY = [
    (lambda result: result["validation_accuracy"]["5"] * 100, "Validation accuracy (%)",
     "{:.1f}"),
    (lambda result: result["seconds_per_epoch"], "Training time per epoch (s)", "{:.1f}"),
    (lambda result: result["inference_microseconds"], "Inference per position (µs)", "{:.1f}"),
]


def read_benchmark_results(path: Path) -> tuple[str, list[dict]]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error
    results = data.get("results") if isinstance(data, dict) else None
    if not results:
        raise ValueError(f"{path}: no benchmark results")
    if len(results) > len(COLORS):
        raise ValueError(f"{path}: at most {len(COLORS)} variants can be drawn together")
    keys = ("config", "history", "validation_accuracy", "confusion", "classes",
            "seconds_per_epoch", "inference_microseconds")
    if any(not isinstance(result, dict) or any(key not in result for key in keys)
           for result in results):
        raise ValueError(f"{path}: every variant needs {', '.join(keys)}")
    return str(data.get("title", path.stem)), results


def style(axis, title: str) -> None:
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", color=INK, fontsize=12)
    axis.tick_params(colors=MUTED)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)


def plot_summary(title: str, results: list[dict], output: Path) -> None:
    """One horizontal bar per variant, in its curve color, for each measure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [result["config"] for result in results]
    size = (5 * len(SUMMARY), 1.2 + 0.5 * len(results))
    figure, axes = plt.subplots(1, len(SUMMARY), figsize=size, facecolor=SURFACE)
    for axis, (value, name, text) in zip(axes, SUMMARY):
        values = [value(result) for result in results]
        positions = np.arange(len(results))[::-1]
        axis.barh(positions, values, color=COLORS[:len(results)], height=0.6)
        axis.set_yticks(positions, names if axis is axes[0] else [""] * len(names))
        axis.grid(True, axis="x", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        low = min(values)
        if "accuracy" in name:
            axis.set_xlim(max(0, low - (max(values) - low) - 2), max(values) + 1.5)
        for position, number in zip(positions, values):
            axis.annotate(text.format(number), (number, position), xytext=(4, 0),
                          textcoords="offset points", va="center", fontsize=9, color=INK)
        style(axis, name)
    figure.suptitle(title, x=0.01, ha="left", color=INK, fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=120, facecolor=SURFACE)
    plt.close(figure)


def plot_confusion(result: dict, output: Path) -> None:
    """The confusion matrix of a variant, each row as shares of its expected class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confusion = np.array(result["confusion"], dtype=float)
    shares = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    classes = result["classes"]
    figure, axis = plt.subplots(figsize=(1.3 * len(classes) + 2.5, 1.1 * len(classes) + 1.5),
                                facecolor=SURFACE)
    axis.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(classes)):
        for column in range(len(classes)):
            dark = shares[row, column] > 0.55
            axis.text(column, row, f"{shares[row, column] * 100:.1f} %\n"
                      f"{int(confusion[row, column])}", ha="center", va="center", fontsize=8,
                      color=SURFACE if dark else INK)
    axis.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("Predicted", color=MUTED)
    axis.set_ylabel("Expected", color=MUTED)
    style(axis, f"Confusion matrix: {result['config']}")
    for side in ("left", "bottom"):
        axis.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=120, facecolor=SURFACE)
    plt.close(figure)


def best(results: list[dict]) -> dict:
    """The variant with the best validation accuracy, the first one on a tie."""
    return max(results, key=lambda result: result["validation_accuracy"]["5"])


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", required=True, type=Path, help="results of scripts.benchmark")
    parser.add_argument("--output", required=True, type=Path,
                        help="start of the image names, e.g. docs/images/benchmarks/activations")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    arguments = parse_arguments(argv)
    try:
        title, results = read_benchmark_results(arguments.json)
    except ValueError as error:
        sys.exit(f"plot_benchmark: {error}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    stem = arguments.output.name
    images = [arguments.output.with_name(f"{stem}_{kind}.png")
              for kind in ("curves", "summary", "confusion")]
    plot_curves(results, images[0], title)
    plot_summary(title, results, images[1])
    plot_confusion(best(results), images[2])
    for image in images:
        print(image)


if __name__ == "__main__":
    main(sys.argv[1:])
