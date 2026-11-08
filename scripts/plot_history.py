"""Draw the learning curves of the configurations compared by compare_configs.

Run from the root of the repository, for example:
    python3 -m scripts.compare_configs --json results.json ... configs/candidates/*.conf
    python3 -m scripts.plot_history --json results.json --output docs/images/curves.png
Needs matplotlib, which is only used by this script (see requirements-dev.txt).
"""

import argparse
import json
import sys
from pathlib import Path

# Fixed order: the n-th configuration of the file always gets the n-th color.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PANELS = [("loss", "Loss"), ("accuracy", "Accuracy")]


def read_results(path: Path) -> list[dict]:
    """Read the results written by compare_configs, and check they hold the curves."""
    try:
        results = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(results, list) or not results:
        raise ValueError(f"{path}: no configuration")
    if len(results) > len(COLORS):
        raise ValueError(f"{path}: at most {len(COLORS)} configurations can be drawn together")
    for result in results:
        history = result.get("history") if isinstance(result, dict) else None
        if not isinstance(history, dict) or any(
                not history.get(key) for key in ("loss", "accuracy", "val_loss", "val_accuracy")):
            raise ValueError(f"{path}: every configuration needs its training and validation "
                             "loss and accuracy")
    return results


def kept_epoch(result: dict) -> int:
    """The epoch whose weights the network keeps: the best one with early stopping."""
    return result.get("kept_epoch", len(result["history"]["val_loss"]))


def label_line_ends(axis, results: list[dict], key: str, color: str, gap: float = 13) -> None:
    """Write each configuration's name at the end of its curve, moved apart vertically when
    two names would overlap (gap in points)."""
    figure = axis.figure
    figure.canvas.draw()
    ends = sorted(
        ((len(result["history"][key]), result["history"][key][-1], Path(result["config"]).stem)
         for result in results), key=lambda end: end[1])
    to_points = 72 / figure.dpi
    heights = [axis.transData.transform((x, y))[1] * to_points for x, y, _ in ends]
    for index in range(1, len(heights)):
        heights[index] = max(heights[index], heights[index - 1] + gap)
    for (x, y, name), height in zip(ends, heights):
        shift = height - axis.transData.transform((x, y))[1] * to_points
        axis.annotate(name, (x, y), xytext=(8, shift), textcoords="offset points",
                      va="center", fontsize=9, color=color)


def plot(results: list[dict], output: Path, title: str | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(PANELS), figsize=(12, 4.8), facecolor=SURFACE)
    for axis, (key, name) in zip(axes, PANELS):
        axis.set_facecolor(SURFACE)
        for color, result in zip(COLORS, results):
            label = Path(result["config"]).stem
            history = result["history"]
            epochs = range(1, len(history[key]) + 1)
            axis.plot(epochs, history[key], color=color, linewidth=1.5, linestyle=(0, (4, 3)))
            axis.plot(epochs, history[f"val_{key}"], color=color, linewidth=2, label=label)
            kept = kept_epoch(result)
            axis.plot(kept, history[f"val_{key}"][kept - 1], "o", color=color, markersize=8,
                      markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        axis.set_title(name, loc="left", color=INK, fontsize=12)
        axis.set_xlabel("Epoch", color=MUTED)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.tick_params(colors=MUTED)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        longest = max(len(result["history"][key]) for result in results)
        axis.set_xlim(1, longest + max(2, longest * 0.25))
    legend = [
        plt.Line2D([], [], color=MUTED, linewidth=2, label="validation"),
        plt.Line2D([], [], color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), label="training"),
        plt.Line2D([], [], color=MUTED, marker="o", markersize=8, linestyle="",
                   label="kept epoch"),
    ]
    axes[0].legend(handles=axes[0].get_legend_handles_labels()[0] + legend, frameon=False,
                   labelcolor=INK, fontsize=9)
    if title:
        figure.suptitle(title, x=0.01, ha="left", color=INK, fontsize=14)
    figure.tight_layout()
    # Placed once the layout is final, since moving the names apart works in points.
    label_line_ends(axes[-1], results, f"val_{PANELS[-1][0]}", INK)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120, facecolor=SURFACE)
    plt.close(figure)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", required=True, type=Path, help="results of compare_configs")
    parser.add_argument("--output", required=True, type=Path, help="image to write, e.g. a .png")
    parser.add_argument("--title", help="title drawn above the curves")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    arguments = parse_arguments(argv)
    try:
        results = read_results(arguments.json)
    except ValueError as error:
        sys.exit(f"plot_history: {error}")
    plot(results, arguments.output, arguments.title)
    print(f"{arguments.output}: {len(results)} configurations")


if __name__ == "__main__":
    main(sys.argv[1:])
