"""Draw how hyperparameter searches progress, trial after trial.

Run from the root of the repository, for example:
    python3 -m scripts.plot_search --output docs/images/recherche.png \\
        datasets/search_random.jsonl datasets/search_genetic.jsonl
Each file of trials, written by search_hyperparameters, gets its own color: a dot per trial,
and a line with the best score found so far.
Needs matplotlib, which is only used by the drawing scripts (see requirements-dev.txt).
"""

import argparse
import sys
from pathlib import Path

from scripts.plot_history import COLORS, GRID, INK, MUTED, SURFACE
from scripts.search_hyperparameters import read_trials


def best_so_far(scores: list[float]) -> list[float]:
    best, result = float("-inf"), []
    for score in scores:
        best = max(best, score)
        result.append(best)
    return result


def plot(searches: dict[str, list[float]], output: Path, title: str | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    for index, (color, (name, scores)) in enumerate(zip(COLORS, searches.items())):
        trials = range(1, len(scores) + 1)
        axis.plot(trials, [score * 100 for score in scores], "o", color=color, markersize=6,
                  alpha=0.45, markeredgewidth=0)
        # Searches often share their best score: each next line is thinner and dashed, so
        # that the lines below stay visible.
        axis.plot(trials, [score * 100 for score in best_so_far(scores)], color=color,
                  linewidth=max(4 - 1.5 * index, 1.5), linestyle="-" if index == 0 else "--",
                  label=f"{name}: best so far")
    axis.set_xlabel("Trial", color=MUTED)
    axis.set_ylabel("Validation accuracy (%)", color=MUTED)
    axis.set_title(title or "Hyperparameter search", loc="left", color=INK, fontsize=12)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.tick_params(colors=MUTED)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower right")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120, facecolor=SURFACE)
    plt.close(figure)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, type=Path, help="image to write")
    parser.add_argument("--title", help="title drawn above the plot")
    parser.add_argument("trials", nargs="+", type=Path, help="files of trials, one per search")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    arguments = parse_arguments(argv)
    searches = {}
    for path in arguments.trials:
        try:
            trials = sorted(read_trials(path), key=lambda trial: trial.number)
        except (OSError, ValueError, TypeError) as error:
            sys.exit(f"plot_search: {path}: {error}")
        if not trials:
            sys.exit(f"plot_search: {path}: no trial")
        searches[path.stem] = [trial.score for trial in trials]
    if len(searches) > len(COLORS):
        sys.exit(f"plot_search: at most {len(COLORS)} searches can be drawn together")
    plot(searches, arguments.output, arguments.title)
    print(f"{arguments.output}: {len(searches)} searches")


if __name__ == "__main__":
    main(sys.argv[1:])
