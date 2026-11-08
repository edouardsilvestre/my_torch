"""Follow the signal through a deep ReLU network, for each weight initializer.

Run from the root of the repository:
    python3 -m scripts.signal_by_initializer --output docs/images/benchmarks/signal.png
The variance of the values before activation is measured after each of 10 ReLU layers of 500
neurons, for the same random inputs. A good initializer keeps it steady: a vanishing variance
means vanishing gradients, an exploding one saturates or overflows.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from my_torch import Network

INITIALIZERS = ["uniform", "normal", "lecun", "xavier_uniform", "xavier_normal", "he"]


def variances(initializer: str, layers: int, width: int, examples: int, seed: int) -> list[float]:
    """The variance of the values before activation of each layer."""
    network = Network(width, [(width, "relu")] * layers, initializer, seed)
    signal = np.random.default_rng(seed).standard_normal((examples, width))
    result = []
    for layer in network.layers:
        before = signal @ layer.weights + layer.biases
        result.append(float(before.var()))
        signal = np.maximum(before, 0.0)
    return result


def plot(table: dict[str, list[float]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.plot_history import COLORS, GRID, INK, MUTED, SURFACE

    figure, axis = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    for color, (name, values) in zip(COLORS, table.items()):
        layers = range(1, len(values) + 1)
        axis.plot(layers, values, color=color, linewidth=2, marker="o", markersize=5, label=name)
    # lecun and xavier give the same variance on square layers: a legend keeps both readable.
    axis.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower left")
    axis.set_yscale("log")
    axis.set_title("Variance before activation, 10 ReLU layers of 500 neurons", loc="left",
                   color=INK, fontsize=12)
    axis.set_xlabel("Layer", color=MUTED)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.tick_params(colors=MUTED)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120, facecolor=SURFACE)
    plt.close(figure)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--width", type=int, default=500)
    parser.add_argument("--examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="also draw the variances into this image")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    arguments = parse_arguments(argv)
    table = {name: variances(name, arguments.layers, arguments.width, arguments.examples,
                             arguments.seed) for name in INITIALIZERS}
    shown = (1, arguments.layers // 2, arguments.layers)
    print("| Initializer | " + " | ".join(f"Layer {index}" for index in shown) + " |")
    print("|---|---|---|---|")
    for name, values in table.items():
        picked = (values[0], values[arguments.layers // 2 - 1], values[-1])
        print(f"| {name} | " + " | ".join(f"{value:.3g}" for value in picked) + " |")
    if arguments.output:
        plot(table, arguments.output)


if __name__ == "__main__":
    main(sys.argv[1:])
