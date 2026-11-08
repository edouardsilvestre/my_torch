"""Measure how well a network predicts the labels of a file of chessboards.

Run from the root of the repository, for example:
    python3 -m scripts.evaluate --network my_torch_network.nn --chessboards datasets/all_test.txt
"""

import argparse
import json
import sys
from pathlib import Path

from analyzer.errors import AnalyzerError
from analyzer.evaluate import LevelMetrics, evaluate
from my_torch import MyTorchError

NAME_WIDTH = 17


def format_level(level: LevelMetrics) -> str:
    metrics = level.metrics
    correct = int(metrics.confusion.trace())
    total = int(metrics.confusion.sum())
    lines = [
        f"{len(level.classes)} classes: accuracy {metrics.accuracy:.4f} ({correct} / {total})",
        "",
        f"    {'Class':{NAME_WIDTH}} {'Precision':>9} {'Recall':>9} {'F1':>9} {'Support':>9}",
    ]
    rows = list(zip(level.classes, metrics.per_class)) + [("Macro average", metrics.macro)]
    for name, values in rows:
        lines.append(f"    {name:{NAME_WIDTH}} {values.precision:9.4f} {values.recall:9.4f} "
                     f"{values.f1:9.4f} {values.support:9d}")
    lines += ["", "    Confusion matrix (rows: expected, columns: predicted)"]
    lines.append(f"    {'':{NAME_WIDTH}}" + "".join(f"{name[:16]:>17}" for name in level.classes))
    for name, row in zip(level.classes, metrics.confusion):
        lines.append(f"    {name:{NAME_WIDTH}}" + "".join(f"{count:17d}" for count in row))
    return "\n".join(lines)


def to_json(level: LevelMetrics) -> dict:
    metrics = level.metrics
    return {
        "classes": list(level.classes),
        "accuracy": metrics.accuracy,
        "per_class": {
            name: vars(values) for name, values in zip(level.classes, metrics.per_class)
        },
        "macro": vars(metrics.macro),
        "confusion": metrics.confusion.tolist(),
    }


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--network", required=True, help=".nn file of the network")
    parser.add_argument("--chessboards", required=True, help="file of labelled chessboards")
    parser.add_argument("--json", type=Path, help="also write the metrics to this JSON file")
    parser.add_argument("--mc-passes", type=int, default=0, metavar="N",
                        help="predict with MC dropout: the mean of N passes with dropout active")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    arguments = parse_arguments(argv)
    try:
        levels = evaluate(arguments.network, arguments.chessboards, arguments.mc_passes)
    except (AnalyzerError, MyTorchError) as error:
        sys.exit(f"evaluate: {error}")
    mode = f" with MC dropout ({arguments.mc_passes} passes)" if arguments.mc_passes else ""
    print(f"Network {arguments.network} on {arguments.chessboards}{mode}\n")
    print("\n\n".join(format_level(level) for level in levels))
    if arguments.json:
        report = {"network": arguments.network, "chessboards": arguments.chessboards,
                  "mc_passes": arguments.mc_passes, "levels": [to_json(level) for level in levels]}
        arguments.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
