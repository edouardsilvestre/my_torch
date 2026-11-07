"""Generate a dataset of labelled chess positions, in the format read by my_torch_analyzer.

Run from the root of the repository, for example:
    python3 -m scripts.generate_dataset --size 30000 --seed 0 --output datasets/train.txt
The same arguments always give the same file.
"""

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

from dataset.generate import LABELS, generate_dataset, label_shares


def parse_shares(text: str) -> dict[str, float]:
    """Read 'nothing:check:checkmate' shares, such as '1:1:1'."""
    try:
        nothing, check, checkmate = (float(part) for part in text.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected three numbers, such as 1:1:1") from error
    if min(nothing, check, checkmate) < 0 or nothing + check + checkmate <= 0:
        raise argparse.ArgumentTypeError("shares must be non-negative, and not all zero")
    return label_shares(nothing, check, checkmate)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, required=True, help="number of positions")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random games")
    parser.add_argument("--output", type=Path, required=True, help="file to write")
    parser.add_argument("--shares", type=parse_shares, default=parse_shares("1:1:1"),
                        help="shares of Nothing, Check and Checkmate (default 1:1:1)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="processes generating at once (default: all cores); "
                             "the dataset does not depend on it")
    arguments = parser.parse_args(argv)
    if arguments.size < 1:
        parser.error("--size must be a positive integer")
    if arguments.jobs < 1:
        parser.error("--jobs must be a positive integer")
    return arguments


def main(argv: list[str]) -> None:
    arguments = parse_arguments(argv)
    start = time.perf_counter()
    lines, games = generate_dataset(arguments.size, arguments.seed, arguments.shares,
                                    arguments.jobs)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("\n".join(lines) + "\n")
    counts = Counter(line.split(" ", 6)[6] for line in lines)
    print(f"{len(lines)} positions from {games} games in {time.perf_counter() - start:.0f} s, "
          f"written to {arguments.output}")
    for label in LABELS:
        print(f"    {label:16s} {counts[label]}")


if __name__ == "__main__":
    main(sys.argv[1:])
