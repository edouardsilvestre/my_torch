"""Split a labelled dataset into training, validation and test files, with the same label shares.

Run from the root of the repository, for example:
    python3 -m scripts.split_dataset --input datasets/all.txt --ratios 80:10:10 --seed 0
This writes all_train.txt, all_validation.txt and all_test.txt next to all.txt.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from analyzer.errors import AnalyzerError
from analyzer.labels import LABELS
from dataset.split import PARTS, split_dataset


def parse_ratios(text: str) -> tuple[float, float, float]:
    """Read 'train:validation:test' ratios, such as '80:10:10'."""
    try:
        ratios = tuple(float(part) for part in text.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected three numbers, such as 80:10:10") from error
    if len(ratios) != 3:
        raise argparse.ArgumentTypeError("expected three numbers, such as 80:10:10")
    if min(ratios) < 0 or ratios[0] <= 0:
        raise argparse.ArgumentTypeError("ratios must be non-negative, with a training part")
    return ratios


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="labelled dataset to split")
    parser.add_argument("--ratios", type=parse_ratios, default=parse_ratios("80:10:10"),
                        help="shares of the train, validation and test parts (default 80:10:10)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the split")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    arguments = parse_arguments(argv)
    try:
        lines = arguments.input.read_text().splitlines()
        parts, duplicates = split_dataset(lines, arguments.ratios, arguments.seed)
    except OSError as error:
        sys.exit(f"split_dataset: {arguments.input}: {error.strerror}")
    except (UnicodeDecodeError, AnalyzerError) as error:
        sys.exit(f"split_dataset: {arguments.input}: {error}")
    stem = arguments.input.with_suffix("")
    for part in PARTS:
        path = Path(f"{stem}_{part}.txt")
        path.write_text("".join(f"{line}\n" for line in parts[part]))
        counts = Counter(line.split(" ", 6)[6] for line in parts[part])
        shares = ", ".join(f"{label} {counts[label]}" for label in LABELS if counts[label])
        print(f"{path}: {len(parts[part])} positions ({shares})")
    if duplicates:
        print(f"{duplicates} duplicated positions removed before the split")


if __name__ == "__main__":
    main(sys.argv[1:])
