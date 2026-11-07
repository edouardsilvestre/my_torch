"""Merge the labelled chessboards of the dataset given with the subject into one file.

Run from the root of the repository, for example:
    python3 -m scripts.prepare_official_dataset --archive dataset.zip --output datasets/official.txt
The archive is read without being extracted, and every line is checked. Duplicated positions
are kept here: scripts.split_dataset removes them before splitting.
"""

import argparse
import sys
import zipfile
from collections import Counter
from pathlib import Path

from analyzer.chessfile import parse_line
from analyzer.errors import AnalyzerError
from analyzer.labels import LABELS


def read_archive(path: Path) -> list[tuple[str, list[str]]]:
    """Return the name and the lines of each text file of the archive, sorted by name."""
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".txt"))
        return [(name, archive.read(name).decode("utf-8").splitlines()) for name in names]


def check_lines(name: str, lines: list[str]) -> list[str]:
    """Return the non-empty lines of a file, after checking that each has a valid label."""
    checked = []
    for number, text in enumerate(lines, start=1):
        if not text.strip():
            continue
        try:
            chessboard = parse_line(text, number)
            if chessboard.label is None:
                raise AnalyzerError("missing expected label after the FEN")
        except AnalyzerError as error:
            raise AnalyzerError(f"{name}:{number}: {error}") from error
        checked.append(" ".join(text.split()))
    return checked


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, default=Path("dataset.zip"),
                        help="archive given with the subject (default: dataset.zip)")
    parser.add_argument("--output", type=Path, required=True, help="file to write")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    arguments = parse_arguments(argv)
    try:
        files = read_archive(arguments.archive)
        merged = [(name, check_lines(name, lines)) for name, lines in files]
    except OSError as error:
        sys.exit(f"prepare_official_dataset: {arguments.archive}: {error.strerror or error}")
    except (zipfile.BadZipFile, UnicodeDecodeError, AnalyzerError) as error:
        sys.exit(f"prepare_official_dataset: {arguments.archive}: {error}")
    lines = [line for _, file_lines in merged for line in file_lines]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("".join(f"{line}\n" for line in lines))
    for name, file_lines in merged:
        print(f"    {name}: {len(file_lines)} positions")
    counts = Counter(line.split(" ", 6)[6] for line in lines)
    print(f"{len(lines)} positions written to {arguments.output}")
    for label in (label for label in LABELS if counts[label]):
        print(f"    {label:16s} {counts[label]:7d} ({100 * counts[label] / len(lines):.1f} %)")


if __name__ == "__main__":
    main(sys.argv[1:])
