"""Split of a labelled dataset into training, validation and test parts."""

import random

from analyzer.chessfile import parse_line
from analyzer.errors import AnalyzerError
from analyzer.labels import LABELS
from dataset.board import Board
from dataset.generate import position_key, share_out

PARTS = ("train", "validation", "test")


def read_labelled_lines(lines: list[str]) -> list[tuple[str, str, str]]:
    """Return (position key, label, line) for each chessboard, checking that it has a label."""
    entries = []
    for number, text in enumerate(lines, start=1):
        if not text.strip():
            continue
        try:
            chessboard = parse_line(text, number)
            if chessboard.label is None:
                raise AnalyzerError("missing expected label after the FEN")
        except AnalyzerError as error:
            raise AnalyzerError(f"line {number}: {error}") from error
        key = position_key(Board.from_position(chessboard.position))
        entries.append((key, chessboard.label, " ".join(text.split())))
    return entries


def split_dataset(
    lines: list[str], ratios: tuple[float, ...], seed: int
) -> tuple[dict[str, list[str]], int]:
    """Split labelled lines into parts with the same share of each label.

    Duplicated positions are removed first, so that a position never ends in two parts.
    Return the lines of each part, and the number of duplicates removed.
    """
    seen: set[str] = set()
    groups: dict[str, list[str]] = {label: [] for label in LABELS}
    duplicates = 0
    for key, label, line in read_labelled_lines(lines):
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        groups[label].append(line)
    rng = random.Random(seed)
    parts: dict[str, list[str]] = {part: [] for part in PARTS}
    for label in LABELS:
        group = groups[label]
        rng.shuffle(group)
        start = 0
        for part, count in zip(PARTS, share_out(len(group), list(ratios))):
            parts[part].extend(group[start:start + count])
            start += count
    for part in PARTS:
        rng.shuffle(parts[part])
    return parts, duplicates
