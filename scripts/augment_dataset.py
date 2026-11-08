"""Multiply a labelled dataset by the symmetries of the chessboard.

Run from the root of the repository, for example:
    python3 -m scripts.augment_dataset --input datasets/official_all.txt \\
        --output datasets/official_all_x4.txt
Each position gives four: itself, its mirror image from the a-file to the h-file, the same
position with the colors swapped, and both. None of them changes whether a king is in check
or mated, so the labels are known exactly; a random sample is checked with our own rules.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from analyzer.errors import AnalyzerError
from analyzer.fen import parse_fen
from analyzer.labels import check_label
from dataset.board import Board
from dataset.labels import label_position

FILES = "abcdefgh"
SWAPPED_COLOR = {"White": "Black", "Black": "White"}


def mirror_files(fen: str, label: str) -> tuple[str, str]:
    """Mirror the board from the a-file to the h-file.

    Castling is the only rule that is not symmetric this way, and it never gets a king out
    of check: castling rights are dropped. The en passant square is mirrored too.
    """
    board, side, _castling, en_passant, halfmoves, moves = fen.split()
    ranks = "/".join(rank[::-1] for rank in board.split("/"))
    if en_passant != "-":
        en_passant = FILES[7 - FILES.index(en_passant[0])] + en_passant[1]
    return " ".join([ranks, side, "-", en_passant, halfmoves, moves]), label


def swap_colors(fen: str, label: str) -> tuple[str, str]:
    """Swap the colors: the board is turned upside down, white pieces become black ones,
    and the other side moves. A check by White becomes a check by Black."""
    board, side, castling, en_passant, halfmoves, moves = fen.split()
    ranks = "/".join(rank.swapcase() for rank in reversed(board.split("/")))
    side = "b" if side == "w" else "w"
    if castling != "-":
        castling = "".join(sorted(castling.swapcase()))
    if en_passant != "-":
        en_passant = en_passant[0] + ("6" if en_passant[1] == "3" else "3")
    words = label.split()
    if len(words) == 2:
        label = f"{words[0]} {SWAPPED_COLOR[words[1]]}"
    return " ".join([ranks, side, castling, en_passant, halfmoves, moves]), label


def symmetries(fen: str, label: str) -> list[tuple[str, str]]:
    """The position and its three symmetric images."""
    mirrored = mirror_files(fen, label)
    return [(fen, label), mirrored, swap_colors(fen, label), swap_colors(*mirrored)]


def read_lines(path: Path) -> list[tuple[str, str]]:
    """The positions of a labelled file, as (FEN, label) texts, each one checked."""
    lines = []
    for number, text in enumerate(path.read_text().splitlines(), 1):
        if not text.strip():
            continue
        words = text.split()
        fen, label = " ".join(words[:6]), " ".join(words[6:])
        try:
            parse_fen(fen)
            check_label(label)
        except AnalyzerError as error:
            raise AnalyzerError(f"{path}:{number}: {error}") from error
        lines.append((fen, label))
    return lines


def check_labels(lines: list[tuple[str, str]], count: int, seed: int) -> int:
    """Label a random sample with our rules of chess, and return how many disagree."""
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(lines), min(count, len(lines)), replace=False)
    return sum(label_position(Board.from_fen(lines[index][0])) != lines[index][1]
               for index in picked)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path, help="labelled chessboards")
    parser.add_argument("--output", required=True, type=Path, help="file to write")
    parser.add_argument("--check", type=int, default=2000,
                        help="positions of the output checked with our rules")
    parser.add_argument("--seed", type=int, default=0, help="seed of the checked sample")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    arguments = parse_arguments(argv)
    try:
        originals = read_lines(arguments.input)
    except (OSError, AnalyzerError) as error:
        sys.exit(f"augment_dataset: {error}")
    lines = [line for fen, label in originals for line in symmetries(fen, label)]
    wrong = check_labels(lines, arguments.check, arguments.seed)
    if wrong:
        sys.exit(f"augment_dataset: {wrong} of the checked positions have a wrong label")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("".join(f"{fen} {label}\n" for fen, label in lines))
    print(f"{arguments.output}: {len(lines)} positions from {len(originals)}, "
          f"{min(arguments.check, len(lines))} labels checked with our rules")


if __name__ == "__main__":
    main(sys.argv[1:])
