"""Reading of chessboard files: one FEN per line, optionally followed by its label."""

from dataclasses import dataclass

from analyzer.errors import AnalyzerError
from analyzer.fen import Position, parse_fen
from analyzer.labels import check_label

FEN_FIELDS = 6


@dataclass(frozen=True)
class Chessboard:
    """A chessboard of the file, with its expected label when the line gives one."""

    line_number: int
    position: Position
    label: str | None


def parse_line(text: str, line_number: int) -> Chessboard:
    """Read a line made of the 6 fields of a FEN, then of an optional label."""
    fields = text.split()
    position = parse_fen(" ".join(fields[:FEN_FIELDS]))
    label = " ".join(fields[FEN_FIELDS:])
    return Chessboard(line_number, position, check_label(label) if label else None)


def read_lines(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8-sig") as file:
            return file.read().splitlines()
    except FileNotFoundError as error:
        raise AnalyzerError(f"cannot read chessboards: no such file '{path}'") from error
    except OSError as error:
        raise AnalyzerError(f"cannot read chessboards from '{path}': {error.strerror}") from error
    except UnicodeDecodeError as error:
        raise AnalyzerError(f"cannot read chessboards from '{path}': not a text file") from error


def read_chessfile(path: str, require_labels: bool = False) -> list[Chessboard]:
    """Read every chessboard of a file, in order, skipping the empty lines.

    With require_labels, each line must give the expected label, as in training mode.
    """
    chessboards = []
    for line_number, text in enumerate(read_lines(path), start=1):
        if not text.strip():
            continue
        try:
            chessboard = parse_line(text, line_number)
            if require_labels and chessboard.label is None:
                raise AnalyzerError("missing expected label after the FEN")
        except AnalyzerError as error:
            raise AnalyzerError(f"{path}:{line_number}: {error}") from error
        chessboards.append(chessboard)
    return chessboards
