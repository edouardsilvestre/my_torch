"""Command line arguments of my_torch_analyzer."""

import os
from dataclasses import dataclass

from analyzer.errors import AnalyzerError

PREDICT = "predict"
TRAIN = "train"
USAGE = (
    "USAGE\n"
    "        ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE\n"
    "\n"
    "DESCRIPTION\n"
    "        --train        Launch the neural network in training mode. Each chessboard in FILE "
    "must contain inputs to send to the neural network in FEN notation and the expected output "
    "separated by space. If specified, the newly trained neural network will be saved in "
    "SAVEFILE. Otherwise, it will be saved in the original LOADFILE.\n"
    "\n"
    "        --predict      Launch the neural network in prediction mode. Each chessboard in "
    "FILE must contain inputs to send to the neural network in FEN notation, and optionally an "
    "expected output.\n"
    "\n"
    "        --save         Save neural network into SAVEFILE. Only works in train mode.\n"
    "\n"
    "        LOADFILE       File containing an artificial neural network\n"
    "        CHESSFILE      File containing chessboards"
)


class UsageError(AnalyzerError):
    """Error in the command line arguments, reported together with the usage."""


@dataclass(frozen=True)
class Arguments:
    """Validated command line of the analyzer."""

    mode: str
    load_file: str
    chess_file: str
    save_file: str | None = None


def read_mode(argument: str, mode: str | None) -> str:
    new_mode = argument[2:]
    if mode == new_mode:
        raise UsageError(f"{argument} is given twice")
    if mode is not None:
        raise UsageError("--predict and --train cannot be used together")
    return new_mode


def parse_arguments(argv: list[str]) -> Arguments:
    """Read the mode, the optional save file and the two files of the command line."""
    if not argv:
        raise UsageError("missing arguments")
    mode = None
    save_file = None
    files = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument in ("--predict", "--train"):
            mode = read_mode(argument, mode)
        elif argument == "--save":
            if save_file is not None:
                raise UsageError("--save is given twice")
            if index + 1 == len(argv) or argv[index + 1].startswith("-"):
                raise UsageError("--save must be followed by a file name")
            index += 1
            save_file = argv[index]
        elif argument == "--help":
            raise UsageError("--help must be used alone")
        elif argument.startswith("-") and argument != "-":
            raise UsageError(f"unknown option '{argument}'")
        else:
            files.append(argument)
        index += 1
    if mode is None:
        raise UsageError("missing mode: --predict or --train")
    if save_file is not None and mode != TRAIN:
        raise UsageError("--save only works with --train")
    if len(files) != 2:
        raise UsageError(f"expected LOADFILE and CHESSFILE, got {len(files)} file(s)")
    return Arguments(mode, files[0], files[1], save_file)


def check_readable(path: str, name: str) -> None:
    if not os.path.exists(path):
        raise AnalyzerError(f"cannot read {name} '{path}': no such file")
    if os.path.isdir(path):
        raise AnalyzerError(f"cannot read {name} '{path}': is a directory")
    if not os.access(path, os.R_OK):
        raise AnalyzerError(f"cannot read {name} '{path}': permission denied")


def check_writable(path: str) -> None:
    """Check that the network can be saved to a path, before any long training."""
    directory = os.path.dirname(os.path.abspath(path))
    if os.path.isdir(path):
        raise AnalyzerError(f"cannot save the network to '{path}': is a directory")
    if not os.path.isdir(directory):
        raise AnalyzerError(f"cannot save the network to '{path}': no such directory")
    if not os.access(directory, os.W_OK | os.X_OK):
        raise AnalyzerError(f"cannot save the network to '{path}': permission denied")


def check_files(arguments: Arguments) -> None:
    """Check the files of the command line before starting the work."""
    check_readable(arguments.load_file, "LOADFILE")
    check_readable(arguments.chess_file, "CHESSFILE")
    if arguments.mode == TRAIN:
        check_writable(arguments.save_file or arguments.load_file)
