"""Entry point of my_torch_analyzer."""

import sys

from analyzer.errors import AnalyzerError
from my_torch import MyTorchError

EXIT_SUCCESS = 0
EXIT_ERROR = 84


def run(argv: list[str]) -> None:
    """Run the analyzer with the given command line arguments."""


def main(argv: list[str]) -> int:
    try:
        run(argv)
    except (AnalyzerError, MyTorchError) as error:
        print(f"my_torch_analyzer: {error}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as error:
        print(f"my_torch_analyzer: unexpected error: {error}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_SUCCESS
