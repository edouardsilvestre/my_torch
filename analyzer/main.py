"""Entry point of my_torch_analyzer."""

import sys

from analyzer.arguments import USAGE, UsageError, parse_arguments
from analyzer.errors import AnalyzerError
from my_torch import MyTorchError

EXIT_SUCCESS = 0
EXIT_ERROR = 84


def run(argv: list[str]) -> None:
    """Run the analyzer with the given command line arguments."""
    if argv == ["--help"]:
        print(USAGE)
        return
    arguments = parse_arguments(argv)
    raise AnalyzerError(f"--{arguments.mode} is not available yet")


def main(argv: list[str]) -> int:
    try:
        run(argv)
    except UsageError as error:
        print(f"my_torch_analyzer: {error}\n\n{USAGE}", file=sys.stderr)
        return EXIT_ERROR
    except (AnalyzerError, MyTorchError) as error:
        print(f"my_torch_analyzer: {error}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as error:
        print(f"my_torch_analyzer: unexpected error: {error}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_SUCCESS
