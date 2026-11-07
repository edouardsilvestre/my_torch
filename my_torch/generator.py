"""my_torch_generator: create blank networks from configuration files."""

import re
import sys

from my_torch.errors import MyTorchError

PROGRAM = "my_torch_generator"
EXIT_SUCCESS = 0
EXIT_ERROR = 84
COUNT_PATTERN = re.compile(r"[0-9]+")
USAGE = (
    "USAGE\n"
    "        ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]\n"
    "\n"
    "DESCRIPTION\n"
    "        config_file_i    Configuration file containing description of a neural network "
    "we want to generate.\n"
    "        nb_i             Number of neural networks to generate based on the configuration "
    "file."
)


class UsageError(MyTorchError):
    """Error in the command line arguments, reported together with the usage."""


def parse_count(text: str, config_path: str) -> int:
    if not COUNT_PATTERN.fullmatch(text) or int(text) == 0:
        raise UsageError(
            f"invalid number of networks '{text}' for '{config_path}': "
            "expected a positive integer"
        )
    return int(text)


def parse_arguments(argv: list[str]) -> list[tuple[str, int]]:
    """Split the arguments into (configuration file, number of networks) pairs."""
    if not argv:
        raise UsageError("missing arguments")
    if len(argv) % 2:
        raise UsageError("each configuration file must be followed by a number of networks")
    return [(argv[i], parse_count(argv[i + 1], argv[i])) for i in range(0, len(argv), 2)]


def run(argv: list[str]) -> None:
    """Run the generator with the given command line arguments."""
    if argv == ["--help"]:
        print(USAGE)
        return
    parse_arguments(argv)


def main(argv: list[str]) -> int:
    try:
        run(argv)
    except UsageError as error:
        print(f"{PROGRAM}: {error}\n\n{USAGE}", file=sys.stderr)
        return EXIT_ERROR
    except MyTorchError as error:
        print(f"{PROGRAM}: {error}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as error:
        print(f"{PROGRAM}: unexpected error: {error}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_SUCCESS
