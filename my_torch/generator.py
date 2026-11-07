"""my_torch_generator: create blank networks from configuration files."""

import re
import sys
from collections import Counter
from pathlib import Path

from my_torch.config import Config, load_config
from my_torch.errors import MyTorchError
from my_torch.serialization import save_network

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


def output_name(config_path: str, index: int) -> str:
    """Name the file of the index-th network of a configuration, in the current directory."""
    return f"{Path(config_path).stem}_{index}.nn"


def network_seed(config: Config, index: int) -> int | None:
    """Give each network of a configuration its own seed, so that they all differ."""
    return None if config.seed is None else config.seed + index - 1


def check_names(requests: list[tuple[str, int]]) -> None:
    names = Counter(Path(config_path).stem for config_path, _ in requests)
    for name, count in names.items():
        if count > 1:
            raise MyTorchError(
                f"several configurations would save their networks as '{name}_<i>.nn'"
            )


def generate(config_path: str, config: Config, count: int) -> None:
    for index in range(1, count + 1):
        network = config.create_network(network_seed(config, index))
        save_network(
            output_name(config_path, index), network, config.create_loss(),
            config.create_optimizer(), config.training_settings(),
        )


def run(argv: list[str]) -> None:
    """Run the generator with the given command line arguments."""
    if argv == ["--help"]:
        print(USAGE)
        return
    requests = parse_arguments(argv)
    check_names(requests)
    configs = [load_config(config_path) for config_path, _ in requests]
    for (config_path, count), config in zip(requests, configs):
        generate(config_path, config, count)


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
