"""Search the hyperparameters of the chess network on the validation set.

Run from the root of the repository, for example:
    python3 -m scripts.search_hyperparameters --space configs/search/space.toml \\
        --train datasets/official_train.txt --validation datasets/official_validation.txt \\
        --output datasets/search.jsonl random --trials 20
Each trial is appended to the output as soon as it ends: running the same command again goes
on where an interrupted search stopped. The test set must not be used here.
"""

import argparse
import copy
import json
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from analyzer.errors import AnalyzerError
from analyzer.predict import check_chess_network
from analyzer.train import read_training_data
from my_torch import MyTorchError, train
from my_torch.config import parse_config
from my_torch.search import Search, Trial, read_space
from scripts.compare_configs import kept_epoch

PARAMETERS = ("hidden", "activation", "dropout", "learning_rate", "batch_size", "l2")


def configuration(base: dict, params: dict[str, Any], epochs: int, patience: int) -> dict:
    """The content of a configuration file: the base one, with the searched values."""
    unknown = sorted(set(params) - set(PARAMETERS))
    if unknown:
        raise MyTorchError(f"unknown hyperparameter '{unknown[0]}', expected one of: "
                           f"{', '.join(PARAMETERS)}")
    data = copy.deepcopy(base)
    layers = data["network"]["layers"]
    hidden = params.get("hidden", [layer["size"] for layer in layers[:-1]])
    activation = params.get("activation", layers[0]["activation"])
    dropout = params.get("dropout", layers[0].get("dropout", 0.0))
    data["network"]["layers"] = [
        {"size": size, "activation": activation, "dropout": dropout} for size in hidden
    ] + [layers[-1]]
    training = data["training"]
    for key in ("batch_size", "l2"):
        if key in params:
            training[key] = params[key]
    training.update(epochs=epochs, patience=patience, monitor="val_accuracy")
    if "learning_rate" in params:
        data["optimizer"]["learning_rate"] = params["learning_rate"]
    return data


def subset(inputs: np.ndarray, targets: np.ndarray, size: int | None, seed: int):
    """A fixed random part of a dataset, to keep each trial short."""
    if size is None or size >= len(inputs):
        return inputs, targets
    chosen = np.sort(np.random.default_rng(seed).choice(len(inputs), size, replace=False))
    return inputs[chosen], targets[chosen]


class ChessEvaluation:
    """Train the chess network of a set of hyperparameters, and score it by its best
    validation accuracy, the one early stopping keeps."""

    def __init__(self, arguments: argparse.Namespace) -> None:
        with open(arguments.base, "rb") as file:
            self.base = tomllib.load(file)
        parse_config(self.base)
        self.arguments = arguments
        self.data: dict[tuple, tuple] = {}

    def datasets(self, classes: tuple[str, ...]) -> tuple:
        if classes not in self.data:
            arguments = self.arguments
            self.data[classes] = (
                subset(*read_training_data(arguments.train, classes), arguments.train_size,
                       arguments.seed),
                subset(*read_training_data(arguments.validation, classes),
                       arguments.validation_size, arguments.seed),
            )
        return self.data[classes]

    def __call__(self, params: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        config = parse_config(
            configuration(self.base, params, self.arguments.epochs, self.arguments.patience))
        network = config.create_network()
        (inputs, targets), validation = self.datasets(check_chess_network(network))
        start = time.perf_counter()
        history = train(
            network, config.create_loss(), config.create_optimizer(), inputs, targets,
            config.epochs, config.batch_size, config.shuffle, config.training_seed,
            validation=validation, l2=config.l2, patience=config.patience,
            monitor=config.monitor, schedule=config.create_schedule(),
        )
        kept = kept_epoch(history, config.patience, config.monitor)
        return history["val_accuracy"][kept - 1], {
            "kept_epoch": kept,
            "epochs_run": len(history["val_accuracy"]),
            "train_accuracy": history["accuracy"][kept - 1],
            "seconds": time.perf_counter() - start,
        }


def read_trials(path: Path) -> list[Trial]:
    """The trials saved by an earlier run, one JSON object per line."""
    if not path.exists():
        return []
    return [Trial(**json.loads(line)) for line in path.read_text().splitlines() if line]


def format_ranking(trials: list[Trial], count: int) -> str:
    names = list(trials[0].params) if trials else []
    lines = [
        "| Rank | Trial | Method | " + " | ".join(names)
        + " | Val. acc. | Kept epoch | Time |",
        "|---" * (len(names) + 6) + "|",
    ]
    for rank, trial in enumerate(trials[:count], 1):
        method = trial.method + (f" {trial.generation}" if trial.generation else "")
        values = " | ".join(json.dumps(trial.params.get(name)) for name in names)
        lines.append(
            f"| {rank} | {trial.number} | {method} | {values} | {trial.score:.4f} | "
            f"{trial.details.get('kept_epoch', '-')} | "
            f"{trial.details.get('seconds', 0) / 60:.1f} min |"
        )
    return "\n".join(lines)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", required=True, type=Path,
                        help="TOML file: the search space, in a [space] table")
    parser.add_argument("--base", type=Path, default=Path("configs/my_torch_network.conf"),
                        help="configuration giving every value that is not searched")
    parser.add_argument("--train", required=True, help="labelled training chessboards")
    parser.add_argument("--validation", required=True, help="labelled validation chessboards")
    parser.add_argument("--output", required=True, type=Path,
                        help="trials, one JSON line each, appended as they end")
    parser.add_argument("--train-size", type=int, help="random part of the training set used")
    parser.add_argument("--validation-size", type=int,
                        help="random part of the validation set used")
    parser.add_argument("--epochs", type=int, default=10, help="most epochs of a trial")
    parser.add_argument("--patience", type=int, default=2, help="early stopping of a trial")
    parser.add_argument("--seed", type=int, default=0, help="seed of the search and subsets")
    parser.add_argument("--top", type=int, default=10, help="rows of the ranking shown")
    methods = parser.add_subparsers(dest="method", required=True)
    random = methods.add_parser("random", help="random search")
    random.add_argument("--trials", type=int, default=20, help="sets drawn at random")
    genetic = methods.add_parser("genetic", help="genetic algorithm")
    genetic.add_argument("--population", type=int, default=8, help="sets per generation")
    genetic.add_argument("--generations", type=int, default=4, help="generations")
    genetic.add_argument("--mutation-rate", type=float, default=0.3,
                         help="chance of each hyperparameter to mutate")
    genetic.add_argument("--elite", type=int, default=2, help="best sets kept as they are")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    arguments = parse_arguments(argv)
    try:
        with open(arguments.space, "rb") as file:
            space = read_space(tomllib.load(file).get("space"))
        evaluation = ChessEvaluation(arguments)
        search = Search(space, evaluation, on_trial=lambda trial: save(arguments.output, trial))
        search.restore(read_trials(arguments.output))
        if arguments.method == "random":
            search.random(arguments.trials, arguments.seed)
        else:
            search.genetic(arguments.population, arguments.generations, arguments.seed,
                           arguments.mutation_rate, arguments.elite)
    except (OSError, tomllib.TOMLDecodeError, ValueError, TypeError) as error:
        sys.exit(f"search_hyperparameters: {error}")
    except (AnalyzerError, MyTorchError) as error:
        sys.exit(f"search_hyperparameters: {error}")
    print(format_ranking(search.ranking(), arguments.top))


def save(path: Path, trial: Trial) -> None:
    print(f"trial {trial.number} ({trial.method}): {json.dumps(trial.params)} -> "
          f"{trial.score:.4f}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(trial.__dict__) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
