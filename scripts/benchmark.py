"""Compare variants of the chess network trained in the same conditions.

Run from the root of the repository, for example:
    python3 -m scripts.benchmark --benchmark configs/benchmarks/activations.toml \\
        --train datasets/official_train.txt --validation datasets/official_validation.txt \\
        --train-size 60000 --validation-size 20000 --epochs 8 --json datasets/activations.json
A benchmark file names each variant with the values it changes in the base configuration:
    [variants]
    relu = { activation = "relu" }
    tanh = { activation = "tanh" }
Every variant trains for the same number of epochs, without early stopping, on the same part
of the data. The results hold the learning curves, the metrics and the confusion matrix on
the validation set, and the training and inference times.
"""

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from analyzer.errors import AnalyzerError
from analyzer.evaluate import evaluate_predictions
from analyzer.predict import check_chess_network, predict_classes
from analyzer.train import read_training_data
from my_torch import MyTorchError, train
from my_torch.config import parse_config
from scripts.search_hyperparameters import configuration, subset


def read_benchmark(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    """The title and the variants of a benchmark file."""
    with open(path, "rb") as file:
        data = tomllib.load(file)
    variants = data.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise MyTorchError(f"{path}: needs a [variants] table, one variant per line")
    for name, params in variants.items():
        if not isinstance(params, dict):
            raise MyTorchError(f"{path}: variant '{name}' must be a table of values")
    return str(data.get("title", path.stem)), variants


def measure(name: str, base: dict, params: dict, datasets: dict, arguments) -> dict:
    """Train one variant, then measure it on the validation set."""
    config = parse_config(configuration(base, params, arguments.epochs, None))
    network = config.create_network()
    classes = check_chess_network(network)
    if classes not in datasets:
        datasets[classes] = tuple(
            subset(*read_training_data(path, classes), size, arguments.seed)
            for path, size in ((arguments.train, arguments.train_size),
                               (arguments.validation, arguments.validation_size)))
    (inputs, targets), (validation_inputs, validation_targets) = datasets[classes]
    optimizer = config.create_optimizer()
    print(f"{name}:", flush=True)
    start = time.perf_counter()
    history = train(
        network, config.create_loss(), optimizer, inputs, targets, config.epochs,
        config.batch_size, config.shuffle, config.training_seed,
        validation=(validation_inputs, validation_targets), verbose=True, l2=config.l2,
        schedule=config.create_schedule(),
    )
    seconds = time.perf_counter() - start
    start = time.perf_counter()
    predicted = predict_classes(network, validation_inputs)
    inference = time.perf_counter() - start
    levels = evaluate_predictions(np.argmax(validation_targets, axis=1), predicted, classes)
    return {
        "config": name,
        "params": params,
        "history": history,
        "kept_epoch": len(history["loss"]),
        "train_accuracy": history["accuracy"][-1],
        "validation_accuracy": {len(level.classes): level.metrics.accuracy for level in levels},
        "validation_macro_f1": levels[0].metrics.macro.f1,
        "classes": list(classes),
        "confusion": levels[0].metrics.confusion.tolist(),
        "learning_rate": optimizer.learning_rate,
        "parameters": sum(layer.weights.size + layer.biases.size for layer in network.layers),
        "seconds": seconds,
        "seconds_per_epoch": seconds / len(history["loss"]),
        "inference_microseconds": inference / len(validation_inputs) * 1e6,
    }


def format_table(title: str, results: list[dict]) -> str:
    lines = [
        f"{title}",
        "",
        "| Variant | Parameters | Train acc. | Val. acc. 5 | Val. acc. 2 | Val. macro F1 "
        "| Time per epoch | Inference per position |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        accuracy = result["validation_accuracy"]
        lines.append(
            f"| {result['config']} | {result['parameters']} | "
            f"{result['train_accuracy']:.4f} | {accuracy.get(5, float('nan')):.4f} | "
            f"{accuracy[2]:.4f} | {result['validation_macro_f1']:.4f} | "
            f"{result['seconds_per_epoch']:.1f} s | {result['inference_microseconds']:.1f} µs |"
        )
    return "\n".join(lines)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", required=True, type=Path, help="TOML file of variants")
    parser.add_argument("--base", type=Path, default=Path("configs/my_torch_network.conf"),
                        help="configuration giving every value a variant does not change")
    parser.add_argument("--train", required=True, help="labelled training chessboards")
    parser.add_argument("--validation", required=True, help="labelled validation chessboards")
    parser.add_argument("--train-size", type=int, help="random part of the training set used")
    parser.add_argument("--validation-size", type=int,
                        help="random part of the validation set used")
    parser.add_argument("--epochs", type=int, default=8, help="epochs of every variant")
    parser.add_argument("--seed", type=int, default=0, help="seed of the parts of the data")
    parser.add_argument("--json", type=Path, help="also write the results here")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    arguments = parse_arguments(argv)
    try:
        title, variants = read_benchmark(arguments.benchmark)
        with open(arguments.base, "rb") as file:
            base = tomllib.load(file)
        datasets: dict = {}
        results = [measure(name, base, params, datasets, arguments)
                   for name, params in variants.items()]
    except (OSError, tomllib.TOMLDecodeError) as error:
        sys.exit(f"benchmark: {error}")
    except (AnalyzerError, MyTorchError) as error:
        sys.exit(f"benchmark: {error}")
    print()
    print(format_table(title, results))
    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps({"title": title, "results": results}, indent=2)
                                  + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
