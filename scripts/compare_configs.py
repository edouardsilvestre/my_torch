"""Train several configurations and compare them on the validation set.

Run from the root of the repository, for example:
    python3 -m scripts.compare_configs --train datasets/all_train.txt \\
        --validation datasets/all_validation.txt configs/candidates/*.conf
The test set must not be used here: it only measures the network chosen in the end.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from analyzer.errors import AnalyzerError
from analyzer.evaluate import evaluate_predictions
from analyzer.predict import check_chess_network, predict_classes
from analyzer.train import read_training_data
from my_torch import MyTorchError, load_config, save_network, train


def kept_epoch(history: dict, patience: int | None, monitor: str) -> int:
    """The epoch whose weights the network keeps: the last one, or the best one with early
    stopping. The first of equal values is the best, as in train()."""
    if patience is None:
        return len(history[monitor])
    choose = np.argmin if monitor == "val_loss" else np.argmax
    return int(choose(history[monitor])) + 1


def compare(config_path: Path, train_path: str, validation_path: str, cache: dict,
            save_dir: Path | None = None) -> dict:
    """Train the network of a configuration, and measure it on the validation set.

    With save_dir, the trained network, the one of the kept epoch, is also saved there under
    the name of its configuration, ready for --train or an evaluation."""
    config = load_config(config_path)
    network = config.create_network()
    classes = check_chess_network(network)
    if classes not in cache:
        cache[classes] = (read_training_data(train_path, classes),
                          read_training_data(validation_path, classes))
    (inputs, targets), (validation_inputs, validation_targets) = cache[classes]
    start = time.perf_counter()
    loss, optimizer = config.create_loss(), config.create_optimizer()
    history = train(
        network, loss, optimizer, inputs, targets,
        config.epochs, config.batch_size, config.shuffle, config.training_seed,
        validation=(validation_inputs, validation_targets), verbose=True, l2=config.l2,
        patience=config.patience, monitor=config.monitor, schedule=config.create_schedule(),
    )
    seconds = time.perf_counter() - start
    if save_dir is not None:
        save_network(save_dir / f"{config_path.stem}.nn", network, loss, optimizer,
                     config.training_settings())
    levels = evaluate_predictions(np.argmax(validation_targets, axis=1),
                                  predict_classes(network, validation_inputs), classes)
    best_epoch = int(np.argmax(history["val_accuracy"])) + 1
    return {
        "config": str(config_path),
        "seconds": seconds,
        "kept_epoch": (kept := kept_epoch(history, config.patience, config.monitor)),
        "train_accuracy": history["accuracy"][kept - 1],
        "validation_accuracy": {len(level.classes): level.metrics.accuracy for level in levels},
        "validation_macro_f1": levels[0].metrics.macro.f1,
        "best_epoch": best_epoch,
        "best_validation_accuracy": history["val_accuracy"][best_epoch - 1],
        "history": history,
    }


def format_table(results: list[dict]) -> str:
    lines = [
        "| Configuration | Kept epoch | Train acc. | Val. acc. 5 | Val. acc. 3 | Val. acc. 2 "
        "| Val. macro F1 | Best epoch (val. acc.) | Time |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        accuracy = result["validation_accuracy"]
        cells = [f"{accuracy[size]:.4f}" if size in accuracy else "-" for size in (5, 3, 2)]
        lines.append(
            f"| {Path(result['config']).stem} | {result['kept_epoch']} | "
            f"{result['train_accuracy']:.4f} | "
            f"{' | '.join(cells)} | {result['validation_macro_f1']:.4f} | "
            f"{result['best_epoch']} ({result['best_validation_accuracy']:.4f}) | "
            f"{result['seconds'] / 60:.1f} min |"
        )
    return "\n".join(lines)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True, help="labelled training chessboards")
    parser.add_argument("--validation", required=True, help="labelled validation chessboards")
    parser.add_argument("--json", type=Path, help="also write the results and histories here")
    parser.add_argument("--save", type=Path, metavar="DIR",
                        help="also save each trained network there, as <configuration>.nn")
    parser.add_argument("configs", nargs="+", type=Path, help="configuration files to compare")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    arguments = parse_arguments(argv)
    results, cache = [], {}
    if arguments.save:
        arguments.save.mkdir(parents=True, exist_ok=True)
    for config_path in arguments.configs:
        print(f"{config_path}:", flush=True)
        try:
            result = compare(config_path, arguments.train, arguments.validation, cache,
                             arguments.save)
        except (AnalyzerError, MyTorchError) as error:
            sys.exit(f"compare_configs: {config_path}: {error}")
        results.append(result)
        print(f"{config_path}: validation accuracy {result['validation_accuracy']} "
              f"in {result['seconds'] / 60:.1f} min", flush=True)
    print()
    print(format_table(results))
    if arguments.json:
        arguments.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
