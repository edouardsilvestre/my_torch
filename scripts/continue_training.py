"""Go on training a saved chess network, with new training settings.

Run from the root of the repository, for example:
    python3 -m scripts.continue_training --network my_torch_network.nn \\
        --settings configs/finetune.toml --train datasets/official_train_x4.txt \\
        --validation datasets/official_validation.txt --save datasets/finetuned.nn
The network keeps its weights and the memory of its optimizer. The settings file holds a
[training] table, as in a configuration file but without the loss, and may give a new
learning_rate in an [optimizer] table. The saved network keeps these new settings.
"""

import argparse
import sys
import tomllib
from pathlib import Path

from analyzer.errors import AnalyzerError
from analyzer.predict import check_chess_network
from analyzer.train import read_training_data
from my_torch import MyTorchError, load_network, save_network, train
from my_torch.config import check_training_settings, read_number
from my_torch.schedules import schedule_from_settings


def read_settings(path: Path) -> tuple[dict, float | None]:
    with open(path, "rb") as file:
        data = tomllib.load(file)
    unknown = sorted(set(data) - {"training", "optimizer"})
    if unknown:
        raise MyTorchError(f"{path}: unknown table '{unknown[0]}'")
    settings = check_training_settings(data.get("training"))
    optimizer = data.get("optimizer", {})
    if not isinstance(optimizer, dict) or set(optimizer) - {"learning_rate"}:
        raise MyTorchError(f"{path}: [optimizer] may only give a learning_rate")
    rate = optimizer.get("learning_rate")
    return settings, None if rate is None else read_number(rate, "optimizer.learning_rate")


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--network", required=True, help=".nn file to start from")
    parser.add_argument("--settings", required=True, type=Path, help="TOML training settings")
    parser.add_argument("--train", required=True, help="labelled training chessboards")
    parser.add_argument("--validation", help="labelled validation chessboards")
    parser.add_argument("--save", required=True, help=".nn file to write")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    arguments = parse_arguments(argv)
    try:
        settings, rate = read_settings(arguments.settings)
        saved = load_network(arguments.network)
        if saved.loss is None or saved.optimizer is None:
            raise MyTorchError(f"{arguments.network}: the loss and the optimizer are needed")
        if rate is not None:
            saved.optimizer.learning_rate = rate
        classes = check_chess_network(saved.network)
        inputs, targets = read_training_data(arguments.train, classes)
        validation = (read_training_data(arguments.validation, classes)
                      if arguments.validation else None)
        train(saved.network, saved.loss, saved.optimizer, inputs, targets, settings["epochs"],
              settings["batch_size"], settings["shuffle"], settings["seed"],
              validation=validation, verbose=True, l2=settings["l2"],
              patience=settings["patience"], monitor=settings["monitor"],
              schedule=schedule_from_settings(settings["schedule"]))
        save_network(arguments.save, saved.network, saved.loss, saved.optimizer, settings)
    except (OSError, tomllib.TOMLDecodeError, AnalyzerError, MyTorchError) as error:
        sys.exit(f"continue_training: {error}")


if __name__ == "__main__":
    main(sys.argv[1:])
