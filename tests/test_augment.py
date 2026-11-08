"""Tests of the symmetries of the chessboard and of the training of a saved network."""

import pytest

from dataset.board import Board
from dataset.labels import label_position
from my_torch import Network, get_loss, get_optimizer, load_network, save_network
from scripts.augment_dataset import main as augment_main, mirror_files, swap_colors, symmetries
from scripts.continue_training import main as continue_main

POSITIONS = [
    ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", "Checkmate Black"),
    ("rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3", "Check White"),
    ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1", "Nothing"),
    ("8/8/8/8/8/8/8/k1K5 w - - 0 1", "Nothing"),
    ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "Nothing"),
    ("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1", "Checkmate White"),
]


def test_the_mirror_reverses_each_rank_and_drops_castling():
    fen, label = mirror_files("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
                              "Nothing")
    assert fen == "rnbkqbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBKQBNR b - e3 0 1"
    assert label == "Nothing"


def test_swapping_colors_turns_the_board_and_the_label():
    fen, label = swap_colors("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1", "Checkmate White")
    assert fen == "6k1/8/8/8/8/8/5PPP/r5K1 w - - 0 1"
    assert label == "Checkmate Black"
    fen, _ = swap_colors("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
                         "Nothing")
    assert fen.split()[1:4] == ["w", "KQkq", "d6"]


def test_twice_the_same_symmetry_gives_the_position_back():
    for fen, label in POSITIONS:
        assert swap_colors(*swap_colors(fen, label)) == (fen, label)
        mirrored = mirror_files(*mirror_files(fen, label))
        assert mirrored[0].split()[0] == fen.split()[0] and mirrored[1] == label


@pytest.mark.parametrize("fen, label", POSITIONS)
def test_our_rules_agree_with_every_symmetric_label(fen, label):
    images = symmetries(fen, label)
    assert len(images) == 4 and images[0] == (fen, label)
    assert len({image[0] for image in images}) > 1
    for image, expected in images:
        assert label_position(Board.from_fen(image)) == expected


def test_a_dataset_is_multiplied_by_four(tmp_path, capsys):
    (tmp_path / "in.txt").write_text("".join(f"{fen} {label}\n" for fen, label in POSITIONS))
    augment_main(["--input", str(tmp_path / "in.txt"), "--output", str(tmp_path / "out.txt"),
                  "--check", "24"])
    lines = (tmp_path / "out.txt").read_text().splitlines()
    assert len(lines) == 24
    assert lines[0] == f"{POSITIONS[0][0]} {POSITIONS[0][1]}"
    assert "24 positions from 6, 24 labels checked" in capsys.readouterr().out


def test_a_wrong_label_is_found(tmp_path):
    (tmp_path / "in.txt").write_text(f"{POSITIONS[0][0]} Check Black\n")
    with pytest.raises(SystemExit) as error:
        augment_main(["--input", str(tmp_path / "in.txt"), "--output",
                      str(tmp_path / "out.txt"), "--check", "4"])
    assert "4 of the checked positions have a wrong label" in str(error.value.code)
    assert not (tmp_path / "out.txt").exists()


def test_an_invalid_line_is_refused(tmp_path):
    (tmp_path / "in.txt").write_text(f"{POSITIONS[0][0]} Stalemate\n")
    with pytest.raises(SystemExit) as error:
        augment_main(["--input", str(tmp_path / "in.txt"), "--output",
                      str(tmp_path / "out.txt")])
    assert "in.txt:1: unknown label 'Stalemate'" in str(error.value.code)


@pytest.fixture
def trained(tmp_path):
    network = Network(769, [(16, "relu"), (5, "identity")], seed=0)
    optimizer = get_optimizer("adam", learning_rate=0.01)
    save_network(tmp_path / "net.nn", network, get_loss("cross_entropy"), optimizer,
                 {"epochs": 3, "batch_size": 4})
    (tmp_path / "boards.txt").write_text("".join(f"{fen} {label}\n" for fen, label in POSITIONS))
    (tmp_path / "settings.toml").write_text(
        "[training]\nepochs = 5\nbatch_size = 2\nseed = 0\npatience = 2\n"
        'monitor = "val_accuracy"\n\n[optimizer]\nlearning_rate = 0.002\n')
    return tmp_path


def run_continue(path, *extra):
    continue_main(["--network", str(path / "net.nn"), "--settings", str(path / "settings.toml"),
                   "--train", str(path / "boards.txt"), "--save", str(path / "out.nn"), *extra])


def test_the_training_goes_on_with_the_new_settings(trained, capsys):
    before = load_network(trained / "net.nn")
    run_continue(trained, "--validation", str(trained / "boards.txt"))
    after = load_network(trained / "out.nn")
    assert after.optimizer.learning_rate == 0.002
    assert after.settings["epochs"] == 5 and after.settings["patience"] == 2
    assert after.optimizer.steps > 0 and before.optimizer.steps == 0
    assert (after.network.layers[0].weights != before.network.layers[0].weights).any()
    assert "val_accuracy" in capsys.readouterr().out


def test_the_memory_of_the_optimizer_is_kept(trained):
    (trained / "settings.toml").write_text("[training]\nepochs = 1\nbatch_size = 6\n")
    run_continue(trained)
    first = load_network(trained / "out.nn").optimizer.steps
    continue_main(["--network", str(trained / "out.nn"), "--settings",
                   str(trained / "settings.toml"), "--train", str(trained / "boards.txt"),
                   "--save", str(trained / "again.nn")])
    assert load_network(trained / "again.nn").optimizer.steps == first + 1


@pytest.mark.parametrize("content, message", [
    ("[network]\n", "unknown table 'network'"),
    ("[training]\nepochs = 0\nbatch_size = 2\n", "settings.epochs"),
    ("[training]\nepochs = 1\nbatch_size = 2\n[optimizer]\nname = 'sgd'\n",
     "may only give a learning_rate"),
    ("[training]\nepochs = 1\nbatch_size = 2\npatience = 1\n", "needs a validation set"),
])
def test_invalid_settings_stop_with_a_message(trained, content, message):
    (trained / "settings.toml").write_text(content)
    with pytest.raises(SystemExit) as error:
        run_continue(trained)
    assert message in str(error.value.code)
    assert not (trained / "out.nn").exists()
