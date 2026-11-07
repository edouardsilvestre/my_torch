"""Tests of the labelling of positions and of the generation of the dataset."""

from collections import Counter

import pytest

import dataset.generate as generation
from analyzer.chessfile import read_chessfile
from dataset.board import Board
from dataset.generate import LABELS, generate_dataset, label_shares, position_key, split_size
from dataset.labels import label_position
from scripts.generate_dataset import main as generate_main

NOTHING_A = "8/8/8/8/8/8/8/k1K5 w - - 0 1"
NOTHING_B = "8/8/8/8/8/8/k7/2K5 w - - 0 1"
NOTHING_C = "8/8/8/8/8/k7/8/2K5 w - - 0 1"
NOTHING_D = "8/8/8/8/k7/8/8/2K5 w - - 0 1"
NOTHING_E = "8/8/8/k7/8/8/8/2K5 w - - 0 1"
ILLEGAL = "8/8/8/8/k7/8/2Kq4/8 b - - 0 1"

SIZE = 30


@pytest.mark.parametrize("fen, expected", [
    ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", "Checkmate Black"),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Nothing"),
    ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1", "Nothing"),
    ("rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3", "Check White"),
    ("8/8/8/8/8/8/8/k1K5 w - - 0 1", "Nothing"),
    ("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4", "Checkmate White"),
    ("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1", "Checkmate White"),
    ("R5k1/5pp1/7p/8/8/8/8/6K1 b - - 0 1", "Check White"),
    ("4k3/8/8/8/8/8/8/4K2r w - - 0 1", "Check Black"),
    ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "Nothing"),
    ("8/3Q4/7R/k7/1Pp5/3N4/8/7K b - b3 0 1", "Check White"),
    ("8/3Q4/7R/k7/1Pp5/3N4/8/7K b - - 0 1", "Checkmate White"),
    ("8/8/8/8/k7/8/2Kq4/8 b - - 0 1", None),
])
def test_labels_of_known_positions(fen, expected):
    assert label_position(Board.from_fen(fen)) == expected


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(SIZE, 0)


def labels_of(lines):
    return Counter(line.split(" ", 6)[6] for line in lines)


def test_default_shares_are_one_third_each(dataset):
    lines, _ = dataset
    assert len(lines) == SIZE
    assert labels_of(lines) == {"Nothing": 10, "Check White": 5, "Check Black": 5,
                                "Checkmate White": 5, "Checkmate Black": 5}


@pytest.mark.parametrize("size, shares, expected", [
    (30, (1, 1, 1), [10, 5, 5, 5, 5]),
    (30, (4, 1, 1), [20, 3, 3, 2, 2]),
    (7, (1, 1, 1), [3, 1, 1, 1, 1]),
    (10, (1, 0, 0), [10, 0, 0, 0, 0]),
    (11, (0, 1, 1), [0, 3, 3, 3, 2]),
])
def test_split_size_always_adds_up_to_the_size(size, shares, expected):
    counts = split_size(size, label_shares(*shares))
    assert [counts[label] for label in LABELS] == expected
    assert sum(counts.values()) == size


def test_custom_shares():
    lines, _ = generate_dataset(12, 1, label_shares(4, 1, 1))
    assert labels_of(lines) == {"Nothing": 8, "Check White": 1, "Check Black": 1,
                                "Checkmate White": 1, "Checkmate Black": 1}


def test_every_line_has_the_label_given_by_the_rules(dataset, tmp_path):
    lines, _ = dataset
    path = tmp_path / "train.txt"
    path.write_text("\n".join(lines) + "\n")
    for chessboard, line in zip(read_chessfile(path, require_labels=True), lines):
        fen = " ".join(line.split()[:6])
        assert label_position(Board.from_fen(fen)) == chessboard.label


def test_no_duplicate_positions(dataset):
    lines, _ = dataset
    keys = [position_key(Board.from_fen(" ".join(line.split()[:6]))) for line in lines]
    assert len(set(keys)) == len(keys)


def test_lines_are_shuffled(dataset):
    lines, _ = dataset
    labels = [line.split(" ", 6)[6] for line in lines]
    assert labels != sorted(labels, key=LABELS.index)


def test_same_seed_gives_the_same_dataset(dataset):
    assert generate_dataset(SIZE, 0) == dataset
    assert generate_dataset(10, 1)[0] != generate_dataset(10, 2)[0]


def test_position_key_ignores_the_counters():
    first = Board.from_fen("8/8/8/8/8/8/8/k1K5 w - - 0 1")
    second = Board.from_fen("8/8/8/8/8/8/8/k1K5 w - - 12 40")
    assert position_key(first) == position_key(second)


def test_script_writes_the_same_file_for_the_same_arguments(tmp_path, capsys):
    arguments = ["--size", "10", "--seed", "2"]
    generate_main(arguments + ["--output", str(tmp_path / "out" / "a.txt")])
    generate_main(arguments + ["--output", str(tmp_path / "b.txt")])
    first = (tmp_path / "out" / "a.txt").read_bytes()
    assert first == (tmp_path / "b.txt").read_bytes()
    assert len(first.decode().splitlines()) == 10
    assert "10 positions from" in capsys.readouterr().out


@pytest.mark.parametrize("arguments", [
    ["--size", "0"],
    [],
    ["--size", "10", "--shares", "1:1"],
    ["--size", "10", "--shares", "0:0:0"],
    ["--size", "10", "--shares=-1:1:1"],
    ["--size", "10", "--shares", "a:b:c"],
])
def test_script_refuses_invalid_arguments(tmp_path, arguments, capsys):
    output = tmp_path / "out.txt"
    with pytest.raises(SystemExit) as error:
        generate_main(arguments + ["--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def prepared_games(monkeypatch, games):
    """Replace the random games by prepared ones, given as lists of FEN."""
    queue = iter(games)
    monkeypatch.setattr(generation, "play_random_game",
                        lambda rng: (Board.from_fen(fen) for fen in next(queue)))


def fens_of(lines):
    return [" ".join(line.split()[:6]) for line in lines]


def test_a_position_seen_in_an_earlier_game_is_not_kept_again(monkeypatch):
    prepared_games(monkeypatch, [[NOTHING_A], [NOTHING_A], [NOTHING_B], [NOTHING_C]])
    lines, games = generate_dataset(3, 0, label_shares(1, 0, 0))
    assert sorted(fens_of(lines)) == sorted([NOTHING_A, NOTHING_B, NOTHING_C])
    assert games == 4


def test_illegal_positions_are_never_kept(monkeypatch):
    prepared_games(monkeypatch, [[ILLEGAL, NOTHING_A], [ILLEGAL], [NOTHING_B]])
    lines, _ = generate_dataset(2, 0, label_shares(1, 0, 0))
    assert sorted(fens_of(lines)) == sorted([NOTHING_A, NOTHING_B])


def test_a_game_gives_at_most_one_position_per_label(monkeypatch):
    prepared_games(monkeypatch, [[NOTHING_A, NOTHING_B, NOTHING_C], [NOTHING_D], [NOTHING_E]])
    lines, games = generate_dataset(3, 0, label_shares(1, 0, 0))
    fens = fens_of(lines)
    assert games == 3
    assert NOTHING_D in fens and NOTHING_E in fens
    assert len({NOTHING_A, NOTHING_B, NOTHING_C} & set(fens)) == 1


def test_script_uses_its_seed(tmp_path):
    generate_main(["--size", "10", "--seed", "3", "--output", str(tmp_path / "a.txt")])
    lines = (tmp_path / "a.txt").read_text().splitlines()
    assert lines == generate_dataset(10, 3)[0]
    assert lines != generate_dataset(10, 2)[0]


def test_a_single_chunk_is_the_dataset_generated_from_the_seed_itself():
    from dataset.generate import generate_chunk, split_size
    assert generate_dataset(12, 5) == generate_chunk(split_size(12, label_shares(1, 1, 1)), 5)


def test_the_number_of_jobs_does_not_change_the_dataset():
    sequential = generate_dataset(24, 3, jobs=1, chunk_size=8)
    assert generate_dataset(24, 3, jobs=2, chunk_size=8) == sequential
    assert generate_dataset(24, 3, jobs=3, chunk_size=8) == sequential
    assert generate_dataset(24, 3, jobs=4, chunk_size=8) == sequential
    lines, _ = sequential
    assert labels_of(lines) == {"Nothing": 8, "Check White": 4, "Check Black": 4,
                                "Checkmate White": 4, "Checkmate Black": 4}
    assert len(set(fens_of(lines))) == len(lines)


def test_a_position_found_by_two_chunks_is_replaced(monkeypatch):
    prepared_games(monkeypatch, [[NOTHING_A], [NOTHING_B], [NOTHING_A], [NOTHING_C],
                                 [NOTHING_A], [NOTHING_D]])
    lines, games = generate_dataset(4, 0, label_shares(1, 0, 0), chunk_size=2)
    assert sorted(fens_of(lines)) == sorted([NOTHING_A, NOTHING_B, NOTHING_C, NOTHING_D])
    assert games == 6


def test_script_refuses_zero_jobs(tmp_path):
    with pytest.raises(SystemExit) as error:
        generate_main(["--size", "10", "--jobs", "0", "--output", str(tmp_path / "out.txt")])
    assert error.value.code == 2


def test_each_chunk_comes_from_its_own_seed():
    from dataset.generate import generate_chunk, share_out, split_size
    targets = split_size(24, label_shares(1, 1, 1))
    lines = set(generate_dataset(24, 3, chunk_size=8)[0])
    for index, chunk_seed in enumerate([3, "3-1", "3-2"]):
        plan = {label: share_out(targets[label], [1, 1, 1])[index] for label in LABELS}
        assert set(generate_chunk(plan, chunk_seed)[0]) <= lines
