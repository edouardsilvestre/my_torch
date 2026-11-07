"""Tests of the split of a dataset into training, validation and test parts."""

from collections import Counter

import pytest

from analyzer.errors import AnalyzerError
from analyzer.labels import LABELS
from dataset.board import Board
from dataset.generate import share_out
from dataset.split import PARTS, split_dataset
from scripts.split_dataset import main as split_main

def king_positions():
    """Yield distinct legal positions: two kings on every pair of squares that do not touch."""
    for black in range(64):
        for white in range(64):
            if max(abs(black // 8 - white // 8), abs(black % 8 - white % 8)) > 1:
                squares = [None] * 64
                squares[black], squares[white] = "k", "K"
                yield Board(squares, "w", "", None, 0, 1).to_fen()


def make_lines(counts):
    """Build labelled lines: distinct positions, with the requested number of each label."""
    positions = king_positions()
    return [f"{next(positions)} {label}" for label, count in counts.items() for _ in range(count)]


BALANCED = {"Nothing": 40, "Check White": 20, "Check Black": 20,
            "Checkmate White": 20, "Checkmate Black": 20}


def labels_of(lines):
    return Counter(line.split(" ", 6)[6] for line in lines)


def keys_of(lines):
    return {" ".join(line.split()[:4]) for line in lines}


def test_positions_of_the_test_data_are_distinct():
    lines = make_lines(BALANCED)
    assert len(keys_of(lines)) == len(lines) == 120


def test_each_part_has_the_same_share_of_each_label():
    parts, duplicates = split_dataset(make_lines(BALANCED), (80, 10, 10), 0)
    assert duplicates == 0
    assert labels_of(parts["train"]) == {label: 4 * BALANCED[label] // 5 for label in LABELS
                                         if label in BALANCED}
    for part in ("validation", "test"):
        assert labels_of(parts[part]) == {label: count // 10 for label, count in BALANCED.items()}


def test_parts_are_disjoint_and_keep_every_line():
    lines = make_lines(BALANCED)
    parts, _ = split_dataset(lines, (80, 10, 10), 0)
    keys = [keys_of(parts[part]) for part in PARTS]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    assert sorted(line for part in PARTS for line in parts[part]) == sorted(lines)


def test_parts_are_shuffled():
    parts, _ = split_dataset(make_lines(BALANCED), (80, 10, 10), 0)
    labels = [line.split(" ", 6)[6] for line in parts["train"]]
    assert labels != sorted(labels, key=LABELS.index)


def test_same_seed_gives_the_same_parts():
    lines = make_lines(BALANCED)
    assert split_dataset(lines, (80, 10, 10), 0) == split_dataset(lines, (80, 10, 10), 0)
    assert split_dataset(lines, (80, 10, 10), 0) != split_dataset(lines, (80, 10, 10), 1)


@pytest.mark.parametrize("ratios, sizes", [
    ((80, 10, 10), (96, 12, 12)),
    ((70, 20, 10), (84, 24, 12)),
    ((1, 0, 0), (120, 0, 0)),
    # Each label is split on its own: 40 gives 14, 13, 13 and each 20 gives 7, 7, 6, since
    # equal remainders go to the first parts. The gap is at most one position per label.
    ((1, 1, 1), (42, 41, 37)),
])
def test_ratios(ratios, sizes):
    parts, _ = split_dataset(make_lines(BALANCED), ratios, 0)
    assert tuple(len(parts[part]) for part in PARTS) == sizes


def test_shares_are_kept_label_by_label_with_odd_counts():
    counts = {"Nothing": 7, "Check White": 3, "Checkmate Black": 11}
    parts, _ = split_dataset(make_lines(counts), (80, 10, 10), 0)
    for label, count in counts.items():
        expected = share_out(count, [80, 10, 10])
        assert [labels_of(parts[part])[label] for part in PARTS] == expected


def test_duplicated_positions_are_removed_before_the_split():
    lines = make_lines(BALANCED)
    fields = lines[0].split()
    copy = " ".join(fields[:4] + ["9", "99"] + fields[6:])
    parts, duplicates = split_dataset(lines + [lines[0], copy, lines[1]], (80, 10, 10), 0)
    assert duplicates == 3
    assert sum(len(parts[part]) for part in PARTS) == len(lines)


def test_empty_lines_and_extra_spaces_are_accepted():
    lines = make_lines({"Nothing": 3})
    spaced = ["", "  " + lines[0].replace(" ", "   ") + "  ", "", lines[1], lines[2]]
    parts, _ = split_dataset(spaced, (1, 0, 0), 0)
    assert sorted(parts["train"]) == sorted(lines)


@pytest.mark.parametrize("bad_line, message", [
    ("8/8/8/8/8/8/8/k1K5 w - - 0 1", "line 3: missing expected label"),
    ("8/8/8/8/8/8/8/k1K5 w - - 0 1 Chek", "line 3: unknown label 'Chek'"),
    ("8/8/8 w - - 0 1 Nothing", "line 3: invalid FEN"),
])
def test_invalid_lines_raise_an_error(bad_line, message):
    with pytest.raises(AnalyzerError, match=message):
        split_dataset(["8/8/8/8/8/8/8/k1K5 w - - 0 1 Nothing", "", bad_line], (80, 10, 10), 0)


def test_script_writes_three_files_next_to_the_input(tmp_path, capsys):
    lines = make_lines(BALANCED)
    source = tmp_path / "all.txt"
    source.write_text("\n".join(lines) + "\n")
    split_main(["--input", str(source), "--seed", "4"])
    written = {part: (tmp_path / f"all_{part}.txt").read_text().splitlines() for part in PARTS}
    assert written == split_dataset(lines, (80, 10, 10), 4)[0]
    output = capsys.readouterr().out
    assert "all_train.txt: 96 positions (Nothing 32" in output
    first = [(tmp_path / f"all_{part}.txt").read_bytes() for part in PARTS]
    split_main(["--input", str(source), "--seed", "4"])
    assert [(tmp_path / f"all_{part}.txt").read_bytes() for part in PARTS] == first


def test_script_reports_the_removed_duplicates(tmp_path, capsys):
    lines = make_lines({"Nothing": 10})
    (tmp_path / "all.txt").write_text("\n".join(lines + lines[:2]) + "\n")
    split_main(["--input", str(tmp_path / "all.txt")])
    assert "2 duplicated positions removed" in capsys.readouterr().out


@pytest.mark.parametrize("content, message", [
    (None, "No such file or directory"),
    ("8/8/8/8/8/8/8/k1K5 w - - 0 1\n", "line 1: missing expected label"),
])
def test_script_errors_write_nothing(tmp_path, content, message):
    source = tmp_path / "all.txt"
    if content is not None:
        source.write_text(content)
    with pytest.raises(SystemExit) as error:
        split_main(["--input", str(source)])
    assert message in str(error.value.code)
    assert sorted(path.name for path in tmp_path.iterdir()) == (["all.txt"] if content else [])


@pytest.mark.parametrize("ratios", ["80:20", "0:50:50", "-1:1:1", "a:b:c"])
def test_script_refuses_invalid_ratios(tmp_path, ratios):
    with pytest.raises(SystemExit) as error:
        split_main(["--input", str(tmp_path / "all.txt"), f"--ratios={ratios}"])
    assert error.value.code == 2


def test_the_seed_chooses_which_positions_go_to_each_part():
    lines = make_lines(BALANCED)
    first, _ = split_dataset(lines, (80, 10, 10), 0)
    second, _ = split_dataset(lines, (80, 10, 10), 1)
    assert keys_of(first["test"]) != keys_of(second["test"])
    # Without shuffling each label first, validation and test would always hold the last
    # 20 % of the lines of each label: lines 32 to 39 for the 40 Nothing, and so on.
    last_of_each_label = keys_of(lines[32:40] + lines[56:60] + lines[76:80] + lines[96:100]
                                 + lines[116:120])
    assert keys_of(first["test"]) | keys_of(first["validation"]) != last_of_each_label
