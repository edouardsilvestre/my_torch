"""Tests of the preparation of the dataset given with the subject."""

import zipfile

import pytest

from scripts.prepare_official_dataset import main as prepare_main

NOTHING = "8/8/8/8/8/8/8/k1K5 w - - 0 1 Nothing"
CHECK = "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3 Check White"
MATE = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate Black"


def make_archive(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("dataset/", "")
        for name, content in files.items():
            archive.writestr(name, content)
    return path


def prepare(tmp_path, files):
    archive = make_archive(tmp_path / "dataset.zip", files)
    output = tmp_path / "out" / "official.txt"
    prepare_main(["--archive", str(archive), "--output", str(output)])
    return output.read_text().splitlines()


def test_files_are_merged_in_the_order_of_their_names(tmp_path, capsys):
    lines = prepare(tmp_path, {
        "dataset/nothing/a.txt": f"{NOTHING}\n",
        "dataset/check/b.txt": f"{CHECK}\n\n{CHECK}\n",
        "dataset/checkmate/c.txt": MATE,
    })
    assert lines == [CHECK, CHECK, MATE, NOTHING]
    output = capsys.readouterr().out
    assert "dataset/check/b.txt: 2 positions" in output
    assert "4 positions written to" in output
    assert "Check White            2 (50.0 %)" in output
    assert "Check  " not in output.replace("Check White", "")


def test_extra_spaces_are_normalised_and_duplicates_kept(tmp_path):
    lines = prepare(tmp_path, {"dataset/a.txt": f"  {NOTHING.replace(' ', '   ')}  \n{NOTHING}\n"})
    assert lines == [NOTHING, NOTHING]


def test_files_that_are_not_text_files_are_ignored(tmp_path):
    lines = prepare(tmp_path, {"dataset/a.txt": f"{NOTHING}\n", "dataset/notes.md": "# notes"})
    assert lines == [NOTHING]


@pytest.mark.parametrize("content, message", [
    (f"{NOTHING}\n8/8/8/8/8/8/8/k1K5 w - - 0 1\n", "dataset/bad.txt:2: missing expected label"),
    (f"{NOTHING}\n{NOTHING[:-7]} Chek\n", "dataset/bad.txt:2: unknown label 'Chek'"),
    ("8/8/8 w - - 0 1 Nothing\n", "dataset/bad.txt:1: invalid FEN"),
])
def test_invalid_lines_stop_with_the_file_and_line(tmp_path, content, message):
    archive = make_archive(tmp_path / "dataset.zip", {"dataset/bad.txt": content})
    with pytest.raises(SystemExit) as error:
        prepare_main(["--archive", str(archive), "--output", str(tmp_path / "out.txt")])
    assert message in str(error.value.code)
    assert not (tmp_path / "out.txt").exists()


def test_missing_archive(tmp_path):
    with pytest.raises(SystemExit) as error:
        prepare_main(["--archive", str(tmp_path / "missing.zip"), "--output",
                      str(tmp_path / "out.txt")])
    assert "No such file or directory" in str(error.value.code)


def test_file_that_is_not_an_archive(tmp_path):
    (tmp_path / "dataset.zip").write_text("not a zip")
    with pytest.raises(SystemExit) as error:
        prepare_main(["--archive", str(tmp_path / "dataset.zip"), "--output",
                      str(tmp_path / "out.txt")])
    assert "File is not a zip file" in str(error.value.code)
