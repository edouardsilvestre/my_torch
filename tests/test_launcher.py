"""Tests of the real executables, run as separate processes."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ANALYZER = ROOT / "my_torch_analyzer"
GENERATOR = ROOT / "my_torch_generator"


def run_binary(binary, cwd, *arguments):
    return subprocess.run([binary, *arguments], cwd=cwd, capture_output=True, text=True)


def test_binary_runs_from_another_directory(tmp_path):
    result = run_binary(ANALYZER, tmp_path)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize("binary, package", [
    (ANALYZER, "analyzer"),
    (GENERATOR, "my_torch"),
])
def test_failed_import_returns_84(tmp_path, binary, package):
    isolated_binary = tmp_path / binary.name
    shutil.copy(binary, isolated_binary)
    result = run_binary(isolated_binary, tmp_path)
    assert result.returncode == 84
    assert result.stdout == ""
    assert f"No module named '{package}'" in result.stderr


def test_generator_help(tmp_path):
    result = run_binary(GENERATOR, tmp_path, "--help")
    assert result.returncode == 0
    assert result.stdout.startswith("USAGE\n        ./my_torch_generator config_file_1 nb_1")
    assert result.stderr == ""


def test_generator_example_of_the_subject(tmp_path):
    shutil.copy(ROOT / "configs" / "basic_network.conf", tmp_path)
    result = run_binary(GENERATOR, tmp_path, "basic_network.conf", "3")
    assert result.returncode == 0
    assert result.stdout == "" and result.stderr == ""
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "basic_network.conf", "basic_network_1.nn", "basic_network_2.nn", "basic_network_3.nn",
    ]


@pytest.mark.parametrize("arguments", [
    (),
    ("basic_network.conf",),
    ("basic_network.conf", "zero"),
    ("invalid.conf", "1"),
    ("missing.conf", "1"),
])
def test_generator_errors_return_84(tmp_path, arguments):
    (tmp_path / "invalid.conf").write_text("[network]\ninput_size = 0\n")
    shutil.copy(ROOT / "configs" / "basic_network.conf", tmp_path)
    result = run_binary(GENERATOR, tmp_path, *arguments)
    assert result.returncode == 84
    assert result.stdout == ""
    assert result.stderr.startswith("my_torch_generator: ")
    assert not any(path.suffix == ".nn" for path in tmp_path.iterdir())
