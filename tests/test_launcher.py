"""Tests of the real my_torch_analyzer executable, run as a separate process."""

import shutil
import subprocess
from pathlib import Path

BINARY = Path(__file__).resolve().parent.parent / "my_torch_analyzer"


def run_binary(binary, cwd):
    return subprocess.run([binary], cwd=cwd, capture_output=True, text=True)


def test_binary_runs_from_another_directory(tmp_path):
    result = run_binary(BINARY, tmp_path)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_failed_import_returns_84(tmp_path):
    isolated_binary = tmp_path / "my_torch_analyzer"
    shutil.copy(BINARY, isolated_binary)
    result = run_binary(isolated_binary, tmp_path)
    assert result.returncode == 84
    assert result.stdout == ""
    assert "No module named 'analyzer'" in result.stderr
