"""Tests of the error handling of the analyzer entry point."""

import pytest

from analyzer.arguments import USAGE
from analyzer.errors import AnalyzerError
from analyzer.main import main
from my_torch import MyTorchError


def failing_run(error):
    def run(argv):
        raise error
    return run


def test_success_returns_0_and_writes_nothing_on_the_error_output(capsys):
    assert main(["--help"]) == 0
    captured = capsys.readouterr()
    assert captured.out == USAGE + "\n"
    assert captured.err == ""


@pytest.mark.parametrize("error", [
    AnalyzerError("invalid FEN"),
    MyTorchError("corrupted network"),
])
def test_expected_error_returns_84_with_its_message(monkeypatch, capsys, error):
    monkeypatch.setattr("analyzer.main.run", failing_run(error))
    assert main([]) == 84
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"my_torch_analyzer: {error}\n"


def test_unexpected_error_returns_84_and_is_reported_as_unexpected(monkeypatch, capsys):
    monkeypatch.setattr("analyzer.main.run", failing_run(ZeroDivisionError("division by zero")))
    assert main([]) == 84
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "my_torch_analyzer: unexpected error: division by zero\n"
