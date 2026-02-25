"""Tests for epoch logging (utils.training)."""

from pathlib import Path

from utils import log_epoch


def test_log_epoch_creates_csv_when_log_file_given(tmp_path: Path, capsys: object) -> None:
    log_file = tmp_path / "metrics.csv"
    log_epoch(1, {"loss": 0.5, "acc": 0.9}, log_file=log_file)
    captured = capsys.readouterr()
    assert "epoch 1" in captured.out
    assert "loss=" in captured.out
    assert "acc=" in captured.out
    assert log_file.exists()
    content = log_file.read_text()
    assert "epoch" in content
    assert "loss" in content
    assert "acc" in content


def test_log_epoch_appends_second_row(tmp_path: Path) -> None:
    log_file = tmp_path / "metrics.csv"
    log_epoch(1, {"loss": 0.5}, log_file=log_file)
    log_epoch(2, {"loss": 0.3}, log_file=log_file)
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 3  # header + 2 rows


def test_log_epoch_no_file_does_not_create_file(capsys: object) -> None:
    log_epoch(1, {"loss": 0.5}, log_file=None)
    captured = capsys.readouterr()
    assert "epoch 1" in captured.out
