"""Simple metric logging (console and optional CSV)."""

import csv
from pathlib import Path
from typing import Any


def log_epoch(epoch: int, metrics: dict[str, float], log_file: Path | None = None) -> None:
    """Print metrics to console and optionally append to CSV."""
    parts = [f"epoch {epoch}"]
    for k, v in metrics.items():
        parts.append(f"{k}={v:.4f}")
    print("  ".join(parts))
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_exists = log_file.exists()
        with open(log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch"] + list(metrics.keys()))
            if not file_exists:
                writer.writeheader()
            row: dict[str, Any] = {"epoch": epoch, **metrics}
            writer.writerow(row)
