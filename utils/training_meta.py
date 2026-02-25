"""Save training run metadata to a .txt file alongside checkpoints."""

from datetime import datetime
from pathlib import Path
from typing import Any


def get_run_id() -> str:
    """Return current date and time as Y_M_D_H_M_S for use in filenames."""
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def save_training_metadata(
    save_dir: str | Path,
    run_id: str,
    args: Any,
    metrics: dict[str, Any],
    model_hp: dict[str, Any] | None = None,
) -> None:
    """Write a .txt file with run_id, hyperparameters, and final/best metrics."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"meta_{run_id}.txt"
    lines = [
        f"run_id: {run_id}",
        f"timestamp: {datetime.now().isoformat()}",
        "",
        "=== Metrics ===",
    ]
    for k, v in metrics.items():
        lines.append(f"  {k}: {v}")
    lines.extend(["", "=== Hyperparameters (args) ==="])
    args_dict = vars(args) if hasattr(args, "__dict__") else args
    for k, v in sorted(args_dict.items()):
        lines.append(f"  {k}: {v}")
    if model_hp:
        lines.extend(["", "=== Model hyperparameters ==="])
        for k, v in sorted(model_hp.items()):
            lines.append(f"  {k}: {v}")
    path.write_text("\n".join(lines) + "\n")
    print(f"Metadata saved to {path}")
