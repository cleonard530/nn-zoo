"""Device selection for training and inference."""

import torch


def get_device(use_cuda: bool = True) -> torch.device:
    """Return CUDA device if available and requested, else CPU."""
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
