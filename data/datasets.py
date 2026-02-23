"""Dataset loading and download utilities. All use a single data root (e.g. ./data)."""

import os
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST


def _default_mnist_transform(train: bool) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def get_mnist(
    root: str | Path,
    train: bool = True,
    transform: Optional[Callable] = None,
    download: bool = True,
) -> MNIST:
    """MNIST dataset. Root is e.g. ./data; files go under root/MNIST."""
    root = Path(root)
    root = root / "raw"
    if transform is None:
        transform = _default_mnist_transform(train)
    return MNIST(root=str(root), train=train, transform=transform, download=download)


def get_fashion_mnist(
    root: str | Path,
    train: bool = True,
    transform: Optional[Callable] = None,
    download: bool = True,
) -> FashionMNIST:
    """Fashion-MNIST dataset."""
    root = Path(root)
    root = root / "raw"
    if transform is None:
        transform = _default_mnist_transform(train)
    return FashionMNIST(root=str(root), train=train, transform=transform, download=download)


def _default_cifar_transform(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616),
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )


def get_cifar10(
    root: str | Path,
    train: bool = True,
    transform: Optional[Callable] = None,
    download: bool = True,
) -> CIFAR10:
    """CIFAR-10 dataset."""
    root = Path(root)
    root = root / "raw"
    if transform is None:
        transform = _default_cifar_transform(train)
    return CIFAR10(root=str(root), train=train, transform=transform, download=download)


def get_cifar100(
    root: str | Path,
    train: bool = True,
    transform: Optional[Callable] = None,
    download: bool = True,
) -> CIFAR100:
    """CIFAR-100 dataset."""
    root = Path(root)
    root = root / "raw"
    if transform is None:
        transform = _default_cifar_transform(train)
    return CIFAR100(root=str(root), train=train, transform=transform, download=download)


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader from a dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
    )


# --- Synthetic sequence dataset for LSTM / GPT ---


class SyntheticSequenceDataset(Dataset):
    """
    Simple synthetic sequence dataset for studying RNN/Transformer.
    Each sample is a sequence of integers (e.g. 0..vocab_size-1); target is next token
    (for GPT-style) or class (e.g. sum mod num_classes for classification).
    """

    def __init__(
        self,
        num_samples: int = 10000,
        seq_len: int = 32,
        vocab_size: int = 128,
        target_mode: str = "next_token",  # "next_token" or "class" (uses num_classes)
        num_classes: int = 10,
        seed: int = 42,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.target_mode = target_mode
        self.num_classes = num_classes
        gen = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len), generator=gen)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        if self.target_mode == "next_token":
            # Predict next token at each position (shifted right)
            y = torch.roll(x, -1, dims=0)
            y[-1] = x[0]  # wrap last target
            return x.long(), y.long()
        else:
            # Classification: e.g. sum of sequence mod num_classes
            y = (x.sum().item() % self.num_classes)
            return x.long(), torch.tensor(y, dtype=torch.long)


def get_synthetic_sequence(
    num_samples: int = 10000,
    seq_len: int = 32,
    vocab_size: int = 128,
    target_mode: str = "next_token",
    num_classes: int = 10,
    seed: int = 42,
) -> SyntheticSequenceDataset:
    """Build synthetic sequence dataset for LSTM/GPT experiments."""
    return SyntheticSequenceDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        target_mode=target_mode,
        num_classes=num_classes,
        seed=seed,
    )


class MNISTSegmentationDataset(Dataset):
    """
    MNIST with binary segmentation mask: mask = (image > threshold).
    For studying U-Net on a simple segmentation task.
    """

    def __init__(self, root: str | Path, train: bool = True, threshold: float = 0.3, download: bool = True):
        super().__init__()
        self.mnist = get_mnist(root, train=train, download=download)
        self.threshold = threshold

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.mnist[idx]
        mask = (x > self.threshold).float()
        return x, mask


def get_mnist_segmentation(
    root: str | Path,
    train: bool = True,
    threshold: float = 0.3,
    download: bool = True,
) -> MNISTSegmentationDataset:
    """MNIST with binary mask for U-Net segmentation."""
    return MNISTSegmentationDataset(root=root, train=train, threshold=threshold, download=download)
