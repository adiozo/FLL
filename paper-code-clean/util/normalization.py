"""Estimate per-channel normalization values for an image dataset."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from extract.datasets import load_excluded_names


class NormalizationDataset(Dataset):
    def __init__(self, root: Path | str, ignore_file: Path | str | None = None) -> None:
        excluded = load_excluded_names(ignore_file)
        self.frames = sorted(
            path
            for path in Path(root).rglob("*.png")
            if not path.name.startswith(".") and path.name not in excluded
        )
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int):
        with Image.open(self.frames[index]) as image:
            return self.transform(image.convert("RGB"))


def calculate_mean_std(
    dataset_root: Path | str,
    ignore_file: Path | str | None = None,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[list[float], list[float]]:
    dataset = NormalizationDataset(dataset_root, ignore_file)
    if not dataset:
        raise ValueError("No usable PNG images were found")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_squared_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for batch in loader:
        batch = batch.to(dtype=torch.float64)
        channel_sum += batch.sum(dim=(0, 2, 3))
        channel_squared_sum += (batch**2).sum(dim=(0, 2, 3))
        pixel_count += batch.shape[0] * batch.shape[2] * batch.shape[3]

    mean = channel_sum / pixel_count
    variance = channel_squared_sum / pixel_count - mean**2
    std = variance.clamp_min(0).sqrt()
    return mean.tolist(), std.tolist()


def get_mean_and_std_from_dataset(
    dataset_root: Path | str,
    ignore_file: Path | str | None = None,
) -> tuple[list[float], list[float]]:
    """Backward-compatible wrapper used by the feature extractor."""
    return calculate_mean_std(dataset_root, ignore_file)


def write_normalization(values: tuple[list[float], list[float]], output_json: Path | str) -> None:
    mean, std = values
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"mean": mean, "std": std}, indent=2) + "\n",
        encoding="utf-8",
    )
