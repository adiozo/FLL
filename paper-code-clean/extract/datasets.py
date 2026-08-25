"""Dataset discovery and image loading helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import crop


def get_sample_directories(root: Path | str) -> list[Path]:
    """Return ``root/class/sample`` directories in deterministic order."""
    root = Path(root)
    return sorted(
        sample
        for class_directory in root.iterdir()
        if class_directory.is_dir() and not class_directory.name.startswith(".")
        for sample in class_directory.iterdir()
        if sample.is_dir() and not sample.name.startswith(".")
    )


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def load_excluded_names(ignore_file: Path | str | None) -> set[str]:
    """Load excluded basenames from the current or legacy JSON schema."""
    if ignore_file is None:
        return set()
    path = Path(ignore_file)
    if not path.exists():
        raise FileNotFoundError(f"Ignore file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: Iterable[object]
    if isinstance(payload, dict):
        entries = payload.get("excluded_files", [])
    else:
        entries = payload

    excluded: set[str] = set()
    for entry in entries:
        value = entry.get("moved_from") if isinstance(entry, dict) else entry
        if value:
            excluded.add(Path(str(value)).name)
    return excluded


def get_image_files(sample_directory: Path | str, ignore_file: Path | str | None = None) -> list[Path]:
    excluded = load_excluded_names(ignore_file)
    images = [
        path
        for path in Path(sample_directory).iterdir()
        if path.is_file()
        and path.suffix.lower() == ".png"
        and not path.name.startswith(".")
        and path.name not in excluded
    ]
    return sorted(images, key=_natural_key)


class SampleImageDataset(Dataset):
    """Images from one sample bag."""

    def __init__(
        self,
        sample_directory: Path | str,
        transform=None,
        ignore_file: Path | str | None = None,
        *,
        minimum_images: int = 51,
        crop_box: tuple[int, int, int, int] | None = (20, 0, 204, 204),
    ) -> None:
        self.images = get_image_files(sample_directory, ignore_file)
        if len(self.images) < minimum_images:
            raise ValueError(
                f"{Path(sample_directory).name} contains {len(self.images)} usable images; "
                f"at least {minimum_images} are required"
            )
        self.transform = transform or torchvision.transforms.ToTensor()
        self.crop_box = crop_box

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        with Image.open(self.images[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        if self.crop_box is None:
            return tensor
        top, left, height, width = self.crop_box
        return crop(tensor, top, left, height, width)
