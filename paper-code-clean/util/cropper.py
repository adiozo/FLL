"""Crop PNG files while preserving their relative directory structure."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def crop_dataset(
    source_root: Path | str,
    destination_root: Path | str,
    crop_box: tuple[int, int, int, int],
) -> int:
    source = Path(source_root)
    destination = Path(destination_root)
    count = 0
    for image_path in sorted(source.rglob("*.png")):
        if image_path.name.startswith("."):
            continue
        output_path = destination / image_path.relative_to(source)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            image.crop(crop_box).save(output_path)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("destination_root", type=Path)
    parser.add_argument("--crop-box", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"), required=True)
    args = parser.parse_args()
    count = crop_dataset(args.source_root, args.destination_root, tuple(args.crop_box))
    print(f"Wrote {count} cropped images")


if __name__ == "__main__":
    main()
