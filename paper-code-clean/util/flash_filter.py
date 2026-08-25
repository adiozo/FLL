"""Detect unusually bright frames and write a relative-path exclusion list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def average_brightness(
    image_path: Path | str,
    crop_box: tuple[int, int, int, int] | None = None,
) -> float:
    with Image.open(image_path) as image:
        if crop_box is not None:
            image = image.crop(crop_box)
        grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    return float(grayscale.mean())


def find_bright_frames(
    root: Path | str,
    *,
    threshold: float,
    crop_box: tuple[int, int, int, int] | None = None,
) -> list[str]:
    root = Path(root)
    return [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*.png"))
        if not path.name.startswith(".") and average_brightness(path, crop_box) > threshold
    ]


def write_exclusion_file(excluded_files: list[str], output_json: Path | str) -> None:
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"excluded_files": excluded_files}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--crop-box", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    args = parser.parse_args()
    crop_box = tuple(args.crop_box) if args.crop_box else None
    excluded = find_bright_frames(args.root, threshold=args.threshold, crop_box=crop_box)
    write_exclusion_file(excluded, args.output_json)
    print(f"Recorded {len(excluded)} excluded frames")


if __name__ == "__main__":
    main()
