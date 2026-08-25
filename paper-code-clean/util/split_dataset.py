"""Create a deterministic stratified directory split."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def split_dataset(
    source_root: Path | str,
    output_root: Path | str,
    *,
    test_fraction: float = 0.2,
    minimum_files: int = 50,
    seed: int = 2025,
    mode: str = "copy",
) -> dict[str, dict[str, list[str]]]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if mode not in {"copy", "move"}:
        raise ValueError("mode must be 'copy' or 'move'")

    source = Path(source_root)
    output = Path(output_root)
    rng = random.Random(seed)
    manifest: dict[str, dict[str, list[str]]] = {}

    for class_directory in sorted(path for path in source.iterdir() if path.is_dir()):
        samples = [
            path
            for path in class_directory.iterdir()
            if path.is_dir() and sum(1 for item in path.iterdir() if item.is_file()) >= minimum_files
        ]
        samples.sort()
        rng.shuffle(samples)
        test_size = max(1, round(len(samples) * test_fraction)) if samples else 0
        split = {"test": samples[:test_size], "train": samples[test_size:]}
        manifest[class_directory.name] = {}

        for split_name, selected_samples in split.items():
            manifest[class_directory.name][split_name] = [sample.name for sample in selected_samples]
            for sample in selected_samples:
                destination = output / split_name / class_directory.name / sample.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if mode == "copy":
                    shutil.copytree(sample, destination, dirs_exist_ok=True)
                else:
                    shutil.move(str(sample), destination)

    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--minimum-files", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--mode", choices=["copy", "move"], default="copy")
    args = parser.parse_args()
    split_dataset(
        args.source_root,
        args.output_root,
        test_fraction=args.test_fraction,
        minimum_files=args.minimum_files,
        seed=args.seed,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
