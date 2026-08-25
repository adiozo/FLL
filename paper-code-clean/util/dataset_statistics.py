"""Create aggregate, identifier-free dataset summaries."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Sequence


def summarize_dataset(root: Path | str, extensions: Sequence[str] = (".png",)) -> dict[str, object]:
    root = Path(root)
    normalized_extensions = {extension.lower() for extension in extensions}
    per_class: dict[str, dict[str, int]] = {}
    all_counts: list[int] = []

    for class_directory in sorted(path for path in root.iterdir() if path.is_dir()):
        sample_counts = []
        for sample_directory in sorted(path for path in class_directory.iterdir() if path.is_dir()):
            count = sum(
                1
                for path in sample_directory.iterdir()
                if path.is_file() and path.suffix.lower() in normalized_extensions
            )
            if count:
                sample_counts.append(count)
                all_counts.append(count)
        per_class[class_directory.name] = {
            "samples": len(sample_counts),
            "images": sum(sample_counts),
        }

    return {
        "classes": per_class,
        "total_samples": sum(item["samples"] for item in per_class.values()),
        "total_images": sum(item["images"] for item in per_class.values()),
        "median_images_per_sample": statistics.median(all_counts) if all_counts else None,
        "stdev_images_per_sample": statistics.stdev(all_counts) if len(all_counts) > 1 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--extensions", nargs="+", default=[".png"])
    args = parser.parse_args()
    summary = summarize_dataset(args.root, args.extensions)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
