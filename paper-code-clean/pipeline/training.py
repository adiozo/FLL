"""Run the multiple-instance learning cross-validation grid."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Sequence

from marugoto_adapter.helpers import categorical_crossval


def feature_metadata(feature_directory: Path) -> tuple[str, str]:
    match = re.fullmatch(r"model-(.+)_normalization-(.+)", feature_directory.name)
    if match:
        return match.group(1), match.group(2)
    legacy = re.fullmatch(r"MODEL=(.+)MEAN_AND_STD=(.+)", feature_directory.name)
    if legacy:
        return legacy.group(1), legacy.group(2)
    return feature_directory.name, "unknown"


def train_grid(
    feature_directories: Sequence[Path | str],
    clinical_table: Path | str,
    sample_table: Path | str,
    output_root: Path | str,
    *,
    targets: Sequence[str],
    bag_sizes: Sequence[int],
    batch_sizes: Sequence[int],
    learning_rates: Sequence[float],
    cycles: int = 5,
) -> None:
    output_root = Path(output_root)
    for feature_directory, target, bag_size, batch_size, learning_rate in itertools.product(
        map(Path, feature_directories), targets, bag_sizes, batch_sizes, learning_rates
    ):
        model_name, normalization = feature_metadata(feature_directory)
        for cycle in range(1, cycles + 1):
            output = (
                output_root
                / f"model_{model_name}"
                / f"mstd_{normalization}"
                / f"target_{target}"
                / f"bag_{bag_size}"
                / f"batch_{batch_size}"
                / f"lr_{learning_rate}"
                / f"crossval_{cycle}"
            )
            output.mkdir(parents=True, exist_ok=True)
            metadata = {
                "model": model_name,
                "normalization": normalization,
                "target": target,
                "cycle": cycle,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "bag_size": bag_size,
                "feature_directory": feature_directory.name,
            }
            (output / "run_info.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            categorical_crossval(
                clinical_table,
                sample_table,
                feature_dir=feature_directory,
                output_path=output,
                target_label=target,
                bag_size=bag_size,
                batch_size=batch_size,
                lr=learning_rate,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", nargs="+", type=Path, required=True)
    parser.add_argument("--clinical-table", type=Path, required=True)
    parser.add_argument("--sample-table", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--bag-sizes", nargs="+", type=int, default=[4096])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.001])
    parser.add_argument("--cycles", type=int, default=5)
    args = parser.parse_args()
    train_grid(
        args.features,
        args.clinical_table,
        args.sample_table,
        args.output_root,
        targets=args.targets,
        bag_sizes=args.bag_sizes,
        batch_sizes=args.batch_sizes,
        learning_rates=args.learning_rates,
        cycles=args.cycles,
    )


if __name__ == "__main__":
    main()
