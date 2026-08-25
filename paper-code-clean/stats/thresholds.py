"""Estimate decision thresholds from cross-validation predictions."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from stats.metrics import calculate_threshold, generate_concatenated_preds_file_by_folds


def averaged_fold_threshold(
    root: Path | str,
    labels: list[str],
    target: str,
    positive_label: str,
) -> float:
    thresholds = []
    for prediction_file in sorted(Path(root).glob("*/patient-preds.csv")):
        threshold, _, _ = calculate_threshold(prediction_file, labels, target, positive_label)
        thresholds.append(float(threshold))
    if not thresholds:
        raise FileNotFoundError(f"No fold predictions found below {root}")
    return statistics.mean(thresholds)


def concatenated_threshold(
    root: Path | str,
    labels: list[str],
    target: str,
    positive_label: str,
) -> float:
    predictions = generate_concatenated_preds_file_by_folds(str(root))
    threshold, _, _ = calculate_threshold(predictions, labels, target, positive_label)
    return float(threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--positive-label", required=True)
    parser.add_argument("--method", choices=["averaged", "concatenated"], default="concatenated")
    args = parser.parse_args()
    function = averaged_fold_threshold if args.method == "averaged" else concatenated_threshold
    print(function(args.root, args.labels, args.target, args.positive_label))


if __name__ == "__main__":
    main()
