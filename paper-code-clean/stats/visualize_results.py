"""Generate ROC curves from one or more prediction directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from stats.metrics import plot_auc_curves


def generate_roc_curves(
    root: Path | str,
    *,
    target: str,
    classes: list[str],
    predictions_filename: str = "patient-preds.csv",
) -> None:
    root = Path(root)
    prediction_files = sorted(root.glob(f"*/{predictions_filename}"))
    if not prediction_files and (root / predictions_filename).exists():
        prediction_files = [root / predictions_filename]
    if not prediction_files:
        raise FileNotFoundError(f"No prediction tables found below {root}")
    for prediction_file in prediction_files:
        output_directory = prediction_file.parent / "figures"
        output_directory.mkdir(parents=True, exist_ok=True)
        plot_auc_curves(prediction_file, output_directory, target, classes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--predictions-filename", default="patient-preds.csv")
    args = parser.parse_args()
    generate_roc_curves(
        args.root,
        target=args.target,
        classes=args.classes,
        predictions_filename=args.predictions_filename,
    )


if __name__ == "__main__":
    main()
