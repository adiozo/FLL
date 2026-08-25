"""Aggregate per-fold prediction tables without hard-coded class names."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd


def aggregate_fold_predictions(
    deployment_root: Path | str,
    output_csv: Path | str,
    *,
    id_column: str = "PATIENT",
    target_column: str = "DIAGNOSIS",
    predictions_filename: str = "patient-preds.csv",
) -> pd.DataFrame:
    """Average matching probability columns across fold directories."""
    root = Path(deployment_root)
    prediction_files = sorted(root.glob(f"*/{predictions_filename}"))
    if not prediction_files:
        raise FileNotFoundError(f"No fold predictions found below {root}")

    frames: list[pd.DataFrame] = []
    probability_columns: list[str] | None = None
    reference_targets: pd.Series | None = None
    for fold_index, prediction_file in enumerate(prediction_files):
        frame = pd.read_csv(prediction_file)
        required = {id_column, target_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{prediction_file} is missing columns: {sorted(missing)}")
        if frame[id_column].duplicated().any():
            raise ValueError(f"Duplicate sample identifiers found in {prediction_file}")

        current_targets = frame.set_index(id_column)[target_column].sort_index()
        if reference_targets is None:
            reference_targets = current_targets
        elif not current_targets.equals(reference_targets):
            raise ValueError("Fold sample identifiers or target labels are inconsistent")

        current_probability_columns = sorted(
            column for column in frame.columns if column.startswith(f"{target_column}_")
        )
        if not current_probability_columns:
            raise ValueError(f"No probability columns found in {prediction_file}")
        if probability_columns is None:
            probability_columns = current_probability_columns
        elif current_probability_columns != probability_columns:
            raise ValueError("Fold prediction columns are inconsistent")

        selected = frame[[id_column, target_column, *current_probability_columns]].copy()
        selected = selected.rename(
            columns={column: f"{column}__fold_{fold_index}" for column in current_probability_columns}
        )
        frames.append(selected)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(
            frame.drop(columns=target_column),
            on=id_column,
            how="inner",
            validate="one_to_one",
        )

    assert probability_columns is not None
    for column in probability_columns:
        fold_columns = [name for name in merged.columns if name.startswith(f"{column}__fold_")]
        merged[column] = merged[fold_columns].mean(axis=1)

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deployment_root", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--id-column", default="PATIENT")
    parser.add_argument("--target-column", default="DIAGNOSIS")
    parser.add_argument("--predictions-filename", default="patient-preds.csv")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    aggregate_fold_predictions(
        args.deployment_root,
        args.output_csv,
        id_column=args.id_column,
        target_column=args.target_column,
        predictions_filename=args.predictions_filename,
    )


if __name__ == "__main__":
    main()
