"""Aggregate cross-validation metrics and rank model configurations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd


DEFAULT_GROUP_COLUMNS = ("model", "mstd", "target", "bag", "batch", "lr", "cycle")


def summarize_models(
    input_csv: Path | str,
    *,
    metric_column: str = "AUC_Average",
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
) -> pd.DataFrame:
    frame = pd.read_csv(input_csv)
    required = set(group_columns) | {metric_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    return (
        frame.groupby(list(group_columns), dropna=False)[metric_column]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": f"{metric_column}_mean", "std": f"{metric_column}_std"})
        .sort_values(f"{metric_column}_mean", ascending=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--metric", default="AUC_Average")
    parser.add_argument("--group-columns", nargs="+", default=list(DEFAULT_GROUP_COLUMNS))
    args = parser.parse_args()
    summary = summarize_models(args.input_csv, metric_column=args.metric, group_columns=args.group_columns)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
