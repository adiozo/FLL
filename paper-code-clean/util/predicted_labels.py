"""Add an argmax prediction label to a prediction table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd


def add_predicted_label(
    input_csv: Path | str,
    output_csv: Path | str,
    probability_columns: Sequence[str],
    *,
    output_column: str = "prediction",
) -> pd.DataFrame:
    frame = pd.read_csv(input_csv)
    missing = set(probability_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing probability columns: {sorted(missing)}")
    frame[output_column] = frame[list(probability_columns)].idxmax(axis=1)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("probability_columns", nargs="+")
    parser.add_argument("--output-column", default="prediction")
    args = parser.parse_args()
    add_predicted_label(
        args.input_csv,
        args.output_csv,
        args.probability_columns,
        output_column=args.output_column,
    )


if __name__ == "__main__":
    main()
