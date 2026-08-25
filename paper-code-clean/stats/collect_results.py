"""Collect prediction metrics from a directory of experiment runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def calculate_prediction_metrics(predictions_csv: Path | str, target_column: str) -> dict[str, object]:
    frame = pd.read_csv(predictions_csv)
    classes = sorted(frame[target_column].dropna().astype(str).unique())
    class_aucs: dict[str, float] = {}
    for class_name in classes:
        score_column = f"{target_column}_{class_name}"
        if score_column not in frame:
            raise ValueError(f"Missing probability column: {score_column}")
        class_aucs[class_name] = roc_auc_score(
            frame[target_column].astype(str).eq(class_name),
            frame[score_column],
        )

    result: dict[str, object] = {
        "auc_macro": float(np.mean(list(class_aucs.values()))),
        "auc_class_std": float(np.std(list(class_aucs.values()))),
        **{f"auc_{name}": value for name, value in class_aucs.items()},
    }
    if "pred" in frame:
        result["accuracy"] = float(accuracy_score(frame[target_column], frame["pred"]))
    return result


def _load_run_metadata(prediction_file: Path, root: Path) -> dict[str, object]:
    for parent in prediction_file.parents:
        if parent == root.parent:
            break
        for filename in ("run_info.json", "info.json"):
            metadata_file = parent / filename
            if metadata_file.exists():
                try:
                    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                allowed = {
                    "model", "normalization", "target", "target_label", "cycle",
                    "learning_rate", "batch_size", "bag_size", "modality",
                }
                return {key: value for key, value in payload.items() if key in allowed}
    return {}


def collect_results(root: Path | str, target_column: str) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for prediction_file in sorted(root.rglob("patient-preds.csv")):
        row = {
            "run": prediction_file.parent.relative_to(root).as_posix(),
            **_load_run_metadata(prediction_file, root),
            **calculate_prediction_metrics(prediction_file, target_column),
        }
        history_file = prediction_file.parent / "history.csv"
        if history_file.exists():
            history = pd.read_csv(history_file)
            if "roc_auc_score" in history:
                row["history_auc_max"] = float(history["roc_auc_score"].max())
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No prediction tables found below {root}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--target-column", required=True)
    args = parser.parse_args()
    results = collect_results(args.root, args.target_column)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
