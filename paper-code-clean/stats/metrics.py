"""Metrics and plots used by the experiment pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve


def generate_concatenated_preds_file_by_folds(
    root_path: Path | str,
    write_file: bool = False,
    out_path: Path | str | None = None,
) -> pd.DataFrame:
    """Concatenate ``patient-preds.csv`` files from direct fold children."""
    prediction_files = sorted(Path(root_path).glob("*/patient-preds.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No fold predictions found below {root_path}")
    result = pd.concat((pd.read_csv(path) for path in prediction_files), ignore_index=True)
    if write_file:
        if out_path is None:
            raise ValueError("out_path is required when write_file=True")
        output = Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
    return result


def get_auc_list(
    predictions: Path | str | pd.DataFrame,
    labels: Sequence[str],
    target_label: str = "DIAGNOSIS",
) -> dict[int, tuple[tuple[np.ndarray, np.ndarray, np.ndarray], str]]:
    """Return one-vs-rest ROC arrays for each label in the supplied order."""
    frame = predictions if isinstance(predictions, pd.DataFrame) else pd.read_csv(predictions)
    true_labels = frame[target_label].astype(str)
    result = {}
    for index, label in enumerate(labels):
        score_column = f"{target_label}_{label}"
        if score_column not in frame:
            raise ValueError(f"Missing probability column: {score_column}")
        result[index] = (
            roc_curve(true_labels.eq(label).to_numpy(), frame[score_column].to_numpy()),
            label,
        )
    return result


def plot_auc_curves(
    patient_predictions: Path | str,
    output_directory: Path | str,
    target: str,
    labels: Sequence[str],
) -> None:
    """Write individual and combined ROC curves."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    curves = get_auc_list(patient_predictions, labels, target)

    for index in range(len(labels)):
        (false_positive_rate, true_positive_rate, _), label = curves[index]
        score = auc(false_positive_rate, true_positive_rate)
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.plot(false_positive_rate, true_positive_rate, label=f"AUROC = {score:.3f}")
        axis.plot([0, 1], [0, 1], "r--")
        axis.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC: {label}")
        axis.set_aspect("equal", adjustable="box")
        axis.legend(loc="lower right")
        figure.tight_layout()
        figure.savefig(output / f"roc_{label}.png", dpi=300)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "r--")
    for index in range(len(labels)):
        (false_positive_rate, true_positive_rate, _), label = curves[index]
        score = auc(false_positive_rate, true_positive_rate)
        axis.plot(false_positive_rate, true_positive_rate, label=f"{label}: {score:.3f}")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output / "roc_combined.png", dpi=300)
    plt.close(figure)


def calculate_threshold(
    predictions: Path | str | pd.DataFrame,
    labels: Sequence[str],
    target: str,
    positive_label: str,
) -> tuple[float, float, float]:
    """Select the ROC point closest to perfect classification."""
    curves = get_auc_list(predictions, labels, target)
    selected = next((curve for curve, label in curves.values() if label == positive_label), None)
    if selected is None:
        raise ValueError(f"Label {positive_label!r} is not present in labels")
    false_positive_rate, true_positive_rate, thresholds = selected
    distances = (1 - true_positive_rate) ** 2 + false_positive_rate**2
    index = int(np.argmin(distances))
    return (
        float(thresholds[index]),
        float(true_positive_rate[index]),
        float(false_positive_rate[index]),
    )


def calculate_macro_ovr_auc(
    frame: pd.DataFrame,
    target_label: str,
    classes: Sequence[str],
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Calculate class-wise one-vs-rest AUROC and their macro average."""
    if indices is None:
        indices = np.arange(len(frame))
    true_labels = frame[target_label].astype(str).to_numpy()[indices]
    class_aucs = []
    for class_name in classes:
        score_column = f"{target_label}_{class_name}"
        if score_column not in frame:
            raise ValueError(f"Missing probability column: {score_column}")
        binary_truth = true_labels == class_name
        scores = pd.to_numeric(frame[score_column], errors="raise").to_numpy()[indices]
        if np.unique(binary_truth).size != 2:
            raise ValueError(f"Class {class_name!r} lacks positive or negative cases")
        class_aucs.append(roc_auc_score(binary_truth, scores))
    values = np.asarray(class_aucs, dtype=float)
    return values, float(values.mean())


def bootstrap_macro_ovr_auc(
    frame: pd.DataFrame,
    target_label: str,
    classes: Sequence[str],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = 42,
) -> dict[str, object]:
    """Run a stratified bootstrap for macro one-vs-rest AUROC."""
    if len(classes) < 2:
        raise ValueError("At least two classes are required")
    true_labels = frame[target_label].astype(str).to_numpy()
    unknown = set(np.unique(true_labels)) - set(classes)
    if unknown:
        raise ValueError(f"Observed labels are missing from classes: {sorted(unknown)}")

    class_indices = {label: np.flatnonzero(true_labels == label) for label in classes}
    empty = [label for label, indices in class_indices.items() if not len(indices)]
    if empty:
        raise ValueError(f"Classes without observations: {empty}")

    generator = np.random.default_rng(random_state)
    point_class_aucs, point_macro_auc = calculate_macro_ovr_auc(frame, target_label, classes)
    bootstrap_class_aucs = np.empty((n_bootstrap, len(classes)), dtype=float)
    bootstrap_macro_aucs = np.empty(n_bootstrap, dtype=float)
    for bootstrap_index in range(n_bootstrap):
        sampled_indices = np.concatenate(
            [generator.choice(indices, size=len(indices), replace=True) for indices in class_indices.values()]
        )
        generator.shuffle(sampled_indices)
        class_aucs, macro_auc = calculate_macro_ovr_auc(
            frame, target_label, classes, sampled_indices
        )
        bootstrap_class_aucs[bootstrap_index] = class_aucs
        bootstrap_macro_aucs[bootstrap_index] = macro_auc

    alpha = 1.0 - confidence_level
    quantiles = [alpha / 2, 1 - alpha / 2]
    return {
        "classes": list(classes),
        "class_aucs": point_class_aucs,
        "class_cis": np.quantile(bootstrap_class_aucs, quantiles, axis=0).T,
        "macro_auc": point_macro_auc,
        "macro_ci": np.quantile(bootstrap_macro_aucs, quantiles),
        "bootstrap_macro_aucs": bootstrap_macro_aucs,
    }
