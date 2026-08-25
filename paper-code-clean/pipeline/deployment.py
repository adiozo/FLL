"""Deploy every fold model from one cross-validation run."""

from __future__ import annotations

import argparse
from pathlib import Path

from marugoto_adapter.helpers import deploy_categorical_model


def deploy_cross_validation_models(
    model_root: Path | str,
    feature_directory: Path | str,
    output_root: Path | str,
    clinical_table: Path | str,
    sample_table: Path | str,
    *,
    target_label: str,
) -> None:
    model_root = Path(model_root)
    output_root = Path(output_root)
    fold_directories = sorted(path for path in model_root.iterdir() if path.is_dir())
    if not fold_directories:
        raise FileNotFoundError(f"No fold directories found below {model_root}")

    for fold_directory in fold_directories:
        model_path = fold_directory / "export.pkl"
        if not model_path.exists():
            continue
        output = output_root / fold_directory.name
        deploy_categorical_model(
            clinical_table,
            sample_table,
            feature_directory,
            model_path,
            output,
            target_label=target_label,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("feature_directory", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("clinical_table", type=Path)
    parser.add_argument("sample_table", type=Path)
    parser.add_argument("--target-label", required=True)
    args = parser.parse_args()
    deploy_cross_validation_models(
        args.model_root,
        args.feature_directory,
        args.output_root,
        args.clinical_table,
        args.sample_table,
        target_label=args.target_label,
    )


if __name__ == "__main__":
    main()
