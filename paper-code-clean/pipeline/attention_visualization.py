"""Rank and export images by MIL attention-weighted class score."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import torch
from fastai.learner import load_learner
from marugoto.mil.data import get_target_enc


def load_feature_bag(feature_file: Path | str) -> tuple[torch.Tensor, list[str]]:
    """Load features and their source paths from one HDF5 bag."""
    with h5py.File(feature_file, "r") as handle:
        features = torch.from_numpy(handle["features"][:]).float()
        paths = handle["taken_frames"].asstr()[:].tolist()
    if len(features) != len(paths):
        raise ValueError("The feature and source-path counts differ")
    return features, paths


def _model_scores(
    features: torch.Tensor,
    model_path: Path | str,
    positive_class: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    learner = load_learner(model_path)
    encoder = learner.encoder.eval()
    attention_module = learner.attention.eval()
    head = learner.head.eval()
    target_encoder = get_target_enc(learner)
    positive_index = int(target_encoder.transform([[positive_class]]).argmax())

    with torch.inference_mode():
        encoded = encoder(features).squeeze()
        attention = torch.softmax(attention_module(encoded).squeeze(), dim=0)
        class_scores = torch.softmax(head(encoded), dim=1)[:, positive_index]
    return attention.detach().cpu(), class_scores.detach().cpu()


def rank_attention_tiles(
    feature_file: Path | str,
    model_paths: Sequence[Path | str],
    positive_class: str,
    *,
    n_tiles: int = 10,
    largest: bool = True,
) -> pd.DataFrame:
    """Average model outputs and return the highest-weighted source images."""
    if not model_paths:
        raise ValueError("At least one model path is required")
    features, source_paths = load_feature_bag(feature_file)
    model_outputs = [_model_scores(features, model_path, positive_class) for model_path in model_paths]
    attention = torch.stack([item[0] for item in model_outputs]).mean(dim=0)
    class_scores = torch.stack([item[1] for item in model_outputs]).mean(dim=0)
    weighted_scores = attention * class_scores

    count = min(n_tiles, len(weighted_scores))
    indices = weighted_scores.topk(count, largest=largest).indices.tolist()
    return pd.DataFrame(
        {
            "rank": range(1, count + 1),
            "source_path": [source_paths[index] for index in indices],
            "attention": [float(attention[index]) for index in indices],
            "class_score": [float(class_scores[index]) for index in indices],
            "weighted_score": [float(weighted_scores[index]) for index in indices],
        }
    )


def export_attention_tiles(
    ranking: pd.DataFrame,
    output_directory: Path | str,
    *,
    dataset_root: Path | str | None = None,
) -> None:
    """Write a ranking table, a score plot, and optionally copied source images."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output / "attention_ranking.csv", index=False)

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(ranking["rank"], ranking["attention"], marker="o", label="Attention")
    axis.plot(ranking["rank"], ranking["class_score"], marker="o", label="Class score")
    axis.plot(ranking["rank"], ranking["weighted_score"], marker="o", label="Weighted score")
    axis.set(xlabel="Rank", ylabel="Score")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "attention_scores.png", dpi=300)
    plt.close(figure)

    if dataset_root is None:
        return
    root = Path(dataset_root)
    tile_directory = output / "tiles"
    tile_directory.mkdir(exist_ok=True)
    for row in ranking.itertuples(index=False):
        source = Path(row.source_path)
        if not source.is_absolute():
            source = root / source
        if not source.is_file():
            raise FileNotFoundError(f"Source image not found: {source}")
        destination = tile_directory / f"rank_{row.rank:03d}{source.suffix.lower()}"
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_file", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--positive-class", required=True)
    parser.add_argument("--n-tiles", type=int, default=10)
    parser.add_argument("--smallest", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    ranking = rank_attention_tiles(
        args.feature_file,
        args.models,
        args.positive_class,
        n_tiles=args.n_tiles,
        largest=not args.smallest,
    )
    export_attention_tiles(ranking, args.output_directory, dataset_root=args.dataset_root)


if __name__ == "__main__":
    main()
