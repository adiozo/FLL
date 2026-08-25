"""Extract one HDF5 feature bag per sample directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import h5py
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from extract.datasets import SampleImageDataset, get_sample_directories
from util.normalization import calculate_mean_std

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(model_name: str):
    name = model_name.lower()
    if name == "resnet18":
        return torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )
    if name == "resnet50":
        return torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
    if name == "inception_v3":
        return torchvision.models.inception_v3(
            weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_transform(mean: list[float] | None, std: list[float] | None):
    operations = [
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ]
    if mean is not None and std is not None:
        operations.append(torchvision.transforms.Normalize(mean, std))
    return torchvision.transforms.Compose(operations)


def prepare_extractor(model, device: torch.device):
    if not hasattr(model, "fc"):
        raise TypeError("The selected model has no replaceable 'fc' layer")
    model.fc = nn.Identity()
    return model.eval().to(device)


def write_feature_bag(
    output_path: Path,
    features,
    relative_image_paths: list[str],
    model_name: str,
) -> None:
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("features", data=features)
        handle.create_dataset("taken_frames", data=relative_image_paths, dtype=string_type)
        handle.attrs["extractor"] = model_name
        handle.attrs["paths_are_relative"] = True


def extract_features(
    dataset_root: Path | str,
    output_root: Path | str,
    *,
    model_name: str = "resnet50",
    normalization: str = "imagenet",
    mean: list[float] | None = None,
    std: list[float] | None = None,
    ignore_file: Path | str | None = None,
    minimum_images: int = 51,
    batch_size: int = 16,
    num_workers: int | None = None,
    device: str | None = None,
) -> Path:
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if normalization == "calculated":
        mean, std = calculate_mean_std(dataset_root, ignore_file)
    elif normalization == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif normalization == "none":
        mean, std = None, None
    elif normalization == "custom":
        if mean is None or std is None:
            raise ValueError("Custom normalization requires --mean and --std")
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")

    output_directory = output_root / f"model-{model_name}_normalization-{normalization}"
    output_directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": model_name,
        "normalization": normalization,
        "mean": mean,
        "std": std,
        "minimum_images": minimum_images,
    }
    (output_directory / "info.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    model = prepare_extractor(build_model(model_name), selected_device)
    transform = build_transform(mean, std)
    workers = num_workers if num_workers is not None else max(0, (os.cpu_count() or 1) // 2)

    sample_directories = get_sample_directories(dataset_root)
    sample_names = [path.name for path in sample_directories]
    if len(sample_names) != len(set(sample_names)):
        raise ValueError("Sample directory names must be unique across classes")

    for sample_directory in tqdm(sample_directories, desc="samples"):
        output_path = output_directory / f"{sample_directory.name}.h5"
        if output_path.exists():
            continue
        try:
            images = SampleImageDataset(
                sample_directory,
                transform,
                ignore_file,
                minimum_images=minimum_images,
            )
        except ValueError as error:
            print(f"Skipping {sample_directory.name}: {error}")
            continue

        loader = DataLoader(
            images,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
        )
        batches = []
        with torch.inference_mode():
            for batch in loader:
                batches.append(model(batch.to(selected_device)).detach().cpu())
        features = torch.cat(batches).to(dtype=torch.float16).numpy()
        relative_paths = [path.relative_to(dataset_root).as_posix() for path in images.images]
        write_feature_bag(output_path, features, relative_paths, model_name)
    return output_directory


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--model", default="resnet50", choices=["resnet18", "resnet50", "inception_v3"])
    parser.add_argument("--normalization", default="imagenet", choices=["imagenet", "calculated", "none", "custom"])
    parser.add_argument("--mean", nargs=3, type=float)
    parser.add_argument("--std", nargs=3, type=float)
    parser.add_argument("--ignore-file", type=Path)
    parser.add_argument("--minimum-images", type=int, default=51)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    extract_features(
        args.dataset_root,
        args.output_root,
        model_name=args.model,
        normalization=args.normalization,
        mean=args.mean,
        std=args.std,
        ignore_file=args.ignore_file,
        minimum_images=args.minimum_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
