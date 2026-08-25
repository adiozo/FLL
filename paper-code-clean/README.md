# Medical imaging multiple-instance learning pipeline

Research code for extracting image features, training and deploying multiple-instance learning models,
aggregating cross-validation predictions, and calculating evaluation metrics.

This public copy contains no datasets, model weights, prediction tables, machine-specific paths, or
sample-level identifiers.

## Repository layout

- `extract/`: dataset discovery and pretrained feature extraction
- `pipeline/`: training, deployment, and attention visualization
- `marugoto_adapter/`: project-specific adaptations of Marugoto routines
- `ensemble/`: fold-prediction aggregation
- `stats/`: metrics, bootstrapping, thresholds, and ROC plots
- `util/`: preprocessing and dataset-audit commands
- `tests/`: tests for path-neutral, dependency-light functionality

## Installation

Use Python 3.10 or newer in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Training, deployment, and attention visualization also require a compatible Marugoto installation.
Record the exact Marugoto revision and a complete environment lock file before creating the paper release.

## Expected data layout

Feature extraction expects one directory per class and one subdirectory per sample:

```text
data/images/
├── class_a/
│   ├── sample_001/
│   └── sample_002/
└── class_b/
    ├── sample_003/
    └── sample_004/
```

Input tables are expected to follow the schema required by Marugoto. The current adapter uses `PATIENT`
as the sample identifier column and a user-supplied target column.

## Typical workflow

Extract one HDF5 feature bag per sample:

```bash
python -m extract.features data/images outputs/features \
  --model resnet50 \
  --normalization imagenet
```

Train the cross-validation grid:

```bash
python -m pipeline.training \
  --features outputs/features/model-resnet50_normalization-imagenet \
  --clinical-table data/clinical.csv \
  --sample-table data/samples.csv \
  --output-root outputs/training \
  --targets DIAGNOSIS
```

Deploy every fold model:

```bash
python -m pipeline.deployment \
  outputs/training/path/to/crossval_run \
  outputs/features/model-resnet50_normalization-imagenet \
  outputs/deployment \
  data/clinical.csv \
  data/samples.csv \
  --target-label DIAGNOSIS
```

Aggregate fold probabilities:

```bash
python -m ensemble.aggregate_predictions \
  outputs/deployment \
  outputs/ensemble/patient-preds.csv \
  --target-column DIAGNOSIS
```

Generate ROC plots:

```bash
python -m stats.visualize_results outputs/ensemble \
  --target DIAGNOSIS \
  --classes class_a class_b
```

All command paths are supplied at runtime. Feature files store source-image paths relative to the dataset
root so generated artifacts do not disclose a workstation layout.

## Reproducibility checklist

Before creating the archival release:

1. Freeze the exact Python, CUDA, PyTorch, torchvision, fastai, scikit-learn, and Marugoto versions.
2. Record random seeds, split manifests, model-selection rules, and preprocessing thresholds used for the paper.
3. Run the full pipeline from a clean checkout on a small, non-sensitive fixture dataset.
4. Compare the regenerated headline metrics and figures with the submitted manuscript.
5. Add the final paper title, authors, repository URL, DOI, and release version to `CITATION.cff.template`,
   then rename it to `CITATION.cff`.
6. Select a software license and verify that the adapted Marugoto code may be redistributed under it.
7. Tag the paper version and archive that tag with a DOI-providing service.

## Data and privacy

See [DATA_POLICY.md](DATA_POLICY.md). Do not commit clinical tables, split manifests, HDF5 bags,
model exports, raw predictions, or figures that contain sample identifiers.

## Tests

```bash
python -m compileall -q .
python -m unittest discover -s tests
```

The dependency-light tests do not replace an end-to-end validation with the exact research environment.

Only load model exports and saved fold files produced by a trusted run. Fastai learner exports and PyTorch
serialization formats can execute or reconstruct Python objects and are not safe inputs from untrusted sources.

## License and third-party code

No license has been selected in this cleanup copy. Add one only after checking the license and attribution
requirements of every adapted dependency. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
