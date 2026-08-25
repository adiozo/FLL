import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ensemble.aggregate_predictions import aggregate_fold_predictions


class AggregatePredictionTests(unittest.TestCase):
    def test_aggregate_fold_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for fold, probabilities in enumerate(([0.8, 0.3], [0.6, 0.5])):
                directory = root / f"fold-{fold}"
                directory.mkdir()
                pd.DataFrame(
                    {
                        "PATIENT": ["sample_a", "sample_b"],
                        "DIAGNOSIS": ["class_a", "class_b"],
                        "DIAGNOSIS_class_a": probabilities,
                        "DIAGNOSIS_class_b": [1 - value for value in probabilities],
                    }
                ).to_csv(directory / "patient-preds.csv", index=False)

            output = root / "aggregate.csv"
            result = aggregate_fold_predictions(root, output)

            self.assertTrue(output.exists())
            for actual, expected in zip(result["DIAGNOSIS_class_a"], [0.7, 0.4]):
                self.assertAlmostEqual(actual, expected)
            for actual, expected in zip(result["DIAGNOSIS_class_b"], [0.3, 0.6]):
                self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
