import unittest

import pandas as pd

from stats.metrics import bootstrap_macro_ovr_auc, calculate_threshold


def prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TARGET": ["a", "a", "b", "b", "a", "b"],
            "TARGET_a": [0.9, 0.8, 0.2, 0.1, 0.7, 0.3],
            "TARGET_b": [0.1, 0.2, 0.8, 0.9, 0.3, 0.7],
        }
    )


class MetricTests(unittest.TestCase):
    def test_threshold_is_finite(self) -> None:
        threshold, true_positive_rate, false_positive_rate = calculate_threshold(
            prediction_frame(), ["a", "b"], "TARGET", "b"
        )
        self.assertEqual(threshold, threshold)
        self.assertTrue(0 <= true_positive_rate <= 1)
        self.assertTrue(0 <= false_positive_rate <= 1)

    def test_bootstrap_is_reproducible(self) -> None:
        first = bootstrap_macro_ovr_auc(
            prediction_frame(), "TARGET", ["a", "b"], n_bootstrap=25, random_state=7
        )
        second = bootstrap_macro_ovr_auc(
            prediction_frame(), "TARGET", ["a", "b"], n_bootstrap=25, random_state=7
        )
        self.assertEqual(first["macro_auc"], second["macro_auc"])
        self.assertTrue((first["macro_ci"] == second["macro_ci"]).all())


if __name__ == "__main__":
    unittest.main()
