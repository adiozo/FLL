import tempfile
import unittest
from pathlib import Path

from util.dataset_statistics import summarize_dataset


class DatasetStatisticTests(unittest.TestCase):
    def test_summary_contains_no_sample_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sample = root / "class_a" / "sample_hidden"
            sample.mkdir(parents=True)
            (sample / "image_1.png").touch()
            (sample / "image_2.png").touch()

            summary = summarize_dataset(root)

            self.assertEqual(summary["total_samples"], 1)
            self.assertEqual(summary["total_images"], 2)
            self.assertNotIn("sample_hidden", repr(summary))


if __name__ == "__main__":
    unittest.main()
