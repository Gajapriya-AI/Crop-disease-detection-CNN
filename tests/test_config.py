"""Tests for src/config.py (standard library only - no TensorFlow needed)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402


class ConfigConstantTests(unittest.TestCase):
    def test_image_size_is_224(self) -> None:
        self.assertEqual(config.IMG_SIZE, 224)

    def test_batch_size_is_16(self) -> None:
        self.assertEqual(config.BATCH_SIZE, 16)

    def test_five_expected_classes(self) -> None:
        self.assertEqual(
            config.CLASSES,
            [
                "Tomato___healthy",
                "Tomato___Early_blight",
                "Tomato___Late_blight",
                "Tomato___Leaf_Mold",
                "Tomato___Septoria_leaf_spot",
            ],
        )
        self.assertEqual(config.NUM_CLASSES, 5)

    def test_paths_are_inside_project_root(self) -> None:
        for path in (
            config.DATASET_DIR,
            config.TRAIN_DIR,
            config.VALIDATION_DIR,
            config.TEST_DIR,
            config.MODELS_DIR,
            config.REPORTS_DIR,
            config.CNN_MODEL_PATH,
            config.MOBILENET_MODEL_PATH,
        ):
            self.assertIsInstance(path, Path)
            self.assertTrue(
                str(path).startswith(str(PROJECT_ROOT)),
                f"{path} should live inside the project root",
            )

    def test_dataset_split_paths_use_dataset_folder(self) -> None:
        self.assertEqual(config.TRAIN_DIR, config.DATASET_DIR / "train")
        self.assertEqual(config.VALIDATION_DIR, config.DATASET_DIR / "validation")
        self.assertEqual(config.TEST_DIR, config.DATASET_DIR / "test")


class RequireDatasetTests(unittest.TestCase):
    def test_missing_dataset_raises_friendly_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(config.DatasetNotFoundError) as context:
                config.require_dataset(Path(temp_dir))
            message = str(context.exception)
            self.assertIn("prepare_dataset.py", message)

    def test_incomplete_dataset_raises_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "train").mkdir()
            # validation/ and test/ are missing on purpose.
            with self.assertRaises(config.DatasetNotFoundError):
                config.require_dataset(root)

    def test_complete_dataset_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for split in ("train", "validation", "test"):
                (root / split).mkdir()
            self.assertEqual(config.require_dataset(root), root)


class LabelHelperTests(unittest.TestCase):
    def test_disease_name(self) -> None:
        self.assertEqual(config.disease_name("Tomato___healthy"), "Healthy")
        self.assertEqual(config.disease_name("Tomato___Early_blight"), "Early blight")
        self.assertEqual(config.disease_name("Tomato___Late_blight"), "Late blight")
        self.assertEqual(config.disease_name("Tomato___Leaf_Mold"), "Leaf Mold")
        self.assertEqual(
            config.disease_name("Tomato___Septoria_leaf_spot"), "Septoria leaf spot"
        )

    def test_crop_name(self) -> None:
        for class_name in config.CLASSES:
            self.assertEqual(config.crop_name(class_name), "Tomato")

    def test_is_healthy(self) -> None:
        self.assertTrue(config.is_healthy("Tomato___healthy"))
        for class_name in config.CLASSES:
            if class_name != "Tomato___healthy":
                self.assertFalse(config.is_healthy(class_name))

    def test_display_label(self) -> None:
        self.assertEqual(
            config.display_label("Tomato___Early_blight"), "Tomato - Early blight"
        )
        self.assertIn("Healthy", config.display_label("Tomato___healthy"))


if __name__ == "__main__":
    unittest.main()
