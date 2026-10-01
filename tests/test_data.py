"""Tests for src/data.py (dataset loading + augmentation).

These tests need TensorFlow and Pillow.  When TensorFlow is not installed the
whole module is skipped, so ``py -m unittest discover -s tests`` still works
on a fresh checkout before ``pip install -r requirements.txt``.

The tests build a TINY synthetic dataset in a temporary folder (a few 32x32
coloured images per class) - they never touch the real ``dataset/`` folder
and never download anything.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:  # pragma: no cover - depends on the environment
        return False


@unittest.skipUnless(tensorflow_available(), "TensorFlow is not installed")
class DataPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import tensorflow as tf
        from PIL import Image

        cls.tf = tf

        # Tiny synthetic dataset: 4 images per class per split, 32x32 pixels.
        # Each class gets its own colour so batches are easy to reason about.
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls.dataset_dir = Path(cls._temp_dir.name) / "dataset"
        colours = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30), (128, 60, 200)]
        for split in ("train", "validation", "test"):
            for class_name, colour in zip(config.CLASSES, colours):
                class_dir = cls.dataset_dir / split / class_name
                class_dir.mkdir(parents=True)
                for index in range(4):
                    Image.new("RGB", (32, 32), color=colour).save(
                        class_dir / f"leaf_{index}.png"
                    )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_dir.cleanup()

    def test_missing_split_raises_friendly_error(self) -> None:
        missing = self.dataset_dir / "does_not_exist"
        with self.assertRaises(FileNotFoundError) as context:
            config.require_dataset(missing)
        self.assertIn("prepare_dataset.py", str(context.exception))

    def test_augmentation_pipeline_has_expected_layers(self) -> None:
        from src import data

        pipeline = data.build_augmentation_pipeline()
        layer_names = [layer.__class__.__name__ for layer in pipeline.layers]
        self.assertIn("RandomFlip", layer_names)
        self.assertIn("RandomRotation", layer_names)
        self.assertIn("RandomZoom", layer_names)

    def test_load_split_shapes_and_labels(self) -> None:
        from src import data

        train_dir = self.dataset_dir / "train"
        dataset = data.load_split(train_dir, batch_size=4, shuffle=False)

        images, labels = next(iter(dataset))
        self.assertEqual(tuple(images.shape[1:]), (config.IMG_SIZE, config.IMG_SIZE, 3))
        self.assertEqual(tuple(labels.shape[1:]), (config.NUM_CLASSES,))
        # One-hot labels: exactly one 1 per row.
        self.assertTrue(all(int(row.sum()) == 1 for row in labels.numpy()))

    def test_load_all_datasets_respects_custom_directory(self) -> None:
        from src import data

        train_ds, val_ds, test_ds = data.load_all_datasets(
            batch_size=config.BATCH_SIZE, dataset_dir=self.dataset_dir
        )
        for dataset in (train_ds, val_ds, test_ds):
            images, labels = next(iter(dataset))
            self.assertEqual(images.shape[-1], 3)
            self.assertEqual(labels.shape[-1], config.NUM_CLASSES)

    def test_load_all_datasets_missing_folder_raises(self) -> None:
        from src import data

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(config.DatasetNotFoundError):
                data.load_all_datasets(dataset_dir=Path(temp_dir))


if __name__ == "__main__":
    unittest.main()
