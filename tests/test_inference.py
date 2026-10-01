"""Tests for src/inference.py.

The pure helper functions (path validation, output formatting, model-path
checks) are tested without TensorFlow.  The end-to-end prediction test needs
TensorFlow and a tiny temporary model; it is skipped automatically when
TensorFlow is not installed.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, inference  # noqa: E402


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:  # pragma: no cover - depends on the environment
        return False


class ValidateImagePathTests(unittest.TestCase):
    def test_missing_file_raises_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "nope.jpg"
            with self.assertRaises(FileNotFoundError) as context:
                inference.validate_image_path(missing)
            self.assertIn("not found", str(context.exception).lower())

    def test_folder_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                inference.validate_image_path(temp_dir)

    def test_unsupported_extension_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            text_file = Path(temp_dir) / "leaf.txt"
            text_file.write_text("not an image")
            with self.assertRaises(ValueError) as context:
                inference.validate_image_path(text_file)
            self.assertIn("Unsupported file type", str(context.exception))

    def test_valid_image_path_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_file = Path(temp_dir) / "leaf.JPG"  # uppercase extension
            image_file.write_bytes(b"fake-bytes")
            self.assertEqual(
                inference.validate_image_path(image_file), image_file.resolve()
            )


class ResolveModelPathTests(unittest.TestCase):
    def test_missing_model_raises_friendly_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "model.keras"
            with self.assertRaises(FileNotFoundError) as context:
                inference.resolve_model_path(missing)
            message = str(context.exception)
            self.assertIn("train_mobilenet.py", message)

    def test_existing_model_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_file = Path(temp_dir) / "model.keras"
            model_file.write_bytes(b"fake-model")
            self.assertEqual(
                inference.resolve_model_path(model_file), model_file.resolve()
            )


class FormatPredictionTests(unittest.TestCase):
    def test_format_prediction_contains_key_information(self) -> None:
        result = {
            "class_name": "Tomato___Leaf_Mold",
            "crop": "Tomato",
            "disease": "Leaf Mold",
            "is_healthy": False,
            "confidence": 0.913,
            "confidence_percent": 91.3,
            "probabilities": {
                "Tomato___healthy": 0.05,
                "Tomato___Early_blight": 0.02,
                "Tomato___Late_blight": 0.007,
                "Tomato___Leaf_Mold": 0.913,
                "Tomato___Septoria_leaf_spot": 0.01,
            },
        }
        text = inference.format_prediction(result)
        self.assertIn("Leaf Mold", text)
        self.assertIn("91.3%", text)
        self.assertIn("Tomato___Leaf_Mold", text)
        # Highest probability first in the listing.
        self.assertLess(text.index("Leaf Mold"), text.index("Septoria leaf spot"))


@unittest.skipUnless(tensorflow_available(), "TensorFlow is not installed")
class PredictionEndToEndTests(unittest.TestCase):
    """Build a tiny throw-away model and check the full prediction path."""

    @classmethod
    def setUpClass(cls) -> None:
        import tensorflow as tf

        cls.tf = tf
        cls._temp_dir = tempfile.TemporaryDirectory()
        temp = Path(cls._temp_dir.name)

        # A minimal model with the same interface as the real ones:
        # input 224x224x3 (0..255 pixels), softmax output over 5 classes.
        inputs = tf.keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
        x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(config.NUM_CLASSES, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        cls.model_path = temp / "tiny_model.keras"
        model.save(cls.model_path)

        # A small valid PNG image (solid colour, 32x32 - it gets resized).
        from PIL import Image

        cls.image_path = temp / "leaf.png"
        Image.new("RGB", (32, 32), color=(40, 160, 60)).save(cls.image_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_dir.cleanup()

    def test_predictor_returns_wellformed_result(self) -> None:
        predictor = inference.DiseasePredictor(self.model_path)
        result = predictor.predict(self.image_path)

        self.assertIn(result["class_name"], config.CLASSES)
        self.assertEqual(result["crop"], "Tomato")
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertAlmostEqual(
            result["confidence_percent"], result["confidence"] * 100.0, places=6
        )
        self.assertEqual(len(result["probabilities"]), config.NUM_CLASSES)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=4)
        self.assertEqual(
            result["disease"], config.disease_name(result["class_name"])
        )
        self.assertEqual(
            result["is_healthy"], config.is_healthy(result["class_name"])
        )

    def test_predictor_accepts_file_like_object(self) -> None:
        import io

        predictor = inference.DiseasePredictor(self.model_path)
        with self.image_path.open("rb") as image_file:
            result = predictor.predict(
                io.BytesIO(image_file.read()), filename=self.image_path.name
            )
        self.assertIn(result["class_name"], config.CLASSES)

    def test_predictor_rejects_bad_filename_extension(self) -> None:
        import io

        predictor = inference.DiseasePredictor(self.model_path)
        with self.assertRaises(ValueError):
            predictor.predict(io.BytesIO(b"whatever"), filename="notes.txt")

    def test_predictor_missing_model_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            inference.DiseasePredictor(self.model_path.parent / "missing.keras")

    def test_predictor_corrupt_model_raises_value_error(self) -> None:
        corrupt = Path(self._temp_dir.name) / "corrupt.keras"
        corrupt.write_bytes(b"this is not a keras model")
        with self.assertRaises(ValueError):
            inference.DiseasePredictor(corrupt)


if __name__ == "__main__":
    unittest.main()
