"""Predict the disease of a single tomato-leaf image.

This module powers both the command-line tool and the Streamlit app:

* :func:`validate_image_path` -- friendly checks BEFORE TensorFlow is needed
* :class:`DiseasePredictor`   -- loads a trained model and predicts one image
* command line (``main``)     -- ``py src\\inference.py path\\to\\leaf.jpg``

TensorFlow is imported *lazily* (only when a prediction actually runs).  That
keeps this module importable in tests and tools even on machines where
TensorFlow is not installed yet, and it makes the "model file is missing"
error fast and clear.

Usage (from the project root):

    Windows (VS Code terminal):   py src\\inference.py dataset\\test\\Tomato___Leaf_Mold\\some_leaf.jpg
    macOS / Linux:                python3 src/inference.py dataset/test/Tomato___Leaf_Mold/some_leaf.jpg

Options:

    --model PATH    use a specific model file
                    (default: models/mobilenet_v3_best.keras)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make "from src import ..." work when this file is run directly
# (``py src\inference.py``) as well as a module (``python -m src.inference``).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402


# ---------------------------------------------------------------------------
# Pure helpers (no TensorFlow needed) -- easy to unit-test
# ---------------------------------------------------------------------------

def validate_image_path(image_path: str | Path) -> Path:
    """Check that ``image_path`` points to an existing, supported image file.

    Raises
    ------
    FileNotFoundError
        When the file does not exist (message shows the exact path).
    ValueError
        When the path is a folder or the extension is not supported.
    """
    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Image file not found: {path}\n"
            "Check the path and try again (drag the file into the terminal "
            "to paste its full path)."
        )
    if path.is_dir():
        raise ValueError(
            f"The path is a folder, not an image file: {path}\n"
            "Point the command at one image, e.g. dataset\\test\\Tomato___healthy\\leaf1.jpg"
        )
    if path.suffix.lower() not in config.SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(config.SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type '{path.suffix}' for: {path}\n"
            f"Supported image extensions: {supported}"
        )
    return path.resolve()


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Return the model path to use, raising a friendly error when missing."""
    path = Path(model_path).expanduser() if model_path else config.DEFAULT_MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {path}\n"
            "Train the MobileNetV3 model first (from the project root):\n"
            "    Windows (VS Code terminal):  py src\\train_mobilenet.py\n"
            "    macOS / Linux:               python3 src/train_mobilenet.py"
        )
    return path.resolve()


def format_prediction(result: dict) -> str:
    """Turn a prediction dictionary into a human-readable multi-line string."""
    lines = [
        f"Predicted class : {result['class_name']}",
        f"Crop            : {result['crop']}",
        f"Disease         : {result['disease']}",
        f"Confidence      : {result['confidence_percent']:.1f}%",
    ]
    if result["probabilities"]:
        lines.append("All probabilities:")
        ranked = sorted(
            result["probabilities"].items(), key=lambda item: item[1], reverse=True
        )
        for class_name, probability in ranked:
            lines.append(
                f"    {config.disease_name(class_name):22s} {probability * 100:6.2f}%"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Predictor (TensorFlow is imported only inside these functions)
# ---------------------------------------------------------------------------

def _import_tensorflow():
    """Import TensorFlow lazily with a beginner-friendly error message."""
    try:
        import tensorflow as tf  # noqa: PLC0415 (intentional local import)

        return tf
    except ImportError as error:  # pragma: no cover - environment problem
        raise ImportError(
            "TensorFlow is not installed. Install the requirements first:\n"
            "    Windows (VS Code terminal):  py -m pip install -r requirements.txt\n"
            "    macOS / Linux:               python3 -m pip install -r requirements.txt"
        ) from error


class DiseasePredictor:
    """Loads a trained model once and predicts as many images as you like.

    Example
    -------
    >>> predictor = DiseasePredictor()                    # default model
    >>> result = predictor.predict("leaf.jpg")
    >>> print(result["disease"], result["confidence_percent"])
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        tf = _import_tensorflow()
        path = resolve_model_path(model_path)
        try:
            self.model = tf.keras.models.load_model(path)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"The model file could not be loaded: {path}\n"
                f"Details: {error}\n"
                "It may be corrupted or saved by an incompatible Keras version - "
                "retrain it with 'py src\\train_mobilenet.py'."
            ) from error
        self.model_path = path

    def predict(self, image, filename: str | None = None) -> dict:
        """Predict one image.

        Parameters
        ----------
        image:
            A path (``str`` / ``Path``) to an image file, or an already-open
            binary file-like object (e.g. ``io.BytesIO`` from a web upload).
        filename:
            Optional original file name, used for extension checks when
            ``image`` is a file-like object.

        Returns
        -------
        dict with keys:
            ``class_name``        e.g. "Tomato___Early_blight"
            ``crop``              e.g. "Tomato"
            ``disease``           e.g. "Early blight" (or "Healthy")
            ``is_healthy``        True/False
            ``confidence``        0..1
            ``confidence_percent``0..100
            ``probabilities``     {class_name: probability, ...}
        """
        tf = _import_tensorflow()

        # 1. Validate the input (friendly errors before doing any real work).
        if isinstance(image, (str, Path)):
            source = validate_image_path(image)
        else:
            # File-like object (Streamlit upload): check the name if given.
            if filename is not None:
                suffix = Path(filename).suffix.lower()
                if suffix not in config.SUPPORTED_IMAGE_EXTENSIONS:
                    supported = ", ".join(sorted(config.SUPPORTED_IMAGE_EXTENSIONS))
                    raise ValueError(
                        f"Unsupported file type '{suffix}'. Supported: {supported}"
                    )
            source = image

        # 2. Load and preprocess the image exactly like the training pipeline:
        #    RGB, 224x224, pixel values 0..255 (the model rescales internally).
        try:
            loaded = tf.keras.utils.load_img(
                source, target_size=(config.IMG_SIZE, config.IMG_SIZE)
            )
        except Exception as error:
            raise ValueError(
                "The file could not be read as an image. It may be corrupted "
                f"or not a real image file. Details: {error}"
            ) from error
        array = tf.keras.utils.img_to_array(loaded)      # shape (224, 224, 3)
        batch = tf.expand_dims(array, axis=0)            # shape (1, 224, 224, 3)

        # 3. Predict. The model ends with softmax -> probabilities sum to 1.
        probabilities = self.model.predict(batch, verbose=0)[0]

        best_index = int(probabilities.argmax())
        class_name = config.CLASSES[best_index]
        confidence = float(probabilities[best_index])

        return {
            "class_name": class_name,
            "crop": config.crop_name(class_name),
            "disease": config.disease_name(class_name),
            "is_healthy": config.is_healthy(class_name),
            "confidence": confidence,
            "confidence_percent": confidence * 100.0,
            "probabilities": {
                name: float(probability)
                for name, probability in zip(config.CLASSES, probabilities)
            },
        }


def predict_image(
    image_path: str | Path, model_path: str | Path | None = None
) -> dict:
    """Convenience one-shot function: load the model, predict, return the dict."""
    predictor = DiseasePredictor(model_path)
    return predictor.predict(image_path)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the disease of one tomato-leaf image."
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the leaf image (JPG, JPEG, PNG, BMP or WEBP).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model file to use (default: "
            f"{config.DEFAULT_MODEL_PATH.name} inside models/)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = predict_image(args.image, args.model)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except ImportError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 3

    print(f"Image: {args.image}")
    print(format_prediction(result))

    if result["is_healthy"]:
        print("\nGood news: this leaf looks HEALTHY.")
    else:
        print(f"\nDisease detected: {result['disease']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
