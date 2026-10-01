"""Shared project configuration: paths, constants and small helper functions.

Everything that other modules need to agree on (image size, batch size, class
names, file locations ...) is defined HERE, in one place.  If you want to
experiment with a different image size or batch size, change it in this file
and every script picks the new value up automatically.

This module intentionally uses only the Python standard library, so it can be
imported (and tested) even when TensorFlow is not installed yet.

All paths are built with :mod:`pathlib`, which makes them work identically on
Windows (``C:\\Users\\...``), macOS and Linux (``/home/...``).
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project locations
# ---------------------------------------------------------------------------

# ``config.py`` lives in ``src/``, so the project root is one level up.
# ``resolve()`` turns the path into an absolute path on every operating system.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Dataset folder created by ``scripts/prepare_dataset.py`` (not stored in Git).
DATASET_DIR: Path = PROJECT_ROOT / "dataset"
TRAIN_DIR: Path = DATASET_DIR / "train"
VALIDATION_DIR: Path = DATASET_DIR / "validation"
TEST_DIR: Path = DATASET_DIR / "test"

# Folders for saved models and generated reports.
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# Classes (must match the folder names inside dataset/train, dataset/...)
# ---------------------------------------------------------------------------

CLASSES: list[str] = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
]
NUM_CLASSES: int = len(CLASSES)

# The crop name is encoded in every class folder ("Tomato___<disease>").
CROP_NAME: str = "Tomato"

# ---------------------------------------------------------------------------
# Model / training hyper-parameters
# ---------------------------------------------------------------------------

IMG_SIZE: int = 224          # every image is resized to 224 x 224 pixels
BATCH_SIZE: int = 16         # images per training step
SEED: int = 42               # fixed seed -> reproducible shuffles and splits

# Custom CNN baseline
CNN_EPOCHS: int = 25
CNN_LEARNING_RATE: float = 1e-3
CNN_EARLY_STOP_PATIENCE: int = 5

# MobileNetV3 transfer learning (the main model).
# Phase 1 trains only the new classification head (base frozen),
# phase 2 "fine-tunes" the top layers of the base with a much smaller
# learning rate.
MOBILENET_EPOCHS: int = 12
MOBILENET_FINE_TUNE_EPOCHS: int = 8
MOBILENET_LEARNING_RATE: float = 1e-3
MOBILENET_FINE_TUNE_LEARNING_RATE: float = 1e-5
MOBILENET_FINE_TUNE_LAYERS: int = 30   # unfreeze the last 30 layers of the base
MOBILENET_EARLY_STOP_PATIENCE: int = 4
DROPOUT_RATE: float = 0.3

# ---------------------------------------------------------------------------
# Model files (trained models are large -> they stay local and are NOT in Git)
# ---------------------------------------------------------------------------

CNN_MODEL_PATH: Path = MODELS_DIR / "cnn_baseline.keras"
MOBILENET_MODEL_PATH: Path = MODELS_DIR / "mobilenet_v3_best.keras"

# The model used by ``inference.py`` and the Streamlit app by default.
DEFAULT_MODEL_PATH: Path = MOBILENET_MODEL_PATH

# ---------------------------------------------------------------------------
# Report files produced by the training and evaluation scripts
# ---------------------------------------------------------------------------

CNN_HISTORY_PATH: Path = REPORTS_DIR / "history_cnn.json"
MOBILENET_HISTORY_PATH: Path = REPORTS_DIR / "history_mobilenet_v3.json"
CNN_CURVES_PATH: Path = REPORTS_DIR / "training_curves_cnn.png"
MOBILENET_CURVES_PATH: Path = REPORTS_DIR / "training_curves_mobilenet_v3.png"
CNN_CLASSIFICATION_REPORT_PATH: Path = REPORTS_DIR / "classification_report_cnn.txt"
MOBILENET_CLASSIFICATION_REPORT_PATH: Path = (
    REPORTS_DIR / "classification_report_mobilenet_v3.txt"
)
CNN_CONFUSION_MATRIX_PATH: Path = REPORTS_DIR / "confusion_matrix_cnn.png"
MOBILENET_CONFUSION_MATRIX_PATH: Path = REPORTS_DIR / "confusion_matrix_mobilenet_v3.png"
COMPARISON_CSV_PATH: Path = REPORTS_DIR / "metrics_comparison.csv"
COMPARISON_MD_PATH: Path = REPORTS_DIR / "metrics_comparison.md"

# Image file extensions accepted by the inference helpers.
SUPPORTED_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class DatasetNotFoundError(FileNotFoundError):
    """Raised when the prepared PlantVillage dataset folder is missing."""


def require_dataset(dataset_dir: Path = DATASET_DIR) -> Path:
    """Check that ``dataset/train``, ``dataset/validation`` and ``dataset/test`` exist.

    Raises a friendly :class:`DatasetNotFoundError` (with the exact command to
    fix the problem) when the dataset has not been prepared yet.  Every script
    that needs the dataset calls this function first, so beginners get a clear
    message instead of a confusing TensorFlow stack trace.
    """
    missing = [
        str(dataset_dir / split)
        for split in ("train", "validation", "test")
        if not (dataset_dir / split).is_dir()
    ]
    if missing:
        raise DatasetNotFoundError(
            "The dataset was not found. Missing folder(s): "
            + ", ".join(missing)
            + "\nPrepare it first by running this command from the project root:\n"
            "    Windows (VS Code terminal):  py scripts\\prepare_dataset.py\n"
            "    macOS / Linux:               python3 scripts/prepare_dataset.py"
        )
    return dataset_dir


def disease_name(class_name: str) -> str:
    """Convert a class folder name into a human-readable disease name.

    Examples
    --------
    >>> disease_name("Tomato___Early_blight")
    'Early blight'
    >>> disease_name("Tomato___healthy")
    'Healthy'
    """
    # Folder names look like "Tomato___Early_blight": crop, three underscores,
    # then the disease with single underscores between words.
    disease = class_name.split("___", 1)[-1].replace("_", " ")
    if disease.lower() == "healthy":
        return "Healthy"
    return disease


def crop_name(class_name: str) -> str:
    """Return the crop part of a class folder name ("Tomato")."""
    return class_name.split("___", 1)[0]


def display_label(class_name: str) -> str:
    """Build the label shown to users, e.g. ``Tomato - Early blight``."""
    disease = disease_name(class_name)
    if disease == "Healthy":
        return f"{crop_name(class_name)} - Healthy (no disease)"
    return f"{crop_name(class_name)} - {disease}"


def is_healthy(class_name: str) -> bool:
    """Return True when the class represents a healthy leaf."""
    return class_name.lower().endswith("healthy")


def ensure_dirs() -> None:
    """Create the ``models/`` and ``reports/`` folders if they do not exist yet."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
