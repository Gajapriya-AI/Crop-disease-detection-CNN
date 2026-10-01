"""Dataset loading and data-augmentation pipeline.

This module wraps :func:`tf.keras.utils.image_dataset_from_directory`, which
reads the folder layout produced by ``scripts/prepare_dataset.py``::

    dataset/
        train/       (one folder per class, images inside)
        validation/
        test/

It also defines the *data augmentation* pipeline used while training.  All
paths come from :mod:`src.config`, so the code works unchanged on Windows,
macOS and Linux.

Run order reminder (from the project root):

1. ``py scripts\\prepare_dataset.py``   -- create the dataset (already done)
2. ``py src\\train_cnn.py``             -- train the CNN baseline
3. ``py src\\train_mobilenet.py``       -- train MobileNetV3 (main model)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make "from src import config" work when this file is executed directly
# (e.g. ``py src\data.py``) as well as when it is imported as a package module.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402  (import after the sys.path fix above)

from src import config  # noqa: E402


def build_augmentation_pipeline() -> tf.keras.Sequential:
    """Return the data-augmentation layers applied to TRAINING images only.

    Data augmentation creates slightly modified copies of the training images
    (flipped, rotated, zoomed, ...).  This makes the model robust to different
    camera angles and lighting, and helps prevent overfitting.

    These Keras preprocessing layers are *smart*: they only transform images
    while ``training=True`` (i.e. inside ``model.fit``).  During validation,
    testing and inference they pass images through unchanged, so the same
    layers can safely live inside the saved model.
    """
    return tf.keras.Sequential(
        [
            # Randomly mirror half of the images horizontally.
            tf.keras.layers.RandomFlip("horizontal"),
            # Rotate by up to +/-10% of a full turn (= +/-36 degrees).
            tf.keras.layers.RandomRotation(0.10),
            # Randomly zoom in/out by up to 10%.
            tf.keras.layers.RandomZoom(0.10),
            # Randomly shift the image by up to 10% of its width/height.
            tf.keras.layers.RandomTranslation(0.10, 0.10),
            # Randomly change the contrast by up to 10%.
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )


def load_split(
    split_dir: Path,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
    augment: bool = False,
) -> tf.data.Dataset:
    """Load one dataset split (``train``, ``validation`` or ``test``).

    Parameters
    ----------
    split_dir:
        Folder that contains one sub-folder per class, e.g. ``dataset/train``.
    batch_size:
        Number of images per batch (16 by default, as required).
    shuffle:
        Shuffle the images each epoch (True for training, False for test so
        that predictions always come back in the same order).
    augment:
        Kept for completeness, but augmentation normally lives *inside* the
        model (see :func:`build_augmentation_pipeline`), so this is False.

    Raises
    ------
    FileNotFoundError
        With a friendly message when ``split_dir`` does not exist.
    """
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {split_dir}\n"
            "Run the dataset preparation script first:\n"
            "    Windows (VS Code terminal):  py scripts\\prepare_dataset.py\n"
            "    macOS / Linux:               python3 scripts/prepare_dataset.py"
        )

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=str(split_dir),
        labels="inferred",                 # class = folder name
        label_mode="categorical",          # one-hot encoded labels
        class_names=config.CLASSES,        # FIXED label order for every split
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        shuffle=shuffle,
        seed=config.SEED if shuffle else None,
        interpolation="bilinear",
    )

    if augment:
        augmentation = build_augmentation_pipeline()
        dataset = dataset.map(
            lambda images, labels: (augmentation(images, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Prefetch lets the CPU prepare the next batch while the model trains on
    # the current one -> faster training.
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def load_all_datasets(
    batch_size: int = config.BATCH_SIZE,
    dataset_dir: Path = config.DATASET_DIR,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load the train, validation and test datasets.

    Returns
    -------
    (train_ds, val_ds, test_ds)
        Training data is shuffled; validation and test data are not, so
        evaluation results are stable and reproducible.
    """
    # Fail early with a clear message when the dataset has not been prepared.
    config.require_dataset(Path(dataset_dir))

    dataset_dir = Path(dataset_dir)
    train_ds = load_split(dataset_dir / "train", batch_size=batch_size, shuffle=True)
    val_ds = load_split(dataset_dir / "validation", batch_size=batch_size, shuffle=False)
    test_ds = load_split(dataset_dir / "test", batch_size=batch_size, shuffle=False)
    return train_ds, val_ds, test_ds


def summarize_dataset(dataset_dir: Path = config.DATASET_DIR) -> None:
    """Print how many images each split/class contains (nice sanity check)."""
    dataset_dir = Path(dataset_dir)
    config.require_dataset(dataset_dir)
    print(f"Dataset folder: {dataset_dir}")
    for split in ("train", "validation", "test"):
        split_dir = dataset_dir / split
        total = 0
        print(f"  {split}:")
        for class_name in config.CLASSES:
            class_dir = split_dir / class_name
            count = (
                sum(
                    1
                    for path in class_dir.iterdir()
                    if path.suffix.lower() in config.SUPPORTED_IMAGE_EXTENSIONS
                )
                if class_dir.is_dir()
                else 0
            )
            total += count
            print(f"    {class_name:35s} {count:5d} images")
        print(f"    {'TOTAL':35s} {total:5d} images")


if __name__ == "__main__":
    # Allow a quick check with:  py src\data.py
    summarize_dataset()
