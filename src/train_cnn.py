"""Train the custom CNN baseline model (built from scratch, no transfer learning).

The network is a small stack of convolution blocks::

    Input 224x224x3
      -> Rescaling(1/255)          # pixel values 0..255 -> 0..1
      -> Augmentation              # active only while training
      -> [Conv2D + MaxPool + Dropout] x 3   (32, 64, 128 filters)
      -> Flatten -> Dense(128) -> Dropout -> Dense(5, softmax)

Usage (from the project root):

    Windows (VS Code terminal):   py src\\train_cnn.py
    macOS / Linux:                python3 src/train_cnn.py

Useful options:

    --epochs 5        train for fewer epochs (quick test)
    --batch-size 8    use a smaller batch size (less RAM / GPU memory)

The best model (highest validation accuracy) is saved to
``models/cnn_baseline.keras``; the training history and curves are saved to
``reports/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make "from src import ..." work when this file is run directly
# (``py src\train_cnn.py``) as well as a module (``python -m src.train_cnn``).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reduce TensorFlow C++ log noise so beginners see the important messages.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # save figures to files without opening a window
import matplotlib.pyplot as plt  # noqa: E402
import tensorflow as tf  # noqa: E402

from src import config, data  # noqa: E402


def build_cnn_model(batch_size: int = config.BATCH_SIZE) -> tf.keras.Model:
    """Create and compile the custom CNN baseline.

    The model starts with a ``Rescaling`` layer, so it accepts raw image
    pixels in the 0..255 range (exactly what ``image_dataset_from_directory``
    produces) and normalizes them internally.  The augmentation layers are
    part of the model but are automatically inactive during validation,
    testing and inference.
    """
    inputs = tf.keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))

    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)          # 0..255 -> 0..1
    x = data.build_augmentation_pipeline()(x)                 # training only

    # Three convolution blocks: more filters as the image gets smaller.
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(
            filters, kernel_size=3, padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)

    # Classification head.
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(config.NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="custom_cnn_baseline")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.CNN_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history(history: tf.keras.callbacks.History, model_name: str) -> None:
    """Save the training history as JSON + a loss/accuracy plot (into reports/)."""
    config.ensure_dirs()

    # Keras stores numpy floats -> convert to plain Python floats for JSON.
    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    json_path = config.REPORTS_DIR / f"history_{model_name}.json"
    json_path.write_text(json.dumps(history_dict, indent=2), encoding="utf-8")
    print(f"Training history saved to {json_path}")

    # Plot loss and accuracy curves side by side.
    fig, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(history.history["loss"], label="train loss")
    loss_axis.plot(history.history["val_loss"], label="validation loss")
    loss_axis.set_title(f"Loss ({model_name})")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.legend()

    accuracy_axis.plot(history.history["accuracy"], label="train accuracy")
    accuracy_axis.plot(history.history["val_accuracy"], label="validation accuracy")
    accuracy_axis.set_title(f"Accuracy ({model_name})")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.legend()

    figure_path = config.REPORTS_DIR / f"training_curves_{model_name}.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the custom CNN baseline for tomato leaf disease detection."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.CNN_EPOCHS,
        help=f"Number of training epochs (default: {config.CNN_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Batch size (default: {config.BATCH_SIZE}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        # 1. Make sure the dataset exists (friendly error otherwise).
        config.require_dataset()

        # 2. Load the three splits.
        print("Loading datasets ...")
        train_ds, val_ds, _ = data.load_all_datasets(batch_size=args.batch_size)

        # 3. Build the model and show its structure.
        print("Building the custom CNN ...")
        model = build_cnn_model()
        model.summary()

        # 4. Callbacks: save the best model, stop early, lower LR on plateaus.
        config.ensure_dirs()
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.CNN_MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=config.CNN_EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        )

        # 5. Train.
        print(f"\nTraining for up to {args.epochs} epochs (batch size {args.batch_size}) ...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            # The tf.data training set already reshuffles itself each epoch,
            # so fit() must not shuffle again (silences a Keras warning).
            shuffle=False,
        )

        # 6. Save history + curves into reports/.
        save_history(history, model_name="cnn")

        best_accuracy = max(history.history["val_accuracy"])
        print("\nTraining finished.")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        print(f"Best model saved to:      {config.CNN_MODEL_PATH}")
        print("Next step: run 'py src\\train_mobilenet.py' and then 'py src\\evaluate.py'.")
        return 0

    except config.DatasetNotFoundError as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError) as error:
        print(f"\nTraining failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nTraining stopped by user (Ctrl+C).", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
