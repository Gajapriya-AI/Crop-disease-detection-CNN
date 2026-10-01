"""Train the MobileNetV3 transfer-learning model (the MAIN model of the project).

Transfer learning means: we take MobileNetV3Small, a small network that has
already learned general image features on ImageNet, freeze it, and train only a
new classification head for our 5 tomato classes.  Afterwards we "fine-tune"
the top layers of the base so they specialize on leaf images.

Two training phases
-------------------
Phase 1 -- base FROZEN:      only the new head trains (fast, stable start)
Phase 2 -- base fine-tuned:  the last ``MOBILENET_FINE_TUNE_LAYERS`` layers of
                             the base are unfrozen and trained with a very
                             small learning rate

The single best model across BOTH phases (highest validation accuracy) is
saved to ``models/mobilenet_v3_best.keras``.  History and curves go to
``reports/``.

Usage (from the project root):

    Windows (VS Code terminal):   py src\\train_mobilenet.py
    macOS / Linux:                python3 src/train_mobilenet.py

Useful options:

    --epochs 4 --fine-tune-epochs 2   shorter training (quick test)
    --skip-fine-tune                  only run phase 1
    --batch-size 8                    smaller batches (less memory)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make "from src import ..." work when this file is run directly
# (``py src\train_mobilenet.py``) as well as a module (``python -m src.train_mobilenet``).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reduce TensorFlow C++ log noise so beginners see the important messages.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # save figures to files without opening a window
import matplotlib.pyplot as plt  # noqa: E402
import tensorflow as tf  # noqa: E402

from src import config, data  # noqa: E402


def build_mobilenet_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    """Create and compile MobileNetV3Small + a new classification head.

    Notes for beginners:

    * ``include_top=False`` removes ImageNet's 1000-class head; we add our own
      5-class head instead.
    * ``pooling="avg"`` turns the base's final feature map into one feature
      vector per image (global average pooling).
    * ``base.trainable = False`` freezes all pre-trained weights for phase 1.
    * ``base(x, training=False)`` keeps BatchNormalization layers inside the
      base in "inference mode" - important so the pre-trained statistics are
      not destroyed while fine-tuning.
    """
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",   # download pre-trained ImageNet weights
        pooling="avg",
    )
    base.trainable = False    # phase 1: freeze the whole base

    inputs = tf.keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    # NOTE: there is deliberately NO extra Rescaling layer here!
    # In Keras 3, MobileNetV3 already contains its own preprocessing layer
    # (``include_preprocessing=True`` by default): it expects raw pixel
    # values in 0..255 - exactly what image_dataset_from_directory and our
    # inference helper produce - and converts them internally to the -1..1
    # range the ImageNet weights were trained with. Adding another
    # Rescaling(1/255) in front of it would feed the base almost constant
    # values and the model would not learn anything.
    x = data.build_augmentation_pipeline()(inputs)   # training only
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE)(x)
    outputs = tf.keras.layers.Dense(config.NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenet_v3_small")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.MOBILENET_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def unfreeze_top_layers(base: tf.keras.Model, num_layers: int) -> None:
    """Unfreeze the last ``num_layers`` layers of the base for fine-tuning."""
    base.trainable = True
    # Freeze everything except the final ``num_layers`` layers.
    for layer in base.layers[:-num_layers]:
        layer.trainable = False
    trainable = sum(layer.trainable for layer in base.layers)
    print(f"Fine-tuning: {trainable} of {len(base.layers)} base layers are trainable.")


def save_history(history_entries: list, model_name: str) -> None:
    """Save the (possibly multi-phase) training history as JSON + curve plot."""
    config.ensure_dirs()

    merged: dict[str, list[float]] = {}
    for history in history_entries:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(value) for value in values)

    json_path = config.REPORTS_DIR / f"history_{model_name}.json"
    json_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Training history saved to {json_path}")

    fig, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(merged["loss"], label="train loss")
    loss_axis.plot(merged["val_loss"], label="validation loss")
    loss_axis.set_title(f"Loss ({model_name})")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.legend()

    accuracy_axis.plot(merged["accuracy"], label="train accuracy")
    accuracy_axis.plot(merged["val_accuracy"], label="validation accuracy")
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
        description="Train MobileNetV3 (transfer learning) for tomato leaf disease detection."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.MOBILENET_EPOCHS,
        help=f"Phase-1 epochs with frozen base (default: {config.MOBILENET_EPOCHS}).",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=config.MOBILENET_FINE_TUNE_EPOCHS,
        help=f"Phase-2 fine-tuning epochs (default: {config.MOBILENET_FINE_TUNE_EPOCHS}).",
    )
    parser.add_argument(
        "--skip-fine-tune",
        action="store_true",
        help="Only train the head (phase 1) and skip fine-tuning.",
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

        # 2. Load the datasets.
        print("Loading datasets ...")
        train_ds, val_ds, _ = data.load_all_datasets(batch_size=args.batch_size)

        # 3. Build the model. The first run downloads the pre-trained
        #    MobileNetV3Small weights (~10 MB) - this needs internet access.
        print("Building MobileNetV3Small + classification head ...")
        model, base = build_mobilenet_model()
        model.summary()

        config.ensure_dirs()

        # 4. ONE checkpoint object is used for both phases, so the file
        #    models/mobilenet_v3_best.keras always contains the best model
        #    seen across phase 1 AND phase 2.
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MOBILENET_MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )

        histories = []

        # ---------------- Phase 1: frozen base, train the head -------------
        print(f"\n=== Phase 1: training the head for up to {args.epochs} epochs ===")
        phase1_early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=config.MOBILENET_EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )
        histories.append(
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.epochs,
                callbacks=[checkpoint, phase1_early_stop],
                # The tf.data training set already reshuffles itself each epoch.
                shuffle=False,
            )
        )

        # ---------------- Phase 2: fine-tune the top layers ----------------
        if not args.skip_fine_tune and args.fine_tune_epochs > 0:
            print(
                f"\n=== Phase 2: fine-tuning the last "
                f"{config.MOBILENET_FINE_TUNE_LAYERS} base layers "
                f"for up to {args.fine_tune_epochs} epochs ==="
            )
            unfreeze_top_layers(base, config.MOBILENET_FINE_TUNE_LAYERS)
            # Recompile with a MUCH smaller learning rate: fine-tuning should
            # gently adapt the pre-trained weights, not destroy them.
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=config.MOBILENET_FINE_TUNE_LEARNING_RATE
                ),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            phase2_early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=config.MOBILENET_EARLY_STOP_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            )
            histories.append(
                model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=args.fine_tune_epochs,
                    callbacks=[checkpoint, phase2_early_stop],
                    # The tf.data training set already reshuffles itself each epoch.
                    shuffle=False,
                )
            )
        else:
            print("\nFine-tuning skipped (--skip-fine-tune).")

        # 5. Save history + curves (file names come from src/config.py).
        save_history(histories, model_name="mobilenet_v3")

        best_accuracy = max(
            max(history.history["val_accuracy"]) for history in histories
        )
        print("\nTraining finished.")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        print(f"Best model saved to:      {config.MOBILENET_MODEL_PATH}")
        print("Next step: compare both models with 'py src\\evaluate.py'.")
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
