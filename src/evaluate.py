"""Evaluate and compare the trained models on the TEST split.

For every model found in ``models/`` this script computes:

* Accuracy
* Precision, Recall and F1-score (macro average, weighted average and per class)
* A confusion matrix (saved as a labelled PNG figure)

and writes the results into ``reports/``:

* ``classification_report_<model>.txt``  -- full per-class report
* ``confusion_matrix_<model>.png``       -- confusion matrix figure
* ``metrics_comparison.csv`` / ``.md``   -- side-by-side model comparison

Usage (from the project root):

    Windows (VS Code terminal):   py src\\evaluate.py
    macOS / Linux:                python3 src/evaluate.py

Options:

    --cnn-model PATH        evaluate a specific CNN model file
    --mobilenet-model PATH  evaluate a specific MobileNetV3 model file
    --batch-size N          batch size used for prediction (default 16)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Make "from src import ..." work when this file is run directly.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless figure saving (works on Windows and in VS Code)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src import config, data  # noqa: E402


def predict_test_set(model: tf.keras.Model, test_ds: tf.data.Dataset):
    """Run the model over the whole test set in a single pass.

    Returns
    -------
    (y_true, y_pred, probabilities)
        ``y_true`` / ``y_pred`` are integer class indices (positions inside
        ``config.CLASSES``); ``probabilities`` is the raw softmax output.
    """
    true_chunks: list[np.ndarray] = []
    probability_chunks: list[np.ndarray] = []

    for images, labels in test_ds:
        # training=False -> dropout and augmentation are inactive.
        probabilities = model(images, training=False)
        probability_chunks.append(np.asarray(probabilities))
        true_chunks.append(np.argmax(np.asarray(labels), axis=1))

    y_true = np.concatenate(true_chunks)
    probabilities = np.concatenate(probability_chunks)
    y_pred = np.argmax(probabilities, axis=1)
    return y_true, y_pred, probabilities


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Draw a confusion matrix as a colour-coded PNG figure."""
    # Short labels keep the figure readable (drop the "Tomato___" prefix).
    short_names = [config.disease_name(name) for name in class_names]

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, mat, subtitle in (
        (axes[0], matrix, "Counts"),
        (axes[1], matrix.astype("float") / np.maximum(matrix.sum(axis=1, keepdims=True), 1), "Row-normalized"),
    ):
        image = axis.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
        axis.set_title(f"{title} ({subtitle})")
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(range(len(short_names)))
        axis.set_yticks(range(len(short_names)))
        axis.set_xticklabels(short_names, rotation=30, ha="right")
        axis.set_yticklabels(short_names)
        threshold = mat.max() / 2.0
        for row in range(mat.shape[0]):
            for column in range(mat.shape[1]):
                value = mat[row, column]
                text = f"{value:.2f}" if subtitle == "Row-normalized" else f"{int(value)}"
                axis.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=9,
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Confusion matrix saved to {output_path}")


def evaluate_model(
    model_name: str,
    model_path: Path,
    test_ds: tf.data.Dataset,
    report_path: Path,
    confusion_path: Path,
) -> dict[str, float] | None:
    """Evaluate one saved model; returns a metrics dictionary (or None)."""
    if not model_path.is_file():
        print(
            f"[skip] {model_name}: model file not found ({model_path}).\n"
            f"       Train it first, e.g.  py src\\train_mobilenet.py"
        )
        return None

    print(f"\n=== Evaluating {model_name} ({model_path.name}) ===")
    model = tf.keras.models.load_model(model_path)
    y_true, y_pred, _ = predict_test_set(model, test_ds)

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # Full per-class report (precision / recall / f1 / support for every class).
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=[config.disease_name(name) for name in config.CLASSES],
        digits=4,
        zero_division=0,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"Classification report - {model_name}\nModel file: {model_path}\n\n{report_text}",
        encoding="utf-8",
    )
    print(report_text)
    print(f"Classification report saved to {report_path}")

    # Confusion matrix figure.
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    plot_confusion_matrix(matrix, config.CLASSES, f"{model_name} - test set", confusion_path)

    return metrics


def write_comparison(rows: list[dict[str, float]], csv_path: Path, md_path: Path) -> None:
    """Save the side-by-side comparison of all evaluated models."""
    if not rows:
        print("\nNo models were evaluated - nothing to compare.")
        return

    metric_keys = [key for key in rows[0] if key != "model"]

    # CSV version (easy to open in Excel).
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric"] + [row["model"] for row in rows])
        for key in metric_keys:
            writer.writerow([key] + [f"{row[key]:.4f}" for row in rows])
    print(f"Comparison CSV saved to {csv_path}")

    # Markdown version (readable on GitHub).
    lines = [
        "# Model comparison (test set)",
        "",
        "| Metric | " + " | ".join(row["model"] for row in rows) + " |",
        "| --- | " + " | ".join("---" for _ in rows) + " |",
    ]
    for key in metric_keys:
        values = [row[key] for row in rows]
        best = max(values)
        cells = [
            f"**{value:.4f}**" if value == best else f"{value:.4f}"
            for value in values
        ]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")

    best_model = max(rows, key=lambda row: row["f1_macro"])["model"]
    lines += [
        "",
        f"Best model by macro F1-score: **{best_model}**",
        "",
        "See also the per-class reports (`classification_report_*.txt`) and the",
        "confusion matrices (`confusion_matrix_*.png`) in this folder.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparison Markdown saved to {md_path}")

    # Console summary table.
    print("\n=== Summary ===")
    header = f"{'metric':20s}" + "".join(f"{row['model']:>22s}" for row in rows)
    print(header)
    print("-" * len(header))
    for key in metric_keys:
        print(f"{key:20s}" + "".join(f"{row[key]:>22.4f}" for row in rows))
    print(f"\nBest model by macro F1-score: {best_model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare the CNN baseline and MobileNetV3 models."
    )
    parser.add_argument(
        "--cnn-model",
        type=Path,
        default=config.CNN_MODEL_PATH,
        help=f"CNN model file (default: {config.CNN_MODEL_PATH}).",
    )
    parser.add_argument(
        "--mobilenet-model",
        type=Path,
        default=config.MOBILENET_MODEL_PATH,
        help=f"MobileNetV3 model file (default: {config.MOBILENET_MODEL_PATH}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Batch size for prediction (default: {config.BATCH_SIZE}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        # Both model files missing -> nothing to evaluate, explain how to fix.
        if not args.cnn_model.is_file() and not args.mobilenet_model.is_file():
            print(
                "Error: no trained models were found.\n"
                f"  Expected CNN model:       {args.cnn_model}\n"
                f"  Expected MobileNet model: {args.mobilenet_model}\n"
                "Train them first (from the project root):\n"
                "    py src\\train_cnn.py\n"
                "    py src\\train_mobilenet.py",
                file=sys.stderr,
            )
            return 2

        # Load the TEST split (never shuffled -> stable results).
        config.require_dataset()
        print("Loading the test dataset ...")
        test_ds = data.load_split(
            config.TEST_DIR, batch_size=args.batch_size, shuffle=False
        )

        config.ensure_dirs()
        rows: list[dict[str, float]] = []

        cnn_metrics = evaluate_model(
            "CNN baseline",
            args.cnn_model,
            test_ds,
            config.CNN_CLASSIFICATION_REPORT_PATH,
            config.CNN_CONFUSION_MATRIX_PATH,
        )
        if cnn_metrics is not None:
            rows.append(cnn_metrics)

        mobilenet_metrics = evaluate_model(
            "MobileNetV3",
            args.mobilenet_model,
            test_ds,
            config.MOBILENET_CLASSIFICATION_REPORT_PATH,
            config.MOBILENET_CONFUSION_MATRIX_PATH,
        )
        if mobilenet_metrics is not None:
            rows.append(mobilenet_metrics)

        write_comparison(rows, config.COMPARISON_CSV_PATH, config.COMPARISON_MD_PATH)
        return 0

    except config.DatasetNotFoundError as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError) as error:
        print(f"\nEvaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
