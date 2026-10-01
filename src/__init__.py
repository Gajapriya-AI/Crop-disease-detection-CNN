"""Crop disease detection package.

This package contains every piece of the deep-learning pipeline:

- ``config``            -- shared constants (paths, image size, batch size, class names)
- ``data``              -- dataset loading + augmentation pipeline (TensorFlow/Keras)
- ``train_cnn``         -- trains the custom CNN baseline model
- ``train_mobilenet``   -- trains the MobileNetV3 transfer-learning model (main model)
- ``evaluate``          -- compares both models (accuracy, precision, recall, F1,
                           confusion matrices) and writes the results into ``reports/``
- ``inference``         -- predicts the disease of a single leaf image

All modules use :mod:`pathlib` paths, so they work the same way on Windows,
macOS and Linux.
"""

__version__ = "1.0.0"
