# Crop Disease Detection with CNN + MobileNetV3

Detect diseases in **tomato leaves** from images using deep learning. The project trains and compares two models on a five-class PlantVillage subset:

1. **Custom CNN baseline** - a small convolutional network built from scratch with TensorFlow/Keras
2. **MobileNetV3 transfer learning** - the main model, fine-tuned from ImageNet weights

It also includes a **Streamlit web app** (`app.py`) for uploading a leaf image and getting the predicted disease with a confidence percentage, plus a **command-line predictor** (`src/inference.py`).

| Class folder | Disease |
| --- | --- |
| `Tomato___healthy` | Healthy |
| `Tomato___Early_blight` | Early blight |
| `Tomato___Late_blight` | Late blight |
| `Tomato___Leaf_Mold` | Leaf Mold |
| `Tomato___Septoria_leaf_spot` | Septoria leaf spot |

All images are resized to **224 x 224** pixels and trained with **batch size 16** and **data augmentation** (random flip, rotation, zoom, translation and contrast).

## Project structure

```text
Crop-disease-detection-CNN/
├── app.py                    # Streamlit web app (upload image -> prediction)
├── requirements.txt          # Python dependencies
├── README.md
├── scripts/
│   └── prepare_dataset.py    # Downloads PlantVillage and builds dataset/ (do not modify)
├── src/
│   ├── __init__.py
│   ├── config.py             # Shared constants: paths, image size, batch size, classes
│   ├── data.py               # Dataset loading + augmentation pipeline
│   ├── train_cnn.py          # Trains the custom CNN baseline
│   ├── train_mobilenet.py    # Trains MobileNetV3 (main model, transfer learning)
│   ├── evaluate.py           # Test-set comparison: accuracy/precision/recall/F1 + confusion matrices
│   └── inference.py          # Single-image prediction (CLI, used by the app)
├── models/                   # Trained models are saved here (NOT committed to Git)
├── notebooks/
│   └── explore_dataset.ipynb # Dataset exploration + sample prediction notebook
├── reports/                  # Generated metrics, confusion matrices, curves (NOT in Git)
├── tests/                    # Unit tests (structure, config, data, inference)
└── dataset/                  # Created locally by scripts/prepare_dataset.py (NOT in Git)
    ├── train/        (one folder per class)
    ├── validation/
    └── test/
```

The generated folders `dataset/`, `models/*` and `reports/*` are excluded by `.gitignore` - dataset images and model binaries must never be committed to GitHub.

## 1. Installation (Windows / VS Code)

1. Install **Python 3.9 - 3.12** (check with `py --version`) and open this repository folder in **VS Code**.
2. Open **Terminal -> New Terminal** and make sure it is at the repository root (the folder containing `README.md`).
3. (Recommended) Create and use a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, just call the environment's Python directly (`.\.venv\Scripts\python.exe ...`) or use `py -m pip ...` as shown below.

4. Install the dependencies:

   ```powershell
   py -m pip install -r requirements.txt
   ```

   The `tensorflow` package installs the CPU version by default, which is enough for this project (training just takes longer than on a GPU).

On macOS / Linux use `python3` instead of `py`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 2. Prepare the dataset

The setup script uses the **PlantVillage** dataset, specifically its original, unaugmented color images. To keep the local dataset small, it selects five Tomato classes only (see the table above).

The prepared images are split separately within each class into **70% train, 15% validation, and 15% test**. The default seed is `42`; the script sorts filenames before applying the seeded shuffle, so rerunning it with the same source archive and seed produces the same split. This is an image-level split, not a leaf-group split.

The script downloads the PlantVillage source archive (about 828 MiB according to the TensorFlow Datasets catalog) into a temporary system folder, copies only the five selected classes, and removes the temporary archive when it finishes. The resulting subset is checked to remain below 10 GB. The generated `dataset/` directory is excluded by `.gitignore` and must not be committed or uploaded to GitHub.

Run it once from the repository root:

```powershell
py scripts\prepare_dataset.py
```

The script uses only Python's standard library; no extra `pip install` step is required for it. Allow time and temporary disk space for the source archive download.

### Optional dataset commands

- Use a different reproducible seed:

  ```powershell
  py scripts\prepare_dataset.py --seed 123
  ```

- If you already downloaded the PlantVillage ZIP/TAR archive, skip the script's download. Replace the example path with the archive's actual Windows path:

  ```powershell
  py scripts\prepare_dataset.py --archive "C:\Users\YourName\Downloads\plantvillage.zip"
  ```

- To regenerate the subset when `dataset/` already exists, pass `--force`. The current folder is replaced only after the new dataset has been built successfully:

  ```powershell
  py scripts\prepare_dataset.py --force
  ```

### Output layout

Each split contains one folder per class, as expected by common image-folder loaders (including Keras' `image_dataset_from_directory`):

```text
dataset/
├── metadata.json
├── train/
│   ├── Tomato___healthy/
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   ├── Tomato___Leaf_Mold/
│   └── Tomato___Septoria_leaf_spot/
├── validation/  (the same five class folders)
└── test/        (the same five class folders)
```

`metadata.json` records the source, selected labels, seed, split ratios, and per-class image counts.

Source links:

- [PlantVillage dataset record on Mendeley Data](https://data.mendeley.com/datasets/tywbtsjrjv/1)
- [Direct PlantVillage source archive](https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded) (used by the script)
- [TensorFlow Datasets PlantVillage catalog](https://www.tensorflow.org/datasets/catalog/plant_village)
- Original paper: Mohanty, Hughes & Salathé, [“Using Deep Learning for Image-Based Plant Disease Detection”](https://doi.org/10.3389/fpls.2016.01419)

Review the source dataset's license and attribution terms before using or redistributing the images. This repository contains only the setup script, not PlantVillage images.

## 3. Train the models

Both scripts print progress, save the **best model** (highest validation accuracy) into `models/`, and write the training history and curves into `reports/`. Run them from the repository root.

### 3a. Custom CNN baseline

```powershell
py src\train_cnn.py
```

- Default: 25 epochs, batch size 16, with early stopping (patience 5) and learning-rate reduction.
- Saves: `models/cnn_baseline.keras`, `reports/history_cnn.json`, `reports/training_curves_cnn.png`

### 3b. MobileNetV3 transfer learning (main model)

```powershell
py src\train_mobilenet.py
```

Training happens in two phases:

1. **Phase 1 (frozen base):** only the new 5-class head trains on top of `MobileNetV3Small` with ImageNet weights.
2. **Phase 2 (fine-tuning):** the last 30 layers of the base are unfrozen and trained with a much smaller learning rate.

The single best model across both phases is saved to `models/mobilenet_v3_best.keras`, plus `reports/history_mobilenet_v3.json` and `reports/training_curves_mobilenet_v3.png`.

Useful options (both scripts):

```powershell
py src\train_cnn.py --epochs 5                 # quick test run
py src\train_mobilenet.py --epochs 6 --fine-tune-epochs 3
py src\train_mobilenet.py --skip-fine-tune     # phase 1 only
py src\train_cnn.py --batch-size 8             # less RAM/VRAM per step
```

The first MobileNetV3 run downloads ~10 MB of pre-trained weights, so keep the internet on for that run.

## 4. Evaluate and compare both models

```powershell
py src\evaluate.py
```

This evaluates every model found in `models/` on the untouched **test split** and reports:

- **Accuracy**
- **Precision, Recall, F1-score** (macro average, weighted average and per class)
- **Confusion matrices** as labelled figures

Generated files in `reports/`:

| File | Content |
| --- | --- |
| `classification_report_cnn.txt` / `classification_report_mobilenet_v3.txt` | Per-class precision/recall/F1 |
| `confusion_matrix_cnn.png` / `confusion_matrix_mobilenet_v3.png` | Confusion matrices (counts + row-normalized) |
| `metrics_comparison.csv` / `metrics_comparison.md` | Side-by-side comparison, best values in bold |

You can point it at specific model files:

```powershell
py src\evaluate.py --cnn-model models\cnn_baseline.keras --mobilenet-model models\mobilenet_v3_best.keras
```

## 5. Predict a single image (command line)

```powershell
py src\inference.py dataset\test\Tomato___Leaf_Mold\some_leaf_image.jpg
```

Example output:

```text
Image: dataset\test\Tomato___Leaf_Mold\some_leaf_image.jpg
Predicted class : Tomato___Leaf_Mold
Crop            : Tomato
Disease         : Leaf Mold
Confidence      : 99.2%
All probabilities:
    Leaf Mold                99.20%
    Septoria leaf spot        0.40%
    ...
```

Use the CNN baseline instead of MobileNetV3 with `--model`:

```powershell
py src\inference.py path\to\leaf.jpg --model models\cnn_baseline.keras
```

The tool validates the path, the file type and the model file before predicting, and prints clear instructions when something is missing (for example when the model has not been trained yet).

## 6. Streamlit web app

Start the app from the repository root:

```powershell
py -m streamlit run app.py
```

The browser opens automatically at `http://localhost:8501`. In the app you can:

- Upload a leaf image (JPG, JPEG, PNG, BMP, WEBP)
- See the predicted **disease name** and **confidence percentage** (with a progress bar)
- See the probability of **every class**
- Switch between the MobileNetV3 model and the CNN baseline in the sidebar

Stop the app with `Ctrl+C` in the terminal. If no trained model exists yet, the app shows the exact training command instead of crashing.

## 7. Run the tests

The test suite checks the project structure, the configuration, the dataset helpers, the inference helpers and the original preparation script. From the repository root:

```powershell
py -m unittest discover -s tests
```

Tests that need TensorFlow are skipped automatically when TensorFlow is not installed; they never download the real dataset (a tiny synthetic folder is used instead).

## 8. Notebook

`notebooks/explore_dataset.ipynb` shows how to count the images per class, display samples, and run a prediction with the trained model. Open it in VS Code (Jupyter extension) or with `jupyter notebook`, and select your virtual environment as the kernel.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `DatasetNotFoundError` / missing `dataset/train` | Run `py scripts\prepare_dataset.py` from the repository root |
| `Trained model not found: models\mobilenet_v3_best.keras` | Run `py src\train_mobilenet.py` first |
| `TensorFlow is not installed` | Run `py -m pip install -r requirements.txt` |
| PowerShell blocks `.venv` activation | Use `py -m pip ...` / `py src\...` without activation, or run `.\.venv\Scripts\python.exe src\train_cnn.py` |
| Training is very slow | Normal on CPU. Reduce epochs (`--epochs 5`) or train on a machine with a GPU; TensorFlow uses CUDA automatically when available |
| Out-of-memory errors | Lower the batch size: `py src\train_cnn.py --batch-size 8` |
| Port 8501 already in use (Streamlit) | Run `py -m streamlit run app.py --server.port 8600` |

## Notes

- Models (`*.keras`), reports and the dataset are generated locally and stay out of Git (see `.gitignore`).
- All code uses `pathlib` paths and relative project-root detection, so the same commands work on Windows, macOS and Linux.
- This project is for research/education; it is not a substitute for expert agricultural advice.
