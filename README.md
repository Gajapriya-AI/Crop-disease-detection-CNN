# Crop-disease-detection-CNN

Project for crop disease classification with CNNs and transfer learning. This repository includes a lightweight PlantVillage tomato-leaf dataset setup; images are downloaded and prepared on the user's computer, not stored in Git.

## Dataset

The setup script uses the **PlantVillage** dataset, specifically its original, unaugmented color images. To keep the local dataset small, it selects five Tomato classes only:

| Folder / label | Meaning |
| --- | --- |
| `Tomato___healthy` | Healthy |
| `Tomato___Early_blight` | Early blight |
| `Tomato___Late_blight` | Late blight |
| `Tomato___Leaf_Mold` | Leaf mold |
| `Tomato___Septoria_leaf_spot` | Septoria leaf spot |

The prepared images are split separately within each class into **70% train, 15% validation, and 15% test**. The default seed is `42`; the script sorts filenames before applying the seeded shuffle, so rerunning it with the same source archive and seed produces the same split. This is an image-level split, not a leaf-group split.

The script downloads the PlantVillage source archive (about 828 MiB according to the TensorFlow Datasets catalog) into a temporary system folder, copies only the five selected classes, and removes the temporary archive when it finishes. The resulting subset is checked to remain below 10 GB. The generated `dataset/` directory is excluded by `.gitignore` and must not be committed or uploaded to GitHub.

Source links:

- [PlantVillage dataset record on Mendeley Data](https://data.mendeley.com/datasets/tywbtsjrjv/1)
- [Direct PlantVillage source archive](https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded) (used by the script)
- [TensorFlow Datasets PlantVillage catalog](https://www.tensorflow.org/datasets/catalog/plant_village)
- Original paper: Mohanty, Hughes & Salathé, [“Using Deep Learning for Image-Based Plant Disease Detection”](https://doi.org/10.3389/fpls.2016.01419)

Review the source dataset's license and attribution terms before using or redistributing the images. This repository contains only the setup script, not PlantVillage images.

## Prepare the dataset (Windows / VS Code)

1. Install Python 3.9 or newer and open this repository folder in VS Code.
2. In VS Code, open **Terminal → New Terminal**. Make sure the terminal is at the repository root (the folder containing `README.md`).
3. Run this exact command in the integrated PowerShell terminal:

   ```powershell
   py scripts\prepare_dataset.py
   ```

The script uses only Python's standard library; no `pip install` step is required. It downloads PlantVillage, keeps the five Tomato classes above, and creates the local folder structure below. Allow time and temporary disk space for the source archive download. The full archive is not kept in the repository.

If you prefer to use a virtual environment, run these first from the same VS Code terminal, then run the dataset command above:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can skip activation and run the dataset command with the environment's Python instead:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py
```

### Optional commands

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

The script can also be run outside Windows, for example with `python scripts/prepare_dataset.py` from the repository root.

## Output layout

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

`metadata.json` records the source, selected labels, seed, split ratios, and per-class image counts. Example Keras loader paths are `dataset/train`, `dataset/validation`, and `dataset/test`.

## Verify the setup script

The lightweight tests use a small synthetic archive and do not download PlantVillage:

```powershell
py -m unittest discover -s tests
```
