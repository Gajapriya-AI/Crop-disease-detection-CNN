#!/usr/bin/env python3
"""Download PlantVillage and build a small, reproducible Tomato dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import tarfile
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset"

# The public PlantVillage archive used by TensorFlow Datasets. It contains the
# unaugmented color images and is downloaded to a temporary directory only.
SOURCE_URL = (
    "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/"
    "d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
)
SOURCE_DIRECTORY = "Plant_leave_diseases_dataset_without_augmentation"

# One crop, five classes (four diseases plus healthy).
CLASSES = (
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
)
SPLITS = ("train", "validation", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_DATASET_BYTES = 10_000_000_000  # Keep the prepared subset below 10 GB.
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ArchiveImage:
    """An image member and its portable path inside the source archive."""

    archive_path: str
    member: object


def format_bytes(byte_count: int) -> str:
    """Format a byte count using binary units."""
    size = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def download_archive(destination: Path) -> None:
    """Download the source archive, showing progress and limiting its size."""
    request = Request(
        SOURCE_URL,
        headers={"User-Agent": "Crop-disease-detection-CNN dataset setup"},
    )
    try:
        response = urlopen(request, timeout=60)
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(
            "Could not download PlantVillage. Check your internet connection, "
            "or download the PlantVillage source archive manually and pass it "
            "with --archive."
        ) from error

    with response, destination.open("wb") as output_file:
        length_header = response.headers.get("Content-Length")
        total_bytes = int(length_header) if length_header and length_header.isdigit() else None
        if total_bytes is not None and total_bytes > MAX_DATASET_BYTES:
            raise RuntimeError("The PlantVillage source archive is unexpectedly larger than 10 GB.")

        downloaded = 0
        last_reported = 0
        started = time.monotonic()
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            downloaded += len(chunk)
            if downloaded > MAX_DATASET_BYTES:
                raise RuntimeError("Download stopped because the source archive exceeded 10 GB.")
            output_file.write(chunk)

            if downloaded - last_reported >= 32 * 1024 * 1024:
                if total_bytes:
                    progress = f"{format_bytes(downloaded)} / {format_bytes(total_bytes)}"
                else:
                    progress = f"{format_bytes(downloaded)} downloaded"
                print(f"\rDownloading PlantVillage: {progress}   ", end="", flush=True)
                last_reported = downloaded

        if total_bytes is not None and downloaded != total_bytes:
            raise RuntimeError(
                f"The download was incomplete ({downloaded} of {total_bytes} bytes). "
                "Run the command again to retry."
            )

    elapsed = max(time.monotonic() - started, 0.001)
    print(
        f"\rDownloaded PlantVillage archive ({format_bytes(downloaded)} "
        f"in {elapsed:.0f}s).                         "
    )


def selected_class(member_name: str) -> Optional[str]:
    """Return the selected class for a source image, otherwise None."""
    portable_name = member_name.replace("\\", "/")
    parts = PurePosixPath(portable_name).parts
    try:
        source_index = parts.index(SOURCE_DIRECTORY)
    except ValueError:
        return None

    if source_index + 2 >= len(parts):
        return None
    class_name = parts[source_index + 1]
    if class_name not in CLASSES:
        return None
    if PurePosixPath(parts[-1]).suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    return class_name


def collect_zip_images(archive: zipfile.ZipFile) -> dict[str, list[ArchiveImage]]:
    grouped = {class_name: [] for class_name in CLASSES}
    for member in archive.infolist():
        if member.is_dir():
            continue
        class_name = selected_class(member.filename)
        if class_name is not None:
            grouped[class_name].append(ArchiveImage(member.filename, member))
    return grouped


def collect_tar_images(archive: tarfile.TarFile) -> dict[str, list[ArchiveImage]]:
    grouped = {class_name: [] for class_name in CLASSES}
    for member in archive.getmembers():
        if not member.isfile():
            continue
        class_name = selected_class(member.name)
        if class_name is not None:
            grouped[class_name].append(ArchiveImage(member.name, member))
    return grouped


def split_images(
    grouped: dict[str, list[ArchiveImage]], seed: int
) -> dict[str, dict[str, list[ArchiveImage]]]:
    """Create deterministic, per-class 70/15/15 splits."""
    rng = random.Random(seed)
    assignments: dict[str, dict[str, list[ArchiveImage]]] = {}

    for class_name in CLASSES:
        images = sorted(
            grouped[class_name],
            key=lambda image: (image.archive_path.casefold(), image.archive_path),
        )
        if not images:
            raise RuntimeError(
                f"No images found for {class_name}. The source archive may have "
                "a different folder layout or version."
            )

        rng.shuffle(images)
        train_end = len(images) * 70 // 100
        validation_end = train_end + len(images) * 15 // 100
        per_split = {
            "train": images[:train_end],
            "validation": images[train_end:validation_end],
            "test": images[validation_end:],
        }
        if any(not per_split[split] for split in SPLITS):
            raise RuntimeError(
                f"Class {class_name} has only {len(images)} images; at least 7 are "
                "needed to populate all three splits."
            )
        assignments[class_name] = per_split

    return assignments


def unique_filename(filename: str, archive_path: str, used_names: set[str]) -> str:
    """Avoid overwriting images if the archive repeats a basename."""
    if filename not in used_names:
        used_names.add(filename)
        return filename

    path = PurePosixPath(filename)
    digest = hashlib.sha256(archive_path.encode("utf-8")).hexdigest()[:10]
    candidate = f"{path.stem}__{digest}{path.suffix}"
    counter = 2
    while candidate in used_names:
        candidate = f"{path.stem}__{digest}_{counter}{path.suffix}"
        counter += 1
    used_names.add(candidate)
    return candidate


def write_dataset(
    grouped: dict[str, list[ArchiveImage]],
    open_member: Callable[[ArchiveImage], BinaryIO],
    staging: Path,
    seed: int,
) -> dict[str, dict[str, int]]:
    assignments = split_images(grouped, seed)
    counts = {split: {class_name: 0 for class_name in CLASSES} for split in SPLITS}

    for split in SPLITS:
        for class_name in CLASSES:
            (staging / split / class_name).mkdir(parents=True, exist_ok=True)

    for class_name in CLASSES:
        for split in SPLITS:
            target_dir = staging / split / class_name
            used_names: set[str] = set()
            for image in assignments[class_name][split]:
                basename = PurePosixPath(image.archive_path.replace("\\", "/")).name
                filename = unique_filename(basename, image.archive_path, used_names)
                destination = target_dir / filename
                with open_member(image) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output, length=CHUNK_SIZE)
                counts[split][class_name] += 1

    metadata = {
        "source": "PlantVillage",
        "source_url": SOURCE_URL,
        "source_subdirectory": SOURCE_DIRECTORY,
        "crop": "Tomato",
        "classes": list(CLASSES),
        "seed": seed,
        "split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "split_method": "Stratified per class; image-level shuffle with Python random.Random(seed).",
        "images_per_split": counts,
        "total_images": sum(sum(class_counts.values()) for class_counts in counts.values()),
    }
    (staging / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    dataset_bytes = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
    if dataset_bytes >= MAX_DATASET_BYTES:
        raise RuntimeError(
            f"The prepared subset is {format_bytes(dataset_bytes)}, which is not below 10 GB. "
            "The incomplete output was discarded."
        )

    return counts


def populate_from_archive(archive_path: Path, staging: Path, seed: int) -> dict[str, dict[str, int]]:
    """Read only selected original color images from a ZIP or TAR archive."""
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            grouped = collect_zip_images(archive)
            return write_dataset(
                grouped,
                lambda image: archive.open(image.member, "r"),
                staging,
                seed,
            )

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, mode="r:*") as archive:
            grouped = collect_tar_images(archive)

            def open_tar_member(image: ArchiveImage) -> BinaryIO:
                source = archive.extractfile(image.member)
                if source is None:
                    raise OSError(f"Could not read archive member {image.archive_path}")
                return source

            return write_dataset(grouped, open_tar_member, staging, seed)

    raise RuntimeError(
        f"{archive_path} is not a supported ZIP or TAR archive. "
        "Download the PlantVillage archive from the linked source and try again."
    )


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def install_dataset(staging: Path, output: Path, force: bool) -> None:
    """Replace the target only after the new dataset has been built successfully."""
    exists = output.exists() or output.is_symlink()
    if exists and not force:
        raise FileExistsError(
            f"{output} already exists. To replace it, rerun with --force."
        )

    if not exists:
        staging.rename(output)
        return

    backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
    output.rename(backup)
    try:
        staging.rename(output)
    except OSError:
        backup.rename(output)
        raise
    remove_path(backup)


def prepare_dataset(archive_path: Path, output: Path, seed: int, force: bool) -> dict[str, dict[str, int]]:
    if (output.exists() or output.is_symlink()) and not force:
        raise FileExistsError(f"{output} already exists. To replace it, rerun with --force.")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        counts = populate_from_archive(archive_path, staging, seed)
        install_dataset(staging, output, force)
        return counts
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download PlantVillage and create a reproducible five-class Tomato "
            "dataset in train/validation/test folders."
        )
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an already-downloaded PlantVillage ZIP/TAR archive instead of downloading it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the reproducible per-class split (default: 42).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory after the new subset is ready.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    archive_argument = args.archive.expanduser().resolve() if args.archive else None

    git_dir = PROJECT_ROOT / ".git"
    inside_git = False
    try:
        output.relative_to(git_dir)
        inside_git = True
    except ValueError:
        pass
    if (
        output == output.parent
        or output == PROJECT_ROOT
        or PROJECT_ROOT.is_relative_to(output)
        or inside_git
    ):
        print(
            "Error: choose a dataset output directory, not a filesystem root, "
            "repository ancestor, repository root, or .git path.",
            file=sys.stderr,
        )
        return 2
    if (output.exists() or output.is_symlink()) and not args.force:
        print(f"Error: {output} already exists. Rerun with --force to replace it.", file=sys.stderr)
        return 2

    try:
        if archive_argument is not None:
            if not archive_argument.is_file():
                raise FileNotFoundError(f"Source archive not found: {archive_argument}")
            print(f"Using local PlantVillage archive: {archive_argument}")
            counts = prepare_dataset(archive_argument, output, args.seed, args.force)
        else:
            # The full source archive is not left in the repository or retained
            # after setup; only the selected five-class subset is installed.
            with tempfile.TemporaryDirectory(prefix="plantvillage-download-") as temp_dir:
                archive_path = Path(temp_dir) / "plantvillage-source"
                print("Downloading the PlantVillage source archive...")
                download_archive(archive_path)
                counts = prepare_dataset(archive_path, output, args.seed, args.force)
    except (OSError, RuntimeError, ValueError, URLError, zipfile.BadZipFile, tarfile.TarError) as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 1

    print(f"\nDataset ready: {output}")
    print(f"Seed: {args.seed}; split: 70% train / 15% validation / 15% test")
    for split in SPLITS:
        summary = ", ".join(
            f"{class_name.removeprefix('Tomato___')}={counts[split][class_name]}"
            for class_name in CLASSES
        )
        print(f"  {split:10s} {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
