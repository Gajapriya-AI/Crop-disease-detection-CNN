import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.prepare_dataset import CLASSES, SOURCE_DIRECTORY  # noqa: E402


class PrepareDatasetTests(unittest.TestCase):
    def make_source_archive(self, path: Path) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            for class_index, class_name in enumerate(CLASSES):
                for image_index in range(20):
                    member = (
                        f"{SOURCE_DIRECTORY}/{class_name}/"
                        f"leaf_{image_index:03d}.jpg"
                    )
                    archive.writestr(member, f"{class_index}:{image_index}".encode())

            # These must not be included in the selected unaugmented subset.
            archive.writestr(
                f"{SOURCE_DIRECTORY}/Tomato___Bacterial_spot/ignored.jpg", b"ignored"
            )
            archive.writestr(
                "Plant_leave_diseases_dataset_with_augmentation/"
                f"{CLASSES[0]}/augmented.jpg",
                b"ignored",
            )

    def run_setup(self, archive: Path, output: Path, *extra_args: str) -> None:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_dataset.py"),
            "--archive",
            str(archive),
            "--output",
            str(output),
            "--seed",
            "73",
            *extra_args,
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

    @staticmethod
    def image_manifest(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*.jpg")
        }

    def test_selects_five_classes_and_creates_reproducible_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            archive = temp / "plantvillage.zip"
            output_one = temp / "dataset-one"
            output_two = temp / "dataset-two"
            self.make_source_archive(archive)

            self.run_setup(archive, output_one)
            self.run_setup(archive, output_two)

            self.assertEqual(self.image_manifest(output_one), self.image_manifest(output_two))
            self.assertEqual(
                set(json.loads((output_one / "metadata.json").read_text())["classes"]),
                set(CLASSES),
            )

            for split, expected_per_class in (
                ("train", 14),
                ("validation", 3),
                ("test", 3),
            ):
                split_dir = output_one / split
                self.assertEqual(
                    {path.name for path in split_dir.iterdir()}, set(CLASSES)
                )
                for class_name in CLASSES:
                    self.assertEqual(
                        len(list((split_dir / class_name).glob("*.jpg"))),
                        expected_per_class,
                    )

            all_images = self.image_manifest(output_one)
            self.assertEqual(len(all_images), 100)
            self.assertFalse(any("Bacterial_spot" in name for name in all_images))
            self.assertFalse(any("augmented" in name for name in all_images))

    def test_force_replaces_existing_output_after_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            archive = temp / "plantvillage.zip"
            output = temp / "dataset"
            self.make_source_archive(archive)
            output.mkdir()
            (output / "stale.txt").write_text("old output")

            self.run_setup(archive, output, "--force")

            self.assertFalse((output / "stale.txt").exists())
            self.assertEqual(len(self.image_manifest(output)), 100)


if __name__ == "__main__":
    unittest.main()
