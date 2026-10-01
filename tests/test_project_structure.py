"""Tests for the project structure and documentation.

These tests only use the Python standard library, so they run anywhere:

    py -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402


class ProjectStructureTests(unittest.TestCase):
    """Every file/folder required by the project must exist."""

    REQUIRED_FILES = [
        "app.py",
        "README.md",
        "requirements.txt",
        "scripts/prepare_dataset.py",
        "src/__init__.py",
        "src/config.py",
        "src/data.py",
        "src/train_cnn.py",
        "src/train_mobilenet.py",
        "src/evaluate.py",
        "src/inference.py",
        "models/.gitkeep",
        "reports/.gitkeep",
    ]

    REQUIRED_DIRECTORIES = [
        "src",
        "models",
        "notebooks",
        "reports",
        "scripts",
        "tests",
    ]

    def test_required_files_exist(self) -> None:
        for relative_path in self.REQUIRED_FILES:
            path = PROJECT_ROOT / relative_path
            self.assertTrue(path.is_file(), f"Missing required file: {relative_path}")

    def test_required_directories_exist(self) -> None:
        for relative_path in self.REQUIRED_DIRECTORIES:
            path = PROJECT_ROOT / relative_path
            self.assertTrue(
                path.is_dir(), f"Missing required directory: {relative_path}"
            )

    def test_notebooks_directory_has_content(self) -> None:
        notebooks = list((PROJECT_ROOT / "notebooks").glob("*.ipynb"))
        self.assertTrue(
            notebooks, "The notebooks/ folder should contain at least one .ipynb file."
        )

    def test_dataset_images_are_not_committed(self) -> None:
        """The .gitignore must keep dataset/, models/* and reports/* out of Git."""
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/dataset/", gitignore)
        self.assertIn("*.keras", gitignore)

    def test_existing_dataset_script_is_untouched_and_importable(self) -> None:
        """The original preparation system must stay in place and working."""
        script = PROJECT_ROOT / "scripts" / "prepare_dataset.py"
        self.assertTrue(script.is_file())
        from scripts.prepare_dataset import CLASSES as PREPARE_CLASSES

        self.assertEqual(list(PREPARE_CLASSES), config.CLASSES)


class RequirementsTests(unittest.TestCase):
    def test_requirements_contains_key_packages(self) -> None:
        text = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        for package in ("tensorflow", "pillow", "numpy", "scikit-learn", "matplotlib", "streamlit"):
            self.assertIn(package, text, f"requirements.txt is missing '{package}'")


class ReadmeTests(unittest.TestCase):
    def test_readme_documents_main_commands(self) -> None:
        text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        expected_snippets = [
            "pip install -r requirements.txt",
            "prepare_dataset.py",
            "train_cnn.py",
            "train_mobilenet.py",
            "evaluate.py",
            "inference.py",
            "streamlit run app.py",
            "unittest discover",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, text, f"README.md does not mention: {snippet}")


if __name__ == "__main__":
    unittest.main()
