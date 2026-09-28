"""
tests/test_project_setup.py

Checks the project is set up correctly, before any AWS resource exists:
  - every folder is present
  - .env exists and Git ignores it
  - every package is installed

Run:  pytest tests/test_project_setup.py -v
"""

import importlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FOLDERS = ["infra/aws", "config", "adapters", "dataplane/schema", "dataplane/generator",
           "dataplane/load", "metadata", "agent/steps", "guardrails", "api/static",
           "evals/answer_key", "scripts", "tests", "docs/walkthrough"]

PACKAGES = ["boto3", "requests", "langgraph", "fastapi", "sqlglot", "psycopg", "polars", "numpy"]


@pytest.mark.parametrize("folder", FOLDERS)
def test_folder_exists(folder):
    assert (ROOT / folder).is_dir()


def test_env_file_exists():
    assert (ROOT / ".env").exists(), "copy .env.example to .env"


def test_env_file_is_ignored_by_git():
    # git check-ignore prints the path when the file is ignored.
    result = subprocess.run(["git", "check-ignore", ".env"], cwd=ROOT, capture_output=True, text=True)
    assert result.stdout.strip() == ".env", ".env is NOT ignored: your keys could be committed"


@pytest.mark.parametrize("package", PACKAGES)
def test_package_installed(package):
    importlib.import_module(package)
