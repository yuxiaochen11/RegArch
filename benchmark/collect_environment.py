from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

PYTHON_PACKAGES = [
    "regarch",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "arboreto",
    "celloracle",
    "scenicplus",
]


def _command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"
    return (result.stdout or result.stderr).strip()


def collect_environment() -> pd.DataFrame:
    rows = [
        {"component": "OS", "version": platform.platform()},
        {"component": "Python", "version": sys.version.split()[0]},
        {"component": "R", "version": _command(["Rscript", "--version"])},
        {"component": "Docker", "version": _command(["docker", "--version"])},
    ]
    for package in PYTHON_PACKAGES:
        if package == "regarch":
            version = "repository source"
        else:
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = ""
            if not version:
                try:
                    version = str(importlib.import_module(package).__version__)
                except (ImportError, AttributeError):
                    version = "not installed in host environment"
        rows.append({"component": f"python:{package}", "version": version})

    images = _command(
        [
            "docker",
            "images",
            "--format",
            "{{.Repository}}:{{.Tag}}",
        ]
    )
    for image in images.splitlines():
        if image.startswith(("regarch-", "lfwa/", "kenjikamimoto126/")):
            rows.append({"component": "docker_image", "version": image})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    collect_environment().to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
