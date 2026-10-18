from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from .prepare_dictys import collect_dictys, prepare_dictys

OUTPUT_FILES = [
    "weight.tsv",
    "meanvar.tsv",
    "covfactor.tsv",
    "loss.tsv",
    "stats.tsv",
]


def _lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_dictys(
    replicate_dir: str | Path,
    image: str = "lfwa/dictys-cpu:latest",
    nstep: int = 1000,
    seed: int = 20250612,
) -> None:
    replicate_dir = Path(replicate_dir).resolve()
    prepare_dictys(replicate_dir)
    prepared = replicate_dir / "method_inputs" / "Dictys" / "prepared"

    for index, leaf in enumerate(_lines(replicate_dir / "truth" / "leaf_nodes.txt")):
        leaf_dir = prepared / leaf
        expression = pd.read_csv(
            leaf_dir / "expression.tsv",
            sep="\t",
            index_col=0,
        )
        npc = max(1, min(50, expression.shape[0] - 1, expression.shape[1] - 1))
        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{leaf_dir}:/work",
            image,
            "run_dictys",
            "network",
            "reconstruct",
            "/work/expression.tsv",
            "/work/mask.tsv",
            *[f"/work/{name}" for name in OUTPUT_FILES],
            "--nstep",
            str(nstep),
            "--npc",
            str(npc),
            "--device",
            "cpu",
            "--nth",
            "1",
            "--rseed",
            str(seed + index),
        ]
        subprocess.run(command, check=True)

    collect_dictys(replicate_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official Dictys reconstruction.")
    parser.add_argument("replicate_dir", type=Path)
    parser.add_argument("--image", default="lfwa/dictys-cpu:latest")
    parser.add_argument("--nstep", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20250612)
    args = parser.parse_args()
    run_dictys(args.replicate_dir, args.image, args.nstep, args.seed)


if __name__ == "__main__":
    main()
