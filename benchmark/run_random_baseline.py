from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def run_random_baseline(
    replicate_dir: str | Path,
    seed: int | None = None,
) -> pd.DataFrame:
    replicate_dir = Path(replicate_dir)
    if seed is None:
        config = json.loads((replicate_dir / "config.json").read_text(encoding="utf-8"))
        replicate_index = int(replicate_dir.name.rsplit("_", 1)[-1])
        seed = int(config["seed"]) + 100_000 + replicate_index

    universe = pd.read_csv(
        replicate_dir / "intermediate/evaluation/edge_universe.tsv",
        sep="\t",
    )
    rng = np.random.default_rng(seed)
    result = universe[["cell_type", "regulator", "target"]].copy()
    result.insert(0, "method", "Random")
    result["score"] = rng.random(len(result))
    result["sign_optional"] = rng.choice([-1, 1], size=len(result))

    output = replicate_dir / "method_outputs/standardized/Random.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, sep="\t", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a seeded random GRN baseline.")
    parser.add_argument("replicate_dir", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    run_random_baseline(args.replicate_dir, args.seed)


if __name__ == "__main__":
    main()
