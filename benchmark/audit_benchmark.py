from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import REQUIRED_OUTPUT_COLUMNS, validate_standardized_output
from .input_contract import verify_input_contract
from .run_benchmark import METHODS


def audit_replicates(replicate_dirs: list[str | Path]) -> pd.DataFrame:
    rows = []
    expected_methods = set(METHODS)
    for replicate_dir in map(Path, replicate_dirs):
        verify_input_contract(replicate_dir)
        universe = pd.read_csv(
            replicate_dir / "intermediate/evaluation/edge_universe.tsv",
            sep="\t",
        )
        if (universe["regulator"] == universe["target"]).any():
            raise ValueError(f"{replicate_dir} contains self-pairs in its universe")

        leaves = set(
            (replicate_dir / "truth/leaf_nodes.txt").read_text(encoding="utf-8").split()
        )
        status = json.loads(
            (replicate_dir / "evaluation/method_status.json").read_text(
                encoding="utf-8"
            )
        )
        if set(status) != expected_methods:
            raise ValueError(f"{replicate_dir} has an incomplete method status")

        for method in METHODS:
            if status[method]["status"] != "completed":
                raise ValueError(f"{replicate_dir}: {method} did not complete")
            path = replicate_dir / "method_outputs/standardized" / f"{method}.tsv"
            scores = pd.read_csv(path, sep="\t")
            validate_standardized_output(scores)
            if not REQUIRED_OUTPUT_COLUMNS.issubset(scores.columns):
                raise ValueError(f"{path} has an invalid schema")
            if set(scores["cell_type"]) != leaves:
                raise ValueError(f"{path} does not cover every leaf")
            if not np.isfinite(scores["score"]).all():
                raise ValueError(f"{path} contains non-finite scores")
            rows.append(
                {
                    "replicate": replicate_dir.name,
                    "method": method,
                    "status": "valid",
                    "reported_edges": len(scores),
                    "leaf_count": scores["cell_type"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit completed benchmark replicates."
    )
    parser.add_argument("replicate_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_replicates(args.replicate_dirs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, sep="\t", index=False)
    print(
        f"Validated {result['replicate'].nunique()} replicates, "
        f"{result['method'].nunique()} methods, and {len(result)} runs."
    )


if __name__ == "__main__":
    main()
