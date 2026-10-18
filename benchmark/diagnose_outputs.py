from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def diagnose_replicates(replicate_dirs: list[str | Path]) -> pd.DataFrame:
    rows = []
    for replicate_dir in map(Path, replicate_dirs):
        universe = pd.read_csv(
            replicate_dir / "intermediate/evaluation/edge_universe.tsv",
            sep="\t",
        )
        for path in sorted(
            (replicate_dir / "method_outputs/standardized").glob("*.tsv")
        ):
            scores = pd.read_csv(path, sep="\t")
            for leaf, truth in universe.groupby("cell_type"):
                leaf_scores = scores.loc[scores["cell_type"] == leaf]
                merged = truth.merge(
                    leaf_scores[["regulator", "target", "score"]],
                    on=["regulator", "target"],
                    how="left",
                )
                present = merged["score"].fillna(0).abs().gt(0)
                true_edges = merged["true_edge"].eq(1)
                rows.append(
                    {
                        "replicate": replicate_dir.name,
                        "method": path.stem,
                        "cell_type": leaf,
                        "reported_edge_count": len(leaf_scores),
                        "nonzero_edge_count": int(present.sum()),
                        "edge_universe_count": len(merged),
                        "nonzero_universe_fraction": float(present.mean()),
                        "true_edges_with_nonzero_score": int(
                            (present & true_edges).sum()
                        ),
                        "true_edge_count": int(true_edges.sum()),
                        "nonzero_recall_ceiling": float(
                            (present & true_edges).sum() / true_edges.sum()
                        )
                        if true_edges.any()
                        else np.nan,
                        "unique_nonzero_scores": int(
                            merged.loc[present, "score"].nunique()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose benchmark output coverage.")
    parser.add_argument("replicate_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnostics = diagnose_replicates(args.replicate_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
