from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .evaluation import evaluate_scores
from .standardize_regarch import standardize_regarch


def evaluate_ancestor_recovery(
    replicate_dirs: list[str | Path],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for replicate_dir in map(Path, replicate_dirs):
        scores = standardize_regarch(
            replicate_dir / "method_outputs/raw/RegArch_full/inferred_grns",
            replicate_dir / "method_inputs/RegArch/regulator_list.txt",
            replicate_dir / "evaluation/ancestor_recovery/RegArch_all_nodes.tsv",
            method_name="RegArch",
        )
        universe = pd.read_csv(
            replicate_dir / "intermediate/evaluation/edge_universe_ancestor_nodes.tsv",
            sep="\t",
        )
        ancestors = set(universe["cell_type"])
        metrics = evaluate_scores(
            universe,
            scores.loc[scores["cell_type"].isin(ancestors)],
        )
        metrics.insert(0, "replicate", replicate_dir.name)
        parts.append(metrics)

    metrics = pd.concat(parts, ignore_index=True)
    metrics.to_csv(output_dir / "ancestor_metrics.tsv", sep="\t", index=False)
    node_metrics = metrics.loc[metrics["cell_type"] != "__mean__"]
    summary = (
        node_metrics.groupby("cell_type")
        .agg(
            auprc_mean=("auprc", "mean"),
            auprc_sd=("auprc", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            precision_at_true_k_mean=("precision_at_true_k", "mean"),
            recall_at_true_k_mean=("recall_at_true_k", "mean"),
            top10_regulator_overlap_mean=("top10_regulator_overlap", "mean"),
            replicate_count=("replicate", "nunique"),
        )
        .reset_index()
    )
    overall = pd.DataFrame(
        [
            {
                "cell_type": "__all_ancestors__",
                "auprc_mean": node_metrics["auprc"].mean(),
                "auprc_sd": node_metrics["auprc"].std(),
                "auroc_mean": node_metrics["auroc"].mean(),
                "auroc_sd": node_metrics["auroc"].std(),
                "precision_at_true_k_mean": node_metrics["precision_at_true_k"].mean(),
                "recall_at_true_k_mean": node_metrics["recall_at_true_k"].mean(),
                "top10_regulator_overlap_mean": node_metrics[
                    "top10_regulator_overlap"
                ].mean(),
                "replicate_count": node_metrics["replicate"].nunique(),
            }
        ]
    )
    summary = pd.concat([summary, overall], ignore_index=True)
    summary.to_csv(
        output_dir / "ancestor_summary.tsv",
        sep="\t",
        index=False,
    )
    return metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RegArch ancestor-node recovery."
    )
    parser.add_argument("replicate_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _, summary = evaluate_ancestor_recovery(
        args.replicate_dirs,
        args.output_dir,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
