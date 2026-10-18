from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .evaluation import evaluate_scores, load_standardized_outputs

METRICS = [
    "auprc",
    "auroc",
    "precision_at_true_k",
    "recall_at_true_k",
    "signed_precision_at_true_k",
    "signed_recall_at_true_k",
    "sign_accuracy_on_recovered_true_edges",
    "sign_coverage_on_recovered_true_edges",
    "top10_regulator_overlap",
]


def _bootstrap_interval(
    differences: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 5000,
) -> tuple[float, float]:
    if not len(differences):
        return np.nan, np.nan
    means = np.empty(iterations)
    for index in range(iterations):
        means[index] = rng.choice(differences, len(differences), replace=True).mean()
    return tuple(np.quantile(means, [0.025, 0.975]))


def compare_replicates(
    replicate_dirs: list[str | Path],
    reference: str = "RegArch",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_parts = []
    for replicate_dir in map(Path, replicate_dirs):
        universe = pd.read_csv(
            replicate_dir / "intermediate/evaluation/edge_universe.tsv", sep="\t"
        )
        score_paths = sorted(
            (replicate_dir / "method_outputs/standardized").glob("*.tsv")
        )
        if not score_paths:
            raise FileNotFoundError(f"No standardized scores in {replicate_dir}.")
        metrics = evaluate_scores(
            universe,
            load_standardized_outputs(score_paths),
        )
        metrics.insert(0, "replicate", replicate_dir.name)
        metric_parts.append(metrics)
    all_metrics = pd.concat(metric_parts, ignore_index=True)
    leaf_metrics = all_metrics.loc[all_metrics["cell_type"] != "__mean__"].copy()

    rows = []
    rng = np.random.default_rng(20250611)
    for competitor in sorted(set(leaf_metrics["method"]) - {reference}):
        for metric in METRICS:
            leaf_level = (
                leaf_metrics.loc[
                    leaf_metrics["method"].isin([reference, competitor]),
                    ["replicate", "cell_type", "method", metric],
                ]
                .pivot(
                    index=["replicate", "cell_type"],
                    columns="method",
                    values=metric,
                )
                .dropna()
            )
            # Leaves from one simulated fate map share a generated root and
            # are not independent experimental replicates.
            paired = leaf_level.groupby(level="replicate").mean()
            if paired.empty or reference not in paired or competitor not in paired:
                continue
            differences = (paired[reference] - paired[competitor]).to_numpy()
            low, high = _bootstrap_interval(differences, rng)
            try:
                p_value = (
                    wilcoxon(differences, alternative="greater").pvalue
                    if np.any(differences)
                    else 1.0
                )
            except ValueError:
                p_value = np.nan
            rows.append(
                {
                    "reference": reference,
                    "competitor": competitor,
                    "metric": metric,
                    "paired_leaf_count": len(leaf_level),
                    "independent_replicate_count": len(differences),
                    "reference_mean": paired[reference].mean(),
                    "competitor_mean": paired[competitor].mean(),
                    "mean_difference": differences.mean(),
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                    "win_rate": np.mean(differences > 0),
                    "wilcoxon_one_sided_p": p_value,
                }
            )
    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        comparison["wilcoxon_holm_p"] = np.nan
        for metric, indices in comparison.groupby("metric").groups.items():
            ordered = comparison.loc[indices, "wilcoxon_one_sided_p"].sort_values()
            adjusted = pd.Series(index=ordered.index, dtype=float)
            running = 0.0
            count = len(ordered)
            for rank, (index, p_value) in enumerate(ordered.items()):
                value = min(1.0, float(p_value) * (count - rank))
                running = max(running, value)
                adjusted.loc[index] = running
            comparison.loc[adjusted.index, "wilcoxon_holm_p"] = adjusted
        comparison["advantage_supported"] = (
            comparison["mean_difference"].gt(0)
            & comparison["difference_ci_low"].gt(0)
            & comparison["wilcoxon_holm_p"].lt(0.05)
        )
    return all_metrics, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare leaf GRN reconstruction.")
    parser.add_argument("replicate_dirs", type=Path, nargs="+")
    parser.add_argument("--reference", default="RegArch")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-auprc-advantage",
        action="store_true",
        help="Exit non-zero unless held-out AUPRC advantage is supported versus every competitor.",
    )
    args = parser.parse_args()
    metrics, comparison = compare_replicates(args.replicate_dirs, args.reference)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "leaf_metrics.tsv", sep="\t", index=False)
    comparison.to_csv(
        args.output_dir / "paired_method_comparison.tsv", sep="\t", index=False
    )
    if args.require_auprc_advantage:
        auprc = comparison.loc[comparison["metric"] == "auprc"]
        if auprc.empty or not auprc["advantage_supported"].all():
            raise SystemExit(
                "Held-out results do not support an AUPRC advantage against every method."
            )


if __name__ == "__main__":
    main()
