from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .ablation import save_wrong_fate_map
from .evaluation import evaluate_scores
from .fate_map import FateMapTruth
from .standardize_regarch import standardize_regarch

QUALITY_LEVELS = [
    ("prior_cov020_cor100", 0.20, 1.00),
    ("prior_cov040_cor100", 0.40, 1.00),
    ("prior_cov060_cor100", 0.60, 1.00),
    ("prior_cov080_cor100", 0.80, 1.00),
    ("prior_cov100_cor020", 1.00, 0.20),
    ("prior_cov100_cor040", 1.00, 0.40),
    ("prior_cov100_cor060", 1.00, 0.60),
    ("prior_cov100_cor080", 1.00, 0.80),
]


def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _load_truth_fate_map(replicate_dir: Path) -> FateMapTruth:
    truth_dir = replicate_dir / "truth"
    edges = pd.read_csv(truth_dir / "fate_map_edges.tsv", sep="\t")
    nodes = pd.read_csv(truth_dir / "node_metadata.tsv", sep="\t")
    newick_path = truth_dir / "fate_map.nwk"
    newick = newick_path.read_text(encoding="utf-8").strip()
    return FateMapTruth(edges=edges, nodes=nodes, newick=newick)


def _quality_prior(
    replicate_dir: Path,
    coverage: float,
    correctness: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    base = pd.read_csv(
        replicate_dir / "method_inputs/RegArch/atac_prior_tf_target.tsv",
        sep="\t",
    )
    base = base.loc[base["node"].str.startswith("T")].copy()
    truth = pd.read_csv(
        replicate_dir / "truth/true_grn_all_nodes.tsv",
        sep="\t",
    )
    merged = base.merge(
        truth[["node", "regulator", "target", "true_edge"]],
        on=["node", "regulator", "target"],
        validate="one_to_one",
    )
    merged["prior_score"] = 0.0
    rng = np.random.default_rng(seed)

    for _, group in merged.groupby("node"):
        true_indices = group.index[group["true_edge"].eq(1)].to_numpy()
        false_indices = group.index[group["true_edge"].eq(0)].to_numpy()
        true_count = int(round(coverage * len(true_indices)))
        selected_true = (
            rng.choice(true_indices, size=true_count, replace=False)
            if true_count
            else np.array([], dtype=int)
        )
        total_nonzero = int(round(true_count / correctness)) if true_count else 0
        false_count = min(max(0, total_nonzero - true_count), len(false_indices))
        selected_false = (
            rng.choice(false_indices, size=false_count, replace=False)
            if false_count
            else np.array([], dtype=int)
        )

        observed_scores = base.loc[
            base["node"].eq(group.iloc[0]["node"]) & base["prior_score"].gt(0),
            "prior_score",
        ].to_numpy()
        if not len(observed_scores):
            observed_scores = np.array([0.5])
        selected = np.concatenate([selected_true, selected_false])
        if len(selected):
            merged.loc[selected, "prior_score"] = rng.choice(
                observed_scores,
                size=len(selected),
                replace=True,
            )

    nonzero = merged["prior_score"].gt(0)
    true = merged["true_edge"].eq(1)
    statistics = {
        "actual_coverage": float((nonzero & true).sum() / true.sum()),
        "actual_correctness": float(
            merged.loc[nonzero, "true_edge"].mean() if nonzero.any() else np.nan
        ),
        "nonzero_prior_edges": int(nonzero.sum()),
    }
    return merged.drop(columns="true_edge"), statistics


def _condition_specs(replicate_dir: Path) -> list[dict]:
    specs = [
        {
            "condition": "full",
            "factor": "reference",
            "variant": "full",
            "run_name": "full",
            "root_prior": "atac",
            "prior_file": None,
        },
        {
            "condition": "no_fate_map",
            "factor": "fate_map",
            "variant": "no_fate_map",
            "run_name": "ablation_no_fate_map",
            "root_prior": "atac",
            "prior_file": None,
        },
        {
            "condition": "wrong_fate_map",
            "factor": "fate_map",
            "variant": "wrong_fate_map",
            "run_name": "ablation_wrong_fate_map",
            "root_prior": "atac",
            "prior_file": None,
        },
        {
            "condition": "no_atac",
            "factor": "atac",
            "variant": "no_atac",
            "run_name": "ablation_no_atac",
            "root_prior": "uniform",
            "prior_file": None,
        },
        {
            "condition": "uniform_root",
            "factor": "atac",
            "variant": "full",
            "run_name": "ablation_uniform_root",
            "root_prior": "uniform",
            "prior_file": None,
        },
    ]
    for condition, coverage, correctness in QUALITY_LEVELS:
        specs.append(
            {
                "condition": condition,
                "factor": "atac_quality",
                "variant": "full",
                "run_name": f"ablation_{condition}",
                "root_prior": "atac",
                "prior_file": (
                    replicate_dir / "ablations/controlled_atac" / f"{condition}.tsv"
                ),
                "coverage": coverage,
                "correctness": correctness,
            }
        )
    return specs


def _standardize_and_evaluate(
    replicate_dir: Path,
    spec: dict,
) -> pd.DataFrame:
    run_dir = replicate_dir / "method_outputs/raw" / f"RegArch_{spec['run_name']}"
    if spec["condition"] == "full":
        run_dir = replicate_dir / "method_outputs/raw/RegArch_full"
    scores = standardize_regarch(
        run_dir / "inferred_grns",
        replicate_dir / "method_inputs/RegArch/regulator_list.txt",
        replicate_dir / "evaluation/ablations" / f"{spec['condition']}_all_nodes.tsv",
        method_name=spec["condition"],
    )
    universe = pd.read_csv(
        replicate_dir / "intermediate/evaluation/edge_universe_all_nodes.tsv",
        sep="\t",
    )
    if spec["condition"] == "no_fate_map":
        leaves = set(
            pd.read_csv(replicate_dir / "truth/node_metadata.tsv", sep="\t").loc[
                lambda x: x["node_type"].eq("leaf"), "node_id"
            ]
        )
        universe = universe.loc[universe["cell_type"].isin(leaves)]
    elif spec["condition"] == "wrong_fate_map":
        # Leaves retain their identities. Internal wrong-map clades cross the
        # two true clades, so only the shared Root is uniquely comparable.
        keep = {"Root"} | set(
            pd.read_csv(replicate_dir / "truth/node_metadata.tsv", sep="\t").loc[
                lambda x: x["node_type"].eq("leaf"), "node_id"
            ]
        )
        universe = universe.loc[universe["cell_type"].isin(keep)]
        scores = scores.loc[scores["cell_type"].isin(keep)]
    metrics = evaluate_scores(universe, scores)
    return metrics.loc[metrics["cell_type"].ne("__mean__")].copy()


def _paired_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for node_group in ["leaf", "ancestor", "root"]:
        if node_group == "leaf":
            subset = metrics.loc[metrics["node_type"].eq("leaf")]
        elif node_group == "ancestor":
            subset = metrics.loc[metrics["node_type"].isin(["root", "internal"])]
        else:
            subset = metrics.loc[metrics["node_type"].eq("root")]
        for condition in sorted(set(subset["condition"]) - {"full"}):
            for metric in [
                "auprc",
                "auroc",
                "precision_at_true_k",
                "signed_precision_at_true_k",
            ]:
                paired = (
                    subset.loc[
                        subset["condition"].isin(["full", condition]),
                        ["replicate", "cell_type", "condition", metric],
                    ]
                    .pivot(
                        index=["replicate", "cell_type"],
                        columns="condition",
                        values=metric,
                    )
                    .dropna()
                )
                if "full" not in paired or condition not in paired:
                    continue
                # Nodes within one simulated fate map are correlated. Pair
                # matched cell types first, then average within replicate so
                # each replicate contributes one independent paired value.
                paired = paired.groupby(level="replicate").mean()
                differences = paired["full"] - paired[condition]
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
                        "node_group": node_group,
                        "condition": condition,
                        "metric": metric,
                        "independent_replicates": len(differences),
                        "full_mean": paired["full"].mean(),
                        "condition_mean": paired[condition].mean(),
                        "full_minus_condition": differences.mean(),
                        "full_win_rate": (differences > 0).mean(),
                        "wilcoxon_one_sided_p": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "node_group",
                "condition",
                "metric",
                "independent_replicates",
                "full_mean",
                "condition_mean",
                "full_minus_condition",
                "full_win_rate",
                "wilcoxon_one_sided_p",
                "holm_adjusted_p",
            ]
        )
    result["holm_adjusted_p"] = np.nan
    for _, indices in result.groupby(["node_group", "metric"]).groups.items():
        ordered = result.loc[indices, "wilcoxon_one_sided_p"].sort_values()
        adjusted = np.maximum.accumulate(
            [
                min(1.0, value * (len(ordered) - rank))
                for rank, value in enumerate(ordered)
            ]
        )
        result.loc[ordered.index, "holm_adjusted_p"] = adjusted
    return result


def _plot_results(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        "full",
        "no_fate_map",
        "wrong_fate_map",
        "uniform_root",
        "no_atac",
    ]
    plot = summary.loc[
        summary["condition"].isin(selected)
        & summary["node_group"].isin(["leaf", "ancestor"])
    ].copy()
    pivot = plot.pivot(
        index="condition", columns="node_group", values="auprc_mean"
    ).reindex(selected)
    ax = pivot.plot(kind="bar", figsize=(9, 4.8), color=["#4C78A8", "#F58518"])
    ax.set_ylabel("AUPRC")
    ax.set_xlabel("")
    ax.set_ylim(0, 0.9)
    ax.legend(title="Node group")
    ax.tick_params(axis="x", rotation=25)
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "fate_atac_ablation.png", dpi=220)
    plt.close(ax.figure)

    quality = summary.loc[
        summary["condition"].str.startswith("prior_cov")
        & summary["node_group"].isin(["leaf", "ancestor"])
    ].copy()
    parameters = (
        quality["condition"]
        .str.extract(r"prior_cov(?P<coverage>\d{3})_cor(?P<correctness>\d{3})")
        .astype(float)
        / 100
    )
    quality = pd.concat([quality.reset_index(drop=True), parameters], axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for node_group, color in [("leaf", "#4C78A8"), ("ancestor", "#F58518")]:
        coverage = quality.loc[
            quality["node_group"].eq(node_group) & quality["correctness"].eq(1.0)
        ].sort_values("coverage")
        axes[0].plot(
            coverage["coverage"],
            coverage["auprc_mean"],
            marker="o",
            label=node_group,
            color=color,
        )
        correctness = quality.loc[
            quality["node_group"].eq(node_group) & quality["coverage"].eq(1.0)
        ].sort_values("correctness")
        axes[1].plot(
            correctness["correctness"],
            correctness["auprc_mean"],
            marker="o",
            label=node_group,
            color=color,
        )
    axes[0].set_xlabel("Prior coverage")
    axes[1].set_xlabel("Prior correctness")
    axes[0].set_ylabel("AUPRC")
    axes[0].set_title("Correctness fixed at 1.0")
    axes[1].set_title("Coverage fixed at 1.0")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "atac_prior_quality_ablation.png", dpi=220)
    plt.close(fig)


def _write_report(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> None:
    leaf = summary.loc[summary["node_group"].eq("leaf")].set_index("condition")
    ancestor = summary.loc[summary["node_group"].eq("ancestor")].set_index("condition")
    lines = [
        "# RegArch ablation study",
        "",
        "## Design",
        "",
        "- Ten independently generated final-benchmark replicates were reused.",
        "- RNA, truth GRNs, candidate edges, optimization settings, and evaluation metrics were fixed.",
        "- Statistical tests average correlated nodes within each replicate before paired testing.",
        "- `no_fate_map` is evaluated only at leaves because it has no identifiable ancestor output.",
        "- Wrong-topology internal clades cross the true clades; only leaves and the shared Root are evaluated.",
        "- Controlled ATAC priors use truth only to create known-quality perturbations before inference.",
        "  The 100%/100% prior is an oracle upper bound, not a realistic benchmark input.",
        "",
        "## Primary AUPRC",
        "",
        "| Condition | Leaf | Ancestor |",
        "|---|---:|---:|",
    ]
    preferred_order = [
        "full",
        "no_fate_map",
        "wrong_fate_map",
        "uniform_root",
        "no_atac",
        "prior_cov100_cor100",
        "prior_cov075_cor100",
        "prior_cov050_cor100",
        "prior_cov025_cor100",
        "prior_cov100_cor090",
        "prior_cov100_cor075",
        "prior_cov100_cor050",
    ]
    available = set(leaf.index) | set(ancestor.index)
    conditions = [condition for condition in preferred_order if condition in available]
    conditions.extend(sorted(available - set(conditions)))
    for condition in conditions:
        leaf_value = (
            f"{leaf.loc[condition, 'auprc_mean']:.4f}"
            if condition in leaf.index
            else "NA"
        )
        ancestor_value = (
            f"{ancestor.loc[condition, 'auprc_mean']:.4f}"
            if condition in ancestor.index
            else "NA"
        )
        lines.append(f"| {condition} | {leaf_value} | {ancestor_value} |")
    lines.extend(
        [
            "",
            "## Interpretation safeguards",
            "",
            "- High sign accuracy in a weak condition is conditional on the small subset of true edges recovered;",
            "  signed precision at K should be used to assess joint edge-and-sign recovery.",
            "- AUPRC and AUROC assess the complete ranking. Precision at one cutoff can occasionally improve",
            "  even when the global ranking deteriorates.",
            "- Holm-adjusted paired tests are available in `paired_comparisons.tsv`.",
            "",
            "All target-gene fits and convergence counts are recorded in `optimization_convergence.tsv`.",
        ]
    )
    (output_dir / "ABLATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_study(
    replicate_dirs: list[Path],
    output_dir: Path,
    processes: int,
    optimizer_maxfun: int,
    force: bool,
    conditions: set[str] | None = None,
) -> None:
    repo = Path(__file__).resolve().parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_parts = []
    parameter_rows = []
    prior_rows = []
    convergence_rows = []

    for replicate_index, replicate_dir in enumerate(replicate_dirs, start=1):
        config = json.loads((replicate_dir / "config.json").read_text())
        seed = int(config["seed"]) + replicate_index + 700_000
        nodes = pd.read_csv(replicate_dir / "truth/node_metadata.tsv", sep="\t").rename(
            columns={"node_id": "cell_type"}
        )
        specs = _condition_specs(replicate_dir)
        if conditions is not None:
            specs = [spec for spec in specs if spec["condition"] in conditions]
        for spec in specs:
            if spec["condition"] == "wrong_fate_map":
                save_wrong_fate_map(
                    _load_truth_fate_map(replicate_dir),
                    replicate_dir / "ablations/wrong_fate_map",
                    mode="cross_clade_short_branches",
                )

            prior_file = spec.get("prior_file")
            if spec["condition"] in {"full", "uniform_root"}:
                base = pd.read_csv(
                    replicate_dir / "method_inputs/RegArch/atac_prior_tf_target.tsv",
                    sep="\t",
                )
                base = base.loc[base["node"].str.startswith("T")]
                truth = pd.read_csv(
                    replicate_dir / "truth/true_grn_all_nodes.tsv", sep="\t"
                )
                observed = base.merge(
                    truth[["node", "regulator", "target", "true_edge"]],
                    on=["node", "regulator", "target"],
                    validate="one_to_one",
                )
                nonzero = observed["prior_score"].gt(0)
                true = observed["true_edge"].eq(1)
                prior_rows.append(
                    {
                        "replicate": replicate_dir.name,
                        "condition": spec["condition"],
                        "requested_coverage": np.nan,
                        "requested_correctness": np.nan,
                        "actual_coverage": float((nonzero & true).sum() / true.sum()),
                        "actual_correctness": float(
                            observed.loc[nonzero, "true_edge"].mean()
                        ),
                        "nonzero_prior_edges": int(nonzero.sum()),
                    }
                )
            elif spec["condition"] == "no_atac":
                prior_rows.append(
                    {
                        "replicate": replicate_dir.name,
                        "condition": spec["condition"],
                        "requested_coverage": 0.0,
                        "requested_correctness": np.nan,
                        "actual_coverage": 0.0,
                        "actual_correctness": np.nan,
                        "nonzero_prior_edges": 0,
                    }
                )
            if prior_file is not None:
                prior_file.parent.mkdir(parents=True, exist_ok=True)
                prior, statistics = _quality_prior(
                    replicate_dir,
                    spec["coverage"],
                    spec["correctness"],
                    seed + int(spec["coverage"] * 1000),
                )
                prior.to_csv(prior_file, sep="\t", index=False)
                prior_rows.append(
                    {
                        "replicate": replicate_dir.name,
                        "condition": spec["condition"],
                        "requested_coverage": spec["coverage"],
                        "requested_correctness": spec["correctness"],
                        **statistics,
                    }
                )

            run_name = spec["run_name"]
            run_dir = replicate_dir / "method_outputs/raw" / f"RegArch_{run_name}"
            if spec["condition"] != "full" and force and run_dir.exists():
                shutil.rmtree(run_dir)
            inferred_files = list((run_dir / "inferred_grns").glob("target_gene_*.csv"))
            diagnostic_files = list(
                (run_dir / "optimization_diagnostics").glob("*.json")
            )
            successful_diagnostics = (
                sum(
                    bool(json.loads(path.read_text())["success"])
                    for path in diagnostic_files
                )
                if diagnostic_files
                else 0
            )
            complete_run = (
                len(inferred_files) == int(config["grn"]["target_count"])
                and len(diagnostic_files) == int(config["grn"]["target_count"])
                and successful_diagnostics == int(config["grn"]["target_count"])
            )
            if spec["condition"] != "full" and (force or not complete_run):
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                command = [
                    sys.executable,
                    "-m",
                    "benchmark.run_regarch",
                    str(replicate_dir),
                    "--variant",
                    spec["variant"],
                    "--run-name",
                    run_name,
                    "--processes",
                    str(processes),
                    "--lineage-model",
                    "ctmc_signed",
                    "--lineage-weight",
                    "0.5",
                    "--atac-weight",
                    "1.0",
                    "--likelihood-weight",
                    "1.0",
                    "--root-prior",
                    spec["root_prior"],
                    "--optimizer-maxfun",
                    str(optimizer_maxfun),
                ]
                if prior_file is not None:
                    command.extend(["--prior-file", str(prior_file)])
                _run(command, repo)

            metrics = _standardize_and_evaluate(replicate_dir, spec)
            metrics.insert(0, "replicate", replicate_dir.name)
            metrics.insert(1, "condition", spec["condition"])
            metrics = metrics.merge(
                nodes[["cell_type", "node_type", "developmental_time"]],
                on="cell_type",
                validate="many_to_one",
            )
            metric_parts.append(metrics)

            actual_run_dir = (
                replicate_dir / "method_outputs/raw/RegArch_full"
                if spec["condition"] == "full"
                else run_dir
            )
            diagnostics = list(
                (actual_run_dir / "optimization_diagnostics").glob("*.json")
            )
            convergence_rows.append(
                {
                    "replicate": replicate_dir.name,
                    "condition": spec["condition"],
                    "target_fits": len(diagnostics),
                    "successful_fits": sum(
                        bool(json.loads(path.read_text())["success"])
                        for path in diagnostics
                    ),
                }
            )
            parameter_rows.append(
                {
                    "condition": spec["condition"],
                    "factor": spec["factor"],
                    "fate_map": (
                        "none"
                        if spec["condition"] == "no_fate_map"
                        else "wrong topology + short branches"
                        if spec["condition"] == "wrong_fate_map"
                        else "correct"
                    ),
                    "lineage_weight": (
                        0.0 if spec["condition"] == "no_fate_map" else 0.5
                    ),
                    "atac_weight": (0.0 if spec["condition"] == "no_atac" else 1.0),
                    "root_prior": spec["root_prior"],
                    "requested_coverage": spec.get("coverage", np.nan),
                    "requested_correctness": spec.get("correctness", np.nan),
                    "RNA_data": "unchanged",
                    "truth_GRN": "unchanged",
                }
            )

    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics.to_csv(output_dir / "ablation_metrics.tsv", sep="\t", index=False)
    summary = metrics.groupby(["condition", "node_type"], as_index=False).agg(
        nodes_evaluated=("cell_type", "size"),
        auprc_mean=("auprc", "mean"),
        auprc_sd=("auprc", "std"),
        auroc_mean=("auroc", "mean"),
        precision_at_true_k_mean=("precision_at_true_k", "mean"),
        signed_precision_at_true_k_mean=("signed_precision_at_true_k", "mean"),
        sign_accuracy_mean=("sign_accuracy_on_recovered_true_edges", "mean"),
    )
    summary.to_csv(output_dir / "ablation_summary.tsv", sep="\t", index=False)
    grouped = metrics.assign(
        node_group=np.where(metrics["node_type"].eq("leaf"), "leaf", "ancestor")
    )
    group_summary = grouped.groupby(["condition", "node_group"], as_index=False).agg(
        nodes_evaluated=("cell_type", "size"),
        auprc_mean=("auprc", "mean"),
        auprc_sd=("auprc", "std"),
        auroc_mean=("auroc", "mean"),
        precision_at_true_k_mean=("precision_at_true_k", "mean"),
        signed_precision_at_true_k_mean=("signed_precision_at_true_k", "mean"),
    )
    group_summary.to_csv(
        output_dir / "ablation_group_summary.tsv", sep="\t", index=False
    )
    comparisons = _paired_comparisons(metrics)
    comparisons.to_csv(output_dir / "paired_comparisons.tsv", sep="\t", index=False)
    pd.DataFrame(parameter_rows).drop_duplicates().to_csv(
        output_dir / "ablation_parameters.tsv", sep="\t", index=False
    )
    pd.DataFrame(prior_rows).to_csv(
        output_dir / "atac_prior_quality.tsv", sep="\t", index=False
    )
    pd.DataFrame(convergence_rows).to_csv(
        output_dir / "optimization_convergence.tsv", sep="\t", index=False
    )
    _plot_results(group_summary, output_dir)
    _write_report(group_summary, comparisons, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the matched RegArch ablation study."
    )
    parser.add_argument("replicate_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--optimizer-maxfun", type=int, default=60000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--conditions",
        nargs="+",
        help="Optional subset of ablation condition names to run.",
    )
    args = parser.parse_args()
    run_study(
        [path.resolve() for path in args.replicate_dirs],
        args.output_dir,
        args.processes,
        args.optimizer_maxfun,
        args.force,
        set(args.conditions) if args.conditions else None,
    )


if __name__ == "__main__":
    main()
