from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from .audit_ancestor_recovery import _descendant_leaves, _score_table
from .evaluation import evaluate_scores
from .standardize_regarch import standardize_regarch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_matched_simulation(
    replicate_dirs: list[Path],
    output_dir: Path,
    run_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_parts = []
    baseline_parts = []
    optimizer_rows = []
    identity_rows = []
    sign_rows = []
    lineage_parts = []

    for replicate_dir in replicate_dirs:
        run_dir = replicate_dir / "method_outputs/raw" / f"RegArch_{run_name}"
        scores = standardize_regarch(
            run_dir / "inferred_grns",
            replicate_dir / "method_inputs/RegArch/regulator_list.txt",
            replicate_dir / "evaluation/matched_validation/RegArch_all_nodes.tsv",
            method_name="RegArch",
        )
        universe = pd.read_csv(
            replicate_dir / "intermediate/evaluation/edge_universe_all_nodes.tsv",
            sep="\t",
        )
        metrics = evaluate_scores(universe, scores)
        metrics = metrics.loc[metrics["cell_type"].ne("__mean__")].copy()
        metrics.insert(0, "replicate", replicate_dir.name)

        nodes = pd.read_csv(
            replicate_dir / "truth/node_metadata.tsv",
            sep="\t",
        ).rename(columns={"node_id": "cell_type"})
        metrics = metrics.merge(
            nodes[["cell_type", "node_type", "developmental_time"]],
            on="cell_type",
            validate="many_to_one",
        )
        metric_parts.append(metrics)

        truth = pd.read_csv(
            replicate_dir / "truth/true_grn_all_nodes.tsv",
            sep="\t",
        ).rename(columns={"node": "cell_type"})
        signed = truth.merge(
            scores[
                [
                    "cell_type",
                    "regulator",
                    "target",
                    "sign_optional",
                ]
            ],
            on=["cell_type", "regulator", "target"],
            validate="one_to_one",
        )
        signed = signed.loc[
            signed["true_edge"].eq(1) & signed["regulator"].ne(signed["target"])
        ]
        for node, group in signed.groupby("cell_type"):
            sign_rows.append(
                {
                    "replicate": replicate_dir.name,
                    "cell_type": node,
                    "true_edge_count": len(group),
                    "sign_accuracy_on_true_edges": float(
                        (group["true_sign"] == group["sign_optional"]).mean()
                    ),
                    "predicted_positive_fraction": float(
                        group["sign_optional"].gt(0).mean()
                    ),
                    "true_positive_fraction": float(group["true_sign"].gt(0).mean()),
                }
            )

        edges = pd.read_csv(
            replicate_dir / "truth/fate_map_edges.tsv",
            sep="\t",
        )
        ancestors = nodes.loc[
            nodes["node_type"].isin(["root", "internal"]),
            "cell_type",
        ].tolist()
        leaves = nodes.loc[nodes["node_type"].eq("leaf"), "cell_type"].tolist()
        leaf_scores = scores.loc[scores["cell_type"].isin(leaves)]
        baselines = []
        for ancestor in ancestors:
            descendants = _descendant_leaves(edges, ancestor)
            propagated = (
                leaf_scores.loc[leaf_scores["cell_type"].isin(descendants)]
                .groupby(["regulator", "target"], as_index=False)["score"]
                .mean()
            )
            baselines.append(
                _score_table(
                    "descendant_leaf_mean",
                    ancestor,
                    propagated,
                )
            )
            oracle = (
                truth.loc[truth["cell_type"].isin(descendants)]
                .groupby(["regulator", "target"], as_index=False)["true_edge"]
                .mean()
                .rename(columns={"true_edge": "score"})
            )
            baselines.append(
                _score_table(
                    "oracle_descendant_truth_mean",
                    ancestor,
                    oracle,
                )
            )

        rng = np.random.default_rng(
            int(json.loads((replicate_dir / "config.json").read_text())["seed"])
            + int(replicate_dir.name.rsplit("_", 1)[-1])
            + 900_000
        )
        random_scores = universe[["cell_type", "regulator", "target"]].copy()
        random_scores.insert(0, "method", "Random")
        random_scores["score"] = rng.random(len(random_scores))
        random_scores["sign_optional"] = rng.choice(
            [-1, 1],
            size=len(random_scores),
        )
        random_metrics = evaluate_scores(universe, random_scores)
        random_metrics = random_metrics.loc[
            random_metrics["cell_type"].ne("__mean__")
        ].copy()
        random_metrics.insert(0, "replicate", replicate_dir.name)
        random_metrics = random_metrics.merge(
            nodes[["cell_type", "node_type", "developmental_time"]],
            on="cell_type",
            validate="many_to_one",
        )
        metric_parts.append(random_metrics)

        ancestor_universe = universe.loc[universe["cell_type"].isin(ancestors)]
        baseline_metrics = evaluate_scores(
            ancestor_universe,
            pd.concat(baselines, ignore_index=True),
        )
        baseline_metrics = baseline_metrics.loc[
            baseline_metrics["cell_type"].ne("__mean__")
        ].copy()
        baseline_metrics.insert(0, "replicate", replicate_dir.name)
        baseline_parts.append(baseline_metrics)

        run_config = json.loads((run_dir / "run_config.json").read_text())
        for artifact, relative_path in {
            "expression": "method_inputs/RegArch/expression_data.csv",
            "atac": "method_inputs/RegArch/atac_data.csv",
            "fate_map": "method_inputs/RegArch/fate_map_edges.csv",
        }.items():
            path = replicate_dir / relative_path
            actual = _sha256(path)
            identity_rows.append(
                {
                    "replicate": replicate_dir.name,
                    "artifact": artifact,
                    "sha256": actual,
                    "recorded_sha256": run_config["input_sha256"][artifact],
                    "hash_match": (actual == run_config["input_sha256"][artifact]),
                }
            )

        diagnostic_paths = sorted((run_dir / "optimization_diagnostics").glob("*.json"))
        for path in diagnostic_paths:
            row = json.loads(path.read_text())
            row["replicate"] = replicate_dir.name
            optimizer_rows.append(row)

        lineage = pd.read_csv(
            replicate_dir / "diagnostics/lineage_signal.tsv",
            sep="\t",
        )
        lineage.insert(0, "replicate", replicate_dir.name)
        lineage_parts.append(lineage)

    metrics = pd.concat(metric_parts, ignore_index=True)
    baselines = pd.concat(baseline_parts, ignore_index=True)
    signs = pd.DataFrame(sign_rows).merge(
        pd.read_csv(replicate_dirs[0] / "truth/node_metadata.tsv", sep="\t",)[
            ["node_id", "node_type", "developmental_time"]
        ].rename(columns={"node_id": "cell_type"}),
        on="cell_type",
        validate="many_to_one",
    )
    optimizer = pd.DataFrame(optimizer_rows)
    identity = pd.DataFrame(identity_rows)
    lineage = pd.concat(lineage_parts, ignore_index=True)

    metrics.to_csv(output_dir / "node_metrics.tsv", sep="\t", index=False)
    baselines.to_csv(
        output_dir / "ancestor_baseline_metrics.tsv",
        sep="\t",
        index=False,
    )
    signs.to_csv(output_dir / "sign_metrics.tsv", sep="\t", index=False)
    optimizer.to_csv(
        output_dir / "optimizer_diagnostics.tsv",
        sep="\t",
        index=False,
    )
    identity.to_csv(
        output_dir / "input_identity_manifest.tsv",
        sep="\t",
        index=False,
    )
    lineage.to_csv(
        output_dir / "lineage_signal.tsv",
        sep="\t",
        index=False,
    )

    summary = (
        metrics.groupby(
            ["method", "node_type", "developmental_time"],
            as_index=False,
        )
        .agg(
            node_runs=("cell_type", "size"),
            auprc_mean=("auprc", "mean"),
            auprc_sd=("auprc", "std"),
            auroc_mean=("auroc", "mean"),
            precision_at_true_k_mean=("precision_at_true_k", "mean"),
            signed_precision_at_true_k_mean=(
                "signed_precision_at_true_k",
                "mean",
            ),
            signed_recall_at_true_k_mean=(
                "signed_recall_at_true_k",
                "mean",
            ),
            sign_accuracy_on_recovered_true_edges_mean=(
                "sign_accuracy_on_recovered_true_edges",
                "mean",
            ),
            sign_coverage_on_recovered_true_edges_mean=(
                "sign_coverage_on_recovered_true_edges",
                "mean",
            ),
        )
        .sort_values(["method", "developmental_time"])
    )
    summary.to_csv(output_dir / "node_summary.tsv", sep="\t", index=False)
    node_identity_summary = (
        metrics.groupby(
            [
                "method",
                "cell_type",
                "node_type",
                "developmental_time",
            ],
            as_index=False,
        )
        .agg(
            node_runs=("replicate", "size"),
            auprc_mean=("auprc", "mean"),
            auprc_sd=("auprc", "std"),
            auroc_mean=("auroc", "mean"),
            precision_at_true_k_mean=("precision_at_true_k", "mean"),
            signed_precision_at_true_k_mean=("signed_precision_at_true_k", "mean"),
            sign_accuracy_on_recovered_true_edges_mean=(
                "sign_accuracy_on_recovered_true_edges",
                "mean",
            ),
        )
        .sort_values(["method", "developmental_time", "cell_type"])
    )
    node_identity_summary.to_csv(
        output_dir / "node_identity_summary.tsv",
        sep="\t",
        index=False,
    )

    baseline_summary = baselines.groupby(["method", "cell_type"], as_index=False).agg(
        auprc_mean=("auprc", "mean"),
        auprc_sd=("auprc", "std"),
        auroc_mean=("auroc", "mean"),
    )
    baseline_summary.to_csv(
        output_dir / "ancestor_baseline_summary.tsv",
        sep="\t",
        index=False,
    )

    sign_summary = signs.groupby(
        ["node_type", "developmental_time"],
        as_index=False,
    ).agg(
        node_runs=("cell_type", "size"),
        sign_accuracy_mean=(
            "sign_accuracy_on_true_edges",
            "mean",
        ),
        predicted_positive_fraction_mean=(
            "predicted_positive_fraction",
            "mean",
        ),
        true_positive_fraction_mean=(
            "true_positive_fraction",
            "mean",
        ),
    )
    sign_summary.to_csv(
        output_dir / "sign_summary.tsv",
        sep="\t",
        index=False,
    )

    ancestor = metrics.loc[
        metrics["method"].eq("RegArch")
        & metrics["node_type"].isin(["root", "internal"])
    ]
    propagated = baselines.loc[baselines["method"].eq("descendant_leaf_mean")]
    paired = ancestor[["replicate", "cell_type", "auprc"]].merge(
        propagated[["replicate", "cell_type", "auprc"]],
        on=["replicate", "cell_type"],
        suffixes=("_regarch", "_propagation"),
    )
    paired["delta"] = paired["auprc_regarch"] - paired["auprc_propagation"]
    paired.to_csv(
        output_dir / "regarch_vs_propagation.tsv",
        sep="\t",
        index=False,
    )
    paired_rows = []
    for node, group in paired.groupby("cell_type"):
        values = group["delta"].to_numpy(dtype=float)
        standard_error = values.std(ddof=1) / np.sqrt(len(values))
        margin = float(t.ppf(0.975, len(values) - 1) * standard_error)
        paired_rows.append(
            {
                "cell_type": node,
                "replicate_count": len(values),
                "mean_auprc_delta": values.mean(),
                "ci95_low": values.mean() - margin,
                "ci95_high": values.mean() + margin,
                "win_rate": float(np.mean(values > 0)),
            }
        )
    paired_summary = pd.DataFrame(paired_rows)
    paired_summary.to_csv(
        output_dir / "regarch_vs_propagation_summary.tsv",
        sep="\t",
        index=False,
    )

    lineage_metrics = metrics.loc[metrics["method"].eq("RegArch")]
    random_metrics = metrics.loc[metrics["method"].eq("Random")]
    normalized = lineage_metrics.merge(
        random_metrics[["replicate", "cell_type", "auprc"]],
        on=["replicate", "cell_type"],
        suffixes=("_regarch", "_random"),
        validate="one_to_one",
    )
    normalized["normalized_auprc_lift"] = (
        normalized["auprc_regarch"] - normalized["auprc_random"]
    ) / (1.0 - normalized["auprc_random"])
    normalized.to_csv(
        output_dir / "random_normalized_metrics.tsv",
        sep="\t",
        index=False,
    )
    normalized.groupby(["node_type", "developmental_time"], as_index=False,).agg(
        node_runs=("cell_type", "size"),
        normalized_auprc_lift_mean=("normalized_auprc_lift", "mean"),
        normalized_auprc_lift_sd=("normalized_auprc_lift", "std"),
    ).to_csv(
        output_dir / "random_normalized_summary.tsv",
        sep="\t",
        index=False,
    )

    branches = lineage.loc[lineage["record_type"].eq("branch")]
    pairs = lineage.loc[lineage["record_type"].eq("leaf_pair")]
    lineage_node_summary = node_identity_summary.loc[
        node_identity_summary["method"].eq("RegArch")
    ]
    presence_rows = [
        "| Node | Type | Time | Mean AUPRC | Mean AUROC |",
        "|---|---|---:|---:|---:|",
    ]
    signed_rows = [
        "| Node | Signed precision at true K | Sign accuracy |",
        "|---|---:|---:|",
    ]
    for row in lineage_node_summary.itertuples(index=False):
        presence_rows.append(
            f"| {row.cell_type} | {row.node_type} | "
            f"{row.developmental_time:.2f} | {row.auprc_mean:.4f} | "
            f"{row.auroc_mean:.4f} |"
        )
        signed_rows.append(
            f"| {row.cell_type} | "
            f"{row.signed_precision_at_true_k_mean:.4f} | "
            f"{row.sign_accuracy_on_recovered_true_edges_mean:.4f} |"
        )
    report = f"""# Matched ultrametric simulation validation

This validation uses the exact GRN truth, RNA, ATAC, priors, and ultrametric
fate map from the retained benchmark. All leaves are sampled at time 1.
The same fitted RegArch posterior is evaluated at observed leaves and
unobserved ancestors, so the two node groups are directly matched by
replicate.

## Presence recovery

{chr(10).join(presence_rows)}

## Signed recovery

{chr(10).join(signed_rows)}

Mean ancestor AUPRC change relative to direct descendant-leaf score
propagation: {paired.delta.mean():+.4f}. The fraction of ancestor-node runs
where RegArch beats propagation is {(paired.delta > 0).mean():.3f}.
Node-specific paired 95% confidence intervals are reported in
`regarch_vs_propagation_summary.tsv`.

## Audit checks

- Optimizer convergence: {int(optimizer.success.sum())}/{len(optimizer)}.
- Input hashes matching recorded run configuration:
  {int(identity.hash_match.sum())}/{len(identity)}.
- Mean observed branch loss: {branches.observed_loss.mean():.4f};
  configured-model expectation averaged over branches:
  {branches.expected_loss.mean():.4f}.
- Mean sibling-leaf Jaccard: {
    pairs.loc[pairs.relationship.eq("sibling"), "jaccard"].mean():.4f}.
- Mean cross-clade Jaccard: {
    pairs.loc[pairs.relationship.eq("cross_clade"), "jaccard"].mean():.4f}.

## Interpretation boundary

The signed CTMC output has identical semantics at every node: absolute score is
the posterior edge-presence probability and score sign is the posterior MAP
activation/repression state. Presence and signed metrics are therefore directly
comparable across leaves and ancestors. Continuous ancestral effect magnitude
is not identified and is intentionally not reported.

Observed leaves and ancestors are different statistical tasks. Ancestors can be
easier when conserved edges dominate several descendants, so a monotonic
accuracy ordering by developmental time is not guaranteed even in a correct
simulation. The propagation and oracle-descendant baselines quantify that
effect.
"""
    (output_dir / "VALIDATION_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replicate_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", default="full")
    args = parser.parse_args()
    validate_matched_simulation(
        args.replicate_dirs,
        args.output_dir,
        args.run_name,
    )


if __name__ == "__main__":
    main()
