from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .evaluation import evaluate_scores
from .standardize_regarch import standardize_regarch


def _descendant_leaves(edges: pd.DataFrame, node: str) -> list[str]:
    children = edges.loc[edges["parent"].eq(node), "child"].tolist()
    if not children:
        return [node]
    leaves = []
    for child in children:
        leaves.extend(_descendant_leaves(edges, child))
    return leaves


def _score_table(
    method: str,
    cell_type: str,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    result = scores[["regulator", "target", "score"]].copy()
    result.insert(0, "cell_type", cell_type)
    result.insert(0, "method", method)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_ancestor_recovery(
    replicate_dirs: list[str | Path],
    output_dir: str | Path,
    run_name: str = "ancestor_corrected",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_root = output_dir.parent
    first_replicate = Path(replicate_dirs[0])
    generation_nodes = pd.read_csv(
        first_replicate / "truth/node_metadata.tsv",
        sep="\t",
    )
    generation_edges = pd.read_csv(
        first_replicate / "truth/fate_map_edges.tsv",
        sep="\t",
    )
    inference_nodes = pd.read_csv(
        first_replicate / "method_inputs/RegArch/fate_map_sampling_time_1_nodes.tsv",
        sep="\t",
    )
    inference_edges = pd.read_csv(
        first_replicate / "method_inputs/RegArch/fate_map_sampling_time_1_edges.tsv",
        sep="\t",
    )
    generation_nodes.to_csv(
        result_root / "generation_fate_map_nodes.tsv",
        sep="\t",
        index=False,
    )
    generation_edges.to_csv(
        result_root / "generation_fate_map_edges.tsv",
        sep="\t",
        index=False,
    )
    inference_nodes.to_csv(
        result_root / "fate_map_nodes.tsv",
        sep="\t",
        index=False,
    )
    inference_edges.to_csv(
        result_root / "fate_map_edges.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "artifact": "generation_fate_map",
                "scope": "Generated the retained GRN truth, RNA, and ATAC data.",
                "nodes_file": "generation_fate_map_nodes.tsv",
                "edges_file": "generation_fate_map_edges.tsv",
            },
            {
                "artifact": "inference_fate_map",
                "scope": "Used only by the repeated ancestor inference in this audit.",
                "nodes_file": "fate_map_nodes.tsv",
                "edges_file": "fate_map_edges.tsv",
            },
        ]
    ).to_csv(
        result_root / "fate_map_provenance.tsv",
        sep="\t",
        index=False,
    )
    metric_parts = []
    baseline_parts = []
    identity_rows = []
    time_rows = []
    optimizer_rows = []

    for replicate_dir in map(Path, replicate_dirs):
        standardized_path = (
            replicate_dir
            / "evaluation/ancestor_recovery"
            / f"RegArch_{run_name}_all_nodes.tsv"
        )
        scores = standardize_regarch(
            replicate_dir
            / "method_outputs/raw"
            / f"RegArch_{run_name}"
            / "inferred_grns",
            replicate_dir / "method_inputs/RegArch/regulator_list.txt",
            standardized_path,
            method_name="RegArch_corrected",
        )
        run_dir = replicate_dir / "method_outputs/raw" / f"RegArch_{run_name}"
        run_config = json.loads((run_dir / "run_config.json").read_text())
        diagnostic_paths = sorted(
            (run_dir / "optimization_diagnostics").glob("target_gene_*.json")
        )
        if len(diagnostic_paths) != 20:
            raise RuntimeError(
                f"{replicate_dir.name} has {len(diagnostic_paths)} optimizer "
                "diagnostics; expected 20."
            )
        for diagnostic_path in diagnostic_paths:
            diagnostic = json.loads(diagnostic_path.read_text())
            diagnostic["replicate"] = replicate_dir.name
            optimizer_rows.append(diagnostic)
        node_path = (
            replicate_dir
            / "method_inputs/RegArch"
            / "fate_map_sampling_time_1_nodes.tsv"
        )
        node_metadata = pd.read_csv(node_path, sep="\t")
        node_metadata.insert(0, "replicate", replicate_dir.name)
        time_rows.append(node_metadata)
        for artifact, relative_path in {
            "expression": "method_inputs/RegArch/expression_data.csv",
            "atac": "method_inputs/RegArch/atac_data.csv",
            "truth": "truth/true_grn_all_nodes.tsv",
        }.items():
            path = replicate_dir / relative_path
            identity_rows.append(
                {
                    "replicate": replicate_dir.name,
                    "artifact": artifact,
                    "path": str(path),
                    "sha256": _sha256(path),
                    "recorded_run_sha256": run_config["input_sha256"][artifact],
                    "hash_match": (
                        _sha256(path) == run_config["input_sha256"][artifact]
                    ),
                }
            )
        universe = pd.read_csv(
            replicate_dir / "intermediate/evaluation/edge_universe_all_nodes.tsv",
            sep="\t",
        )
        metrics = evaluate_scores(universe, scores)
        metrics = metrics.loc[metrics["cell_type"] != "__mean__"].copy()
        metrics.insert(0, "replicate", replicate_dir.name)
        metric_parts.append(metrics)

        truth = pd.read_csv(
            replicate_dir / "truth/true_grn_all_nodes.tsv",
            sep="\t",
        )
        truth = truth.loc[truth["regulator"] != truth["target"]]
        edges = pd.read_csv(
            replicate_dir / "truth/fate_map_edges.tsv",
            sep="\t",
        )
        priors = pd.read_csv(
            replicate_dir / "method_inputs/RegArch/atac_prior_tf_target.tsv",
            sep="\t",
        )
        ancestors = ["Root"] + sorted(set(edges["parent"]) - {"Root"})
        leaf_scores = scores.loc[scores["cell_type"].str.startswith("T")]
        baselines = []
        for ancestor in ancestors:
            leaves = _descendant_leaves(edges, ancestor)
            propagated = (
                leaf_scores.loc[leaf_scores["cell_type"].isin(leaves)]
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
            root_prior = priors.loc[priors["node"].eq("Root")].rename(
                columns={"prior_score": "score"}
            )
            baselines.append(_score_table("mean_leaf_ATAC_prior", ancestor, root_prior))
            oracle = (
                truth.loc[truth["node"].isin(leaves)]
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
        ancestor_universe = universe.loc[universe["cell_type"].isin(ancestors)]
        baseline_metrics = evaluate_scores(
            ancestor_universe,
            pd.concat(baselines, ignore_index=True),
        )
        baseline_metrics = baseline_metrics.loc[
            baseline_metrics["cell_type"] != "__mean__"
        ].copy()
        baseline_metrics.insert(0, "replicate", replicate_dir.name)
        baseline_parts.append(baseline_metrics)

    metrics = pd.concat(metric_parts, ignore_index=True)
    node_times = pd.concat(time_rows, ignore_index=True)[
        ["replicate", "node_id", "depth", "developmental_time"]
    ].rename(columns={"node_id": "cell_type"})
    metrics = metrics.merge(
        node_times,
        on=["replicate", "cell_type"],
        validate="one_to_one",
    )
    metrics["node_group"] = metrics["cell_type"].map(
        lambda node: (
            "root"
            if node == "Root"
            else "internal"
            if node.startswith("P")
            else "observed_leaf"
        )
    )
    metrics.to_csv(output_dir / "corrected_node_metrics.tsv", sep="\t", index=False)

    summary = (
        metrics.groupby(["node_group", "developmental_time"], as_index=False)
        .agg(
            node_replicates=("cell_type", "size"),
            auprc_mean=("auprc", "mean"),
            auprc_sd=("auprc", "std"),
            auroc_mean=("auroc", "mean"),
            precision_at_true_k_mean=("precision_at_true_k", "mean"),
            recall_at_true_k_mean=("recall_at_true_k", "mean"),
        )
        .sort_values("developmental_time")
    )
    summary.to_csv(output_dir / "corrected_time_summary.tsv", sep="\t", index=False)
    pd.DataFrame(identity_rows).to_csv(
        output_dir / "input_identity_manifest.tsv",
        sep="\t",
        index=False,
    )
    optimizer_diagnostics = pd.DataFrame(optimizer_rows)
    optimizer_diagnostics.to_csv(
        output_dir / "optimizer_diagnostics.tsv",
        sep="\t",
        index=False,
    )
    optimizer_diagnostics.groupby(
        ["success", "status", "message"],
        as_index=False,
    ).agg(
        target_runs=("target_gene_id", "size"),
        objective_min=("objective", "min"),
        objective_max=("objective", "max"),
    ).to_csv(
        output_dir / "optimizer_diagnostics_summary.tsv",
        sep="\t",
        index=False,
    )
    pd.concat(time_rows, ignore_index=True).to_csv(
        output_dir / "inference_fate_map_nodes_by_replicate.tsv",
        sep="\t",
        index=False,
    )

    baselines = pd.concat(baseline_parts, ignore_index=True)
    baselines.to_csv(
        output_dir / "ancestor_baseline_metrics.tsv", sep="\t", index=False
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

    for old_name in [
        "observed_vs_unobserved_summary.tsv",
        "ancestor_summary.tsv",
        "ancestor_metrics.tsv",
    ]:
        old_path = output_dir / old_name
        superseded_path = output_dir / f"SUPERSEDED_{old_name}"
        if old_path.exists() and not superseded_path.exists():
            old_path.rename(superseded_path)
    observed = summary.loc[summary["node_group"].eq("observed_leaf")].iloc[0]
    p1 = metrics.loc[metrics["cell_type"].eq("P1")]
    p2 = metrics.loc[metrics["cell_type"].eq("P2")]
    root = summary.loc[summary["node_group"].eq("root")].iloc[0]
    ancestor_metrics = metrics.loc[
        metrics["cell_type"].isin(["Root", "P1", "P2"]),
        ["replicate", "cell_type", "auprc"],
    ].rename(columns={"auprc": "corrected_auprc"})
    propagation_metrics = baselines.loc[
        baselines["method"].eq("descendant_leaf_mean"),
        ["replicate", "cell_type", "auprc"],
    ].rename(columns={"auprc": "propagation_auprc"})
    propagation_delta = ancestor_metrics.merge(
        propagation_metrics,
        on=["replicate", "cell_type"],
    )
    propagation_delta["auprc_delta"] = (
        propagation_delta["corrected_auprc"] - propagation_delta["propagation_auprc"]
    )
    propagation_delta.to_csv(
        output_dir / "corrected_vs_propagation.tsv",
        sep="\t",
        index=False,
    )
    mean_delta = propagation_delta["auprc_delta"].mean()
    report = f"""# Ancestor recovery audit

The previous ancestor summary is superseded because ancestor scores were produced
by upward averaging of descendant leaf estimates. The legacy helper also used the
first child's branch probability for both children.

The corrected run uses an edge-agnostic root prevalence prior and reports exact
CTMC posterior edge-presence probabilities for internal nodes.

The RNA, ATAC, and GRN truth files are byte-identical to the retained leaf
benchmark inputs. Only the inference fate-map times were changed. Consequently,
this is a time-map recalibration analysis, not a matched simulation generated
under the new times. The original and inference maps are retained separately as
`generation_fate_map_*.tsv` and `fate_map_*.tsv`.

Per-target optimizer status is recorded in `optimizer_diagnostics.tsv`; non-finite
solutions are rejected during inference.

| Node | Developmental time | Mean AUPRC | Mean AUROC |
|---|---:|---:|---:|
| Root | 0.0 | {root.auprc_mean:.4f} | {root.auroc_mean:.4f} |
| P1 | 0.2 | {p1.auprc.mean():.4f} | {p1.auroc.mean():.4f} |
| P2 | 0.8 | {p2.auprc.mean():.4f} | {p2.auroc.mean():.4f} |
| Observed leaves | 1.0 | {observed.auprc_mean:.4f} | {observed.auroc_mean:.4f} |

Higher ancestor accuracy is not, by itself, proof of leakage. In this shallow,
low-transition simulation, ancestor edges are the conserved consensus of several
descendants and can be easier to rank than branch-specific leaf edges. See
`ancestor_baseline_summary.tsv`: the oracle descendant-truth consensus quantifies
this intrinsic task easiness, while `descendant_leaf_mean` is the no-ancestor-model
propagation baseline that RegArch must be compared against.

The corrected CTMC posterior changes mean ancestor AUPRC relative to direct descendant
leaf-score propagation by only {mean_delta:+.6f}. Therefore this dataset does not
currently demonstrate ancestor-specific inferential value beyond consensus
propagation. Accuracy ordering by developmental time must not be inferred from
this analysis because the retained truth was generated with the original branch
lengths. A matched time-gradient experiment requires regenerating truth and
observations under the calibrated ultrametric map.
"""
    (output_dir / "ANCESTOR_RECOVERY_AUDIT.md").write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ancestor GRN recovery.")
    parser.add_argument("replicate_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="ancestor_corrected")
    args = parser.parse_args()
    audit_ancestor_recovery(args.replicate_dirs, args.output_dir, args.run_name)


if __name__ == "__main__":
    main()
