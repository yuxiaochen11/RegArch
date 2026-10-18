from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _row(
    section: str,
    parameter: str,
    configured_value: object,
    unit: str,
    description: str,
    source: str,
    actual_summary: str = "",
) -> dict:
    return {
        "section": section,
        "parameter": parameter,
        "configured_value": configured_value,
        "unit": unit,
        "description": description,
        "source": source,
        "actual_summary_10rep": actual_summary,
    }


def summarize_dataset(
    config_path: str | Path,
    replicate_dirs: list[str | Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    replicate_dirs = list(map(Path, replicate_dirs))
    first = replicate_dirs[0]
    rows = [
        _row(
            "dataset",
            "dataset_name",
            config["name"],
            "",
            "Dataset prefix.",
            str(config_path),
        ),
        _row(
            "dataset",
            "seed",
            config["seed"],
            "",
            "Base random seed; replicate index is added through SeedSequence.",
            str(config_path),
        ),
        _row(
            "dataset",
            "replicates",
            config["replicates"],
            "replicates",
            "Independent generated datasets.",
            str(config_path),
        ),
    ]
    descriptions = {
        "fate_map.node_count": (
            "nodes",
            "Total nodes in a balanced full binary fate map.",
        ),
        "fate_map.branch_lengths": (
            "scaled branch length",
            "Ordered Root-to-child branch lengths used by the CTMC.",
        ),
        "fate_map.wrong_fate_map_mode": (
            "category",
            "Topology-only perturbation used for the wrong-fate-map ablation.",
        ),
        "grn.regulator_count": ("genes", "Candidate regulator/TF count."),
        "grn.target_count": (
            "genes",
            "Total unique target genes; regulators are included in this set.",
        ),
        "grn.density": (
            "fraction",
            "Root-network edge density before per-target coverage enforcement.",
        ),
        "grn.positive_fraction": (
            "fraction",
            "Probability that a newly generated edge is activating.",
        ),
        "grn.weight_mean": ("weight", "Mean of the absolute Normal edge-weight draw."),
        "grn.weight_sd": ("weight", "SD of the absolute Normal edge-weight draw."),
        "grn.weight_min": ("weight", "Minimum clipped nonzero edge magnitude."),
        "grn.weight_max": ("weight", "Maximum clipped nonzero edge magnitude."),
        "grn.loss_rate": (
            "rate/unit branch length",
            "CTMC edge-presence to absence rate.",
        ),
        "grn.gain_rate": (
            "rate/unit branch length",
            "CTMC absence to edge-presence rate.",
        ),
        "grn.sign_switch_rate": (
            "rate/unit branch length",
            "CTMC direct activation/repression switching rate.",
        ),
        "grn.weight_drift_sd": (
            "weight/sqrt(branch length)",
            "Parent-child nonzero edge magnitude drift.",
        ),
        "grn.ensure_each_target": (
            "boolean",
            "Forces at least one incoming edge at the root for every target.",
        ),
        "grn.allow_self_regulation": (
            "boolean",
            "Whether TF self-regulatory candidates are allowed.",
        ),
        "regions.promoter_per_target": (
            "peaks/target",
            "Promoter peaks generated per target.",
        ),
        "regions.enhancer_min": (
            "peaks/target",
            "Minimum enhancer peaks generated per target.",
        ),
        "regions.enhancer_max": (
            "peaks/target",
            "Maximum enhancer peaks generated per target.",
        ),
        "regions.decoy_min": (
            "peaks/target",
            "Minimum decoy peaks generated per target.",
        ),
        "regions.decoy_max": (
            "peaks/target",
            "Maximum decoy peaks generated per target.",
        ),
        "regions.regulatory_regions_min": (
            "regions/true edge",
            "Minimum active regions assigned to a true TF-target edge.",
        ),
        "regions.regulatory_regions_max": (
            "regions/true edge",
            "Maximum active regions assigned to a true TF-target edge.",
        ),
        "regions.motif_true_positive_rate": (
            "probability",
            "Motif-hit probability for true TF-region pairs.",
        ),
        "regions.motif_false_positive_rate": (
            "probability",
            "Motif-hit probability for false TF-region pairs.",
        ),
        "regions.link_false_positive_rate": (
            "probability",
            "Probability that a decoy region is emitted as a candidate peak-gene link.",
        ),
        "observations.cells_per_leaf": (
            "cells/leaf",
            "Paired RNA/ATAC cells observed at each leaf.",
        ),
        "observations.rna_library_mean": (
            "counts/cell",
            "Log-normal RNA library-size mean.",
        ),
        "observations.rna_library_sd": ("log scale", "Log-normal RNA library-size SD."),
        "observations.rna_dispersion": (
            "NB dispersion",
            "Negative-binomial RNA dispersion.",
        ),
        "observations.regulatory_scale": (
            "coefficient",
            "Maximum scale applied to GRN regulatory signal.",
        ),
        "observations.regulator_noise_sd": (
            "log-expression",
            "Regulator structural-equation noise SD.",
        ),
        "observations.target_noise_sd": (
            "log-expression",
            "Target-expression noise SD.",
        ),
        "observations.rna_dropout_rate": (
            "maximum probability",
            "Expression-dependent RNA dropout multiplier.",
        ),
        "observations.atac_depth_sd": (
            "log scale",
            "Cell-specific ATAC depth-factor SD.",
        ),
        "observations.active_peak_alpha": (
            "beta shape",
            "Active-peak accessibility Beta alpha.",
        ),
        "observations.active_peak_beta": (
            "beta shape",
            "Active-peak accessibility Beta beta.",
        ),
        "observations.inactive_peak_alpha": (
            "beta shape",
            "Inactive-peak accessibility Beta alpha.",
        ),
        "observations.inactive_peak_beta": (
            "beta shape",
            "Inactive-peak accessibility Beta beta.",
        ),
        "observations.decoy_peak_alpha": (
            "beta shape",
            "Decoy-peak accessibility Beta alpha.",
        ),
        "observations.decoy_peak_beta": (
            "beta shape",
            "Decoy-peak accessibility Beta beta.",
        ),
        "observations.metacells_per_leaf": (
            "metacells/leaf",
            "Requested metacell groups for methods using aggregation.",
        ),
    }
    for section in ["fate_map", "grn", "regions", "observations"]:
        for key, value in config[section].items():
            name = f"{section}.{key}"
            unit, description = descriptions[name]
            rows.append(
                _row(
                    section,
                    name,
                    json.dumps(value) if isinstance(value, list) else value,
                    unit,
                    description,
                    str(config_path),
                )
            )

    inference_rows = [
        ("optimizer", "L-BFGS-B", "optimizer", "RegArch numerical optimizer."),
        (
            "lineage_model",
            "ctmc_signed",
            "",
            "Simulator-matched signed three-state CTMC lineage prior.",
        ),
        (
            "lineage_weight",
            0.5,
            "objective weight",
            "Lineage-prior weight selected on independent calibration replicates.",
        ),
        ("atac_weight", 1.0, "objective weight", "ATAC-prior objective weight."),
        (
            "likelihood_weight",
            1.0,
            "objective weight",
            "RNA likelihood objective weight.",
        ),
        (
            "root_prior",
            "atac",
            "",
            "Edge-specific root-state prior derived from observed leaf ATAC evidence.",
        ),
        (
            "optimizer_maxfun",
            60000,
            "function evaluations",
            "Maximum function evaluations for each target-gene fit.",
        ),
        ("processes", 1, "process", "Parallel target-gene workers."),
        (
            "lineage_gain_rate",
            config["grn"]["gain_rate"],
            "rate/unit branch length",
            "Inference CTMC gain rate matched to the simulator.",
        ),
        (
            "lineage_loss_rate",
            config["grn"]["loss_rate"],
            "rate/unit branch length",
            "Inference CTMC loss rate matched to the simulator.",
        ),
        (
            "lineage_sign_switch_rate",
            config["grn"].get("sign_switch_rate", 0.0),
            "rate/unit branch length",
            "Inference CTMC sign-switch rate matched to the simulator.",
        ),
        (
            "lineage_positive_fraction",
            config["grn"].get("positive_fraction", 0.5),
            "fraction",
            "Positive share of newly gained edges, matched to the simulator.",
        ),
    ]
    for name, value, unit, description in inference_rows:
        rows.append(
            _row(
                "regarch_inference",
                f"regarch.{name}",
                value,
                unit,
                description,
                "method_outputs/raw/RegArch_full/run_config.json",
            )
        )
    evaluation_rows = [
        (
            "candidate_edges_per_node",
            config["grn"]["regulator_count"] * config["grn"]["target_count"]
            - config["grn"]["regulator_count"],
            "edges/node",
            "All TF-target pairs excluding prohibited TF self-pairs.",
        ),
        (
            "primary_metric",
            "AUPRC",
            "",
            "Average precision on unsigned edge-presence labels.",
        ),
        (
            "secondary_metric",
            "AUROC",
            "",
            "Area under the receiver-operating-characteristic curve.",
        ),
        (
            "top_k",
            "true edge count per node",
            "edges",
            "K used for precision and recall at K.",
        ),
        (
            "missing_edge_score",
            0.0,
            "score",
            "Unreported candidate edges are assigned zero.",
        ),
        (
            "score_mode",
            "absolute inferred weight",
            "",
            "Primary comparison ignores activation/repression sign.",
        ),
        (
            "leaf_nodes",
            ",".join(
                pd.read_csv(first / "truth/node_metadata.tsv", sep="\t").loc[
                    lambda x: x["node_type"].eq("leaf"), "node_id"
                ]
            ),
            "",
            "Observed nodes used in the cross-method benchmark.",
        ),
        (
            "ancestor_nodes",
            ",".join(
                pd.read_csv(first / "truth/node_metadata.tsv", sep="\t").loc[
                    lambda x: x["node_type"].isin(["root", "internal"]),
                    "node_id",
                ]
            ),
            "",
            "Unobserved nodes used in RegArch ancestor recovery.",
        ),
    ]
    for name, value, unit, description in evaluation_rows:
        rows.append(
            _row(
                "evaluation",
                f"evaluation.{name}",
                value,
                unit,
                description,
                "benchmark/evaluation.py",
            )
        )
    rows.extend(
        [
            _row(
                "derived_distribution",
                "active_peak_mean_accessibility",
                8 / 10,
                "probability",
                "Mean of Beta(8,2).",
                str(config_path),
            ),
            _row(
                "derived_distribution",
                "inactive_peak_mean_accessibility",
                2 / 10,
                "probability",
                "Mean of Beta(2,8).",
                str(config_path),
            ),
            _row(
                "derived_distribution",
                "decoy_peak_mean_accessibility",
                1 / 13,
                "probability",
                "Mean of Beta(1,12).",
                str(config_path),
            ),
            _row(
                "derived_distribution",
                "positive_gain_rate",
                config["grn"]["gain_rate"] * config["grn"]["positive_fraction"],
                "rate/unit branch length",
                "CTMC zero-to-positive transition-generator rate.",
                "benchmark/dynamic_grn.py",
            ),
            _row(
                "derived_distribution",
                "negative_gain_rate",
                config["grn"]["gain_rate"] * (1 - config["grn"]["positive_fraction"]),
                "rate/unit branch length",
                "CTMC zero-to-negative transition-generator rate.",
                "benchmark/dynamic_grn.py",
            ),
        ]
    )

    actual_rows = []
    for run in replicate_dirs:
        nodes = pd.read_csv(run / "truth/node_metadata.tsv", sep="\t")
        truth = pd.read_csv(run / "truth/true_grn_all_nodes.tsv", sep="\t")
        metadata = pd.read_csv(
            run / "intermediate/metadata/cell_metadata.tsv", sep="\t"
        )
        counts = pd.read_csv(
            run / "intermediate/rna/rna_counts_gene_by_cell.tsv", sep="\t", index_col=0
        )
        atac = pd.read_csv(
            run / "intermediate/atac/atac_counts_region_by_cell.tsv",
            sep="\t",
            index_col=0,
        )
        peaks = pd.read_csv(run / "intermediate/atac/peak_metadata.tsv", sep="\t")
        motifs = pd.read_csv(run / "intermediate/atac/motif_hits.tsv", sep="\t")
        links = pd.read_csv(
            run / "intermediate/atac/region_to_gene_links.tsv", sep="\t"
        )
        assignments = pd.read_csv(run / "truth/true_tf_region_gene_edges.tsv", sep="\t")
        leaf_truth = truth[truth["node"].isin(nodes.loc[nodes["is_leaf"], "node_id"])]
        actual_rows.append(
            {
                "replicate": run.name,
                "nodes": len(nodes),
                "internal_nodes": int((~nodes["is_leaf"]).sum()),
                "leaves": int(nodes["is_leaf"].sum()),
                "max_depth": int(nodes["depth"].max()),
                "cells": len(metadata),
                "genes": counts.shape[0],
                "regulators": int(
                    pd.read_csv(
                        run / "intermediate/metadata/gene_metadata.tsv", sep="\t"
                    )["is_regulator"].sum()
                ),
                "peaks": atac.shape[0],
                "promoters": int((peaks["region_type"] == "promoter").sum()),
                "enhancers": int((peaks["region_type"] == "enhancer").sum()),
                "decoys": int((peaks["region_type"] == "decoy").sum()),
                "rna_total_counts": int(counts.to_numpy().sum()),
                "rna_zero_fraction": float((counts.to_numpy() == 0).mean()),
                "atac_accessible_fraction": float((atac.to_numpy() > 0).mean()),
                "motif_hits": int(motifs["motif_hit"].sum()),
                "candidate_peak_gene_links": int(links["is_candidate_link"].sum()),
                "true_tf_region_gene_assignments": len(assignments),
                "root_true_edges": int(
                    truth.loc[truth["node"].eq("Root"), "true_edge"].sum()
                ),
                "leaf_true_edges_mean": float(
                    leaf_truth.groupby("node")["true_edge"].sum().mean()
                ),
                "leaf_true_edges_min": int(
                    leaf_truth.groupby("node")["true_edge"].sum().min()
                ),
                "leaf_true_edges_max": int(
                    leaf_truth.groupby("node")["true_edge"].sum().max()
                ),
            }
        )
    actual = pd.DataFrame(actual_rows)
    for column in actual.columns[1:]:
        values = actual[column]
        summary = (
            f"mean={values.mean():.6g}; sd={values.std():.6g}; "
            f"min={values.min():.6g}; max={values.max():.6g}"
        )
        rows.append(
            _row(
                "actual_dataset",
                f"actual.{column}",
                "",
                "",
                "Observed value computed from generated files.",
                "benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_*",
                summary,
            )
        )
    return pd.DataFrame(rows), actual


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark parameters.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("replicate_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameters, actual = summarize_dataset(args.config, args.replicate_dirs)
    parameters.to_csv(
        args.output_dir / "dataset_parameters.tsv",
        sep="\t",
        index=False,
    )
    actual.to_csv(
        args.output_dir / "dataset_actual_statistics.tsv",
        sep="\t",
        index=False,
    )
    first = args.replicate_dirs[0]
    pd.read_csv(first / "truth/node_metadata.tsv", sep="\t").to_csv(
        args.output_dir / "fate_map_nodes.tsv", sep="\t", index=False
    )
    pd.read_csv(first / "truth/fate_map_edges.tsv", sep="\t").to_csv(
        args.output_dir / "fate_map_edges.tsv", sep="\t", index=False
    )
    pd.read_csv(first / "truth/rewiring_transition_log.tsv", sep="\t").to_csv(
        args.output_dir / "example_rewiring_transitions.tsv",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()
