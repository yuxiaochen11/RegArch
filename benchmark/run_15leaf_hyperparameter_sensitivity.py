from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import BenchmarkConfig, load_config, save_config
from .evaluation import build_edge_universe, evaluate_scores
from .standardize_regarch import standardize_regarch


@dataclass(frozen=True)
class Condition:
    name: str
    dataset: str
    factor: str
    optimizer: str
    optimizer_maxfun: int
    cells_per_leaf: int = 30
    regulator_count: int = 8
    target_count: int = 20
    reuse_existing_full: bool = False


CONDITIONS = [
    Condition(
        "baseline_cells30_genes20_lbfgsb_max60000",
        "lineage_signal_15leaf",
        "reference_existing_15leaf_data",
        "L-BFGS-B",
        60000,
        reuse_existing_full=True,
    ),
    Condition(
        "cells15_genes20_lbfgsb_max60000",
        "hyperparam_15leaf_cells15",
        "cells_per_leaf",
        "L-BFGS-B",
        60000,
        cells_per_leaf=15,
    ),
    Condition(
        "cells60_genes20_lbfgsb_max60000",
        "hyperparam_15leaf_cells60",
        "cells_per_leaf",
        "L-BFGS-B",
        60000,
        cells_per_leaf=60,
    ),
    Condition(
        "cells30_genes40_lbfgsb_max60000",
        "hyperparam_15leaf_genes40",
        "gene_count",
        "L-BFGS-B",
        60000,
        regulator_count=12,
        target_count=40,
    ),
    Condition(
        "baseline_lbfgsb_max20000",
        "lineage_signal_15leaf",
        "optimizer_budget_existing_15leaf_data",
        "L-BFGS-B",
        20000,
    ),
    Condition(
        "baseline_lbfgsb_max120000",
        "lineage_signal_15leaf",
        "optimizer_budget_existing_15leaf_data",
        "L-BFGS-B",
        120000,
    ),
    Condition(
        "baseline_slsqp_max20000",
        "lineage_signal_15leaf",
        "optimizer_method_existing_15leaf_data",
        "SLSQP",
        20000,
    ),
    Condition(
        "baseline_cobyla_max20000",
        "lineage_signal_15leaf",
        "optimizer_method_existing_15leaf_data",
        "COBYLA",
        20000,
    ),
]


def _replicate_dirs(root: Path, dataset: str, replicates: int) -> list[Path]:
    nested = sorted((root / dataset).glob(f"{dataset}_rep_*"))
    flat = sorted(root.glob(f"{dataset}_rep_*"))
    dirs = nested if nested else flat
    return dirs[:replicates]


def _write_variant_config(
    base_config: BenchmarkConfig,
    condition: Condition,
    config_dir: Path,
    replicates: int,
    seed: int,
) -> Path:
    config = deepcopy(base_config)
    config.name = condition.dataset
    config.seed = seed
    config.replicates = replicates
    config.observations.cells_per_leaf = condition.cells_per_leaf
    config.grn.regulator_count = condition.regulator_count
    config.grn.target_count = condition.target_count
    if condition.target_count >= 40:
        config.regions.enhancer_min = max(config.regions.enhancer_min, 2)
        config.regions.enhancer_max = max(config.regions.enhancer_max, 3)
    path = config_dir / f"{condition.dataset}_{replicates}rep.json"
    save_config(config, path)
    return path


def _ensure_generated(
    condition: Condition,
    base_config: BenchmarkConfig,
    config_dir: Path,
    run_root: Path,
    replicates: int,
    seed: int,
) -> list[Path]:
    dirs = _replicate_dirs(run_root, condition.dataset, replicates)
    if len(dirs) >= replicates:
        return dirs
    config_path = _write_variant_config(
        base_config, condition, config_dir, replicates, seed
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark.cli",
            "generate",
            "--config",
            str(config_path),
            "--output",
            str(run_root),
        ],
        check=True,
    )
    dirs = _replicate_dirs(run_root, condition.dataset, replicates)
    if len(dirs) < replicates:
        raise RuntimeError(
            f"Expected {replicates} generated replicates for {condition.dataset}; "
            f"found {len(dirs)}."
        )
    return dirs


def _output_complete(raw_dir: Path, target_count: int) -> bool:
    inferred = raw_dir / "inferred_grns"
    if len(list(inferred.glob("target_gene_*.csv"))) != target_count:
        return False
    diagnostics = raw_dir / "optimization_diagnostics"
    if not diagnostics.exists():
        diagnostics = raw_dir / "optimizer_diagnostics"
    return len(list(diagnostics.glob("target_gene_*.json"))) == target_count


def _regarch_raw_dir(replicate_dir: Path, condition: Condition) -> Path:
    if condition.reuse_existing_full:
        return replicate_dir / "method_outputs" / "raw" / "RegArch_full"
    return (
        replicate_dir
        / "method_outputs"
        / "raw"
        / f"RegArch_hyperparam_{condition.name}"
    )


def _estimate_existing_runtime_seconds(raw_dir: Path) -> float | None:
    if not raw_dir.exists():
        return None
    files = list((raw_dir / "inferred_grns").glob("target_gene_*.csv"))
    files += list((raw_dir / "optimization_diagnostics").glob("target_gene_*.json"))
    files += list((raw_dir / "optimizer_diagnostics").glob("target_gene_*.json"))
    if not files:
        return None
    started = raw_dir.stat().st_ctime
    finished = max(path.stat().st_mtime for path in files)
    return max(0.0, finished - started)


def _run_condition(
    replicate_dir: Path,
    condition: Condition,
    processes: int,
) -> dict[str, object]:
    target_count = int(
        json.loads((replicate_dir / "config.json").read_text())["grn"]["target_count"]
    )
    raw_dir = _regarch_raw_dir(replicate_dir, condition)
    started = time.perf_counter()
    skipped = False
    if not _output_complete(raw_dir, target_count):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmark.run_regarch",
                str(replicate_dir),
                "--variant",
                "full",
                "--run-name",
                f"hyperparam_{condition.name}",
                "--processes",
                str(processes),
                "--lineage-model",
                "ctmc_signed",
                "--root-prior",
                "atac",
                "--lineage-weight",
                "0.5",
                "--atac-weight",
                "1.0",
                "--likelihood-weight",
                "1.0",
                "--optimizer",
                condition.optimizer,
                "--optimizer-maxfun",
                str(condition.optimizer_maxfun),
            ],
            check=True,
        )
    else:
        skipped = True
    elapsed = time.perf_counter() - started
    runtime_seconds = (
        _estimate_existing_runtime_seconds(raw_dir) if skipped else elapsed
    )
    return {
        "replicate": replicate_dir.name,
        "condition": condition.name,
        "dataset": condition.dataset,
        "optimizer": condition.optimizer,
        "optimizer_maxfun": condition.optimizer_maxfun,
        "runtime_seconds": runtime_seconds,
        "runtime_source": (
            "filesystem_timestamps_existing_output"
            if skipped
            else "measured_wall_clock"
        ),
        "skipped_existing_output": skipped,
    }


def _collect_optimizer_diagnostics(
    replicate_dir: Path,
    condition: Condition,
) -> pd.DataFrame:
    rows = []
    raw_dir = _regarch_raw_dir(replicate_dir, condition)
    diagnostics_dir = raw_dir / "optimization_diagnostics"
    if not diagnostics_dir.exists():
        diagnostics_dir = raw_dir / "optimizer_diagnostics"
    for path in sorted(diagnostics_dir.glob("target_gene_*.json")):
        data = json.loads(path.read_text())
        data.update(
            {
                "replicate": replicate_dir.name,
                "condition": condition.name,
                "dataset": condition.dataset,
                "target_file": path.name,
            }
        )
        rows.append(data)
    return pd.DataFrame(rows)


def _evaluate_replicate(
    replicate_dir: Path,
    condition: Condition,
    output_dir: Path,
) -> pd.DataFrame:
    condition_dir = output_dir / "condition_runs" / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    score_path = condition_dir / f"{replicate_dir.name}_{condition.name}_scores.tsv"
    metrics_path = (
        condition_dir / f"{replicate_dir.name}_{condition.name}_all_nodes.tsv"
    )
    raw_dir = _regarch_raw_dir(replicate_dir, condition)
    standardize_regarch(
        raw_dir / "inferred_grns",
        replicate_dir / "method_inputs" / "RegArch" / "regulator_list.txt",
        score_path,
        method_name=condition.name,
    )
    true_grn = pd.read_csv(replicate_dir / "truth" / "true_grn_all_nodes.tsv", sep="\t")
    edge_universe = build_edge_universe(true_grn)
    scores = pd.read_csv(score_path, sep="\t")
    metrics = evaluate_scores(edge_universe, scores)
    metrics = metrics.loc[metrics["cell_type"].ne("__mean__")].copy()
    node_meta = pd.read_csv(replicate_dir / "truth" / "node_metadata.tsv", sep="\t")
    metrics = metrics.merge(
        node_meta,
        left_on="cell_type",
        right_on="node_id",
        how="left",
    )
    metrics["replicate"] = replicate_dir.name
    metrics["condition"] = condition.name
    metrics["dataset"] = condition.dataset
    metrics["node_group"] = metrics["node_type"].replace({"internal": "ancestor"})
    metrics.to_csv(metrics_path, sep="\t", index=False)
    return metrics


def _candidate_edges(regulators: int, targets: int) -> int:
    return regulators * targets - min(regulators, targets)


def _write_summaries(
    output_dir: Path,
    metrics: pd.DataFrame,
    runtimes: pd.DataFrame,
    diagnostics: pd.DataFrame,
    replicates: int,
    processes: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        output_dir / "hyperparameter_node_metrics.tsv", sep="\t", index=False
    )
    group_cols = ["condition", "dataset", "node_group"]
    value_cols = [
        "auprc",
        "auroc",
        "precision_at_true_k",
        "signed_precision_at_true_k",
        "top10_regulator_overlap",
    ]
    summary = metrics.groupby(group_cols, as_index=False)[value_cols].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary.to_csv(
        output_dir / "hyperparameter_group_summary.tsv", sep="\t", index=False
    )
    for group_name, file_name in [
        ("leaf", "hyperparameter_leaf_summary.tsv"),
        ("ancestor", "hyperparameter_ancestor_summary.tsv"),
        ("root", "hyperparameter_root_summary.tsv"),
    ]:
        summary.loc[summary["node_group"].eq(group_name)].to_csv(
            output_dir / file_name,
            sep="\t",
            index=False,
        )

    baseline = summary.loc[
        (summary["condition"].eq("baseline_cells30_genes20_lbfgsb_max60000"))
        & (summary["node_group"].eq("leaf"))
    ].set_index("condition")
    if not baseline.empty:
        base_auprc = float(baseline.iloc[0]["auprc_mean"])
        base_auroc = float(baseline.iloc[0]["auroc_mean"])
        leaf_delta = summary.loc[summary["node_group"].eq("leaf")].copy()
        leaf_delta["delta_auprc_vs_baseline"] = leaf_delta["auprc_mean"] - base_auprc
        leaf_delta["delta_auroc_vs_baseline"] = leaf_delta["auroc_mean"] - base_auroc
        leaf_delta.to_csv(
            output_dir / "hyperparameter_leaf_delta_vs_baseline.tsv",
            sep="\t",
            index=False,
        )

    settings = pd.DataFrame(
        [
            {
                "condition": condition.name,
                "dataset": condition.dataset,
                "factor": condition.factor,
                "leaf_nodes": 15,
                "replicates": replicates,
                "cells_per_leaf": condition.cells_per_leaf,
                "regulator_count": condition.regulator_count,
                "target_count": condition.target_count,
                "candidate_edges_per_node": _candidate_edges(
                    condition.regulator_count,
                    condition.target_count,
                ),
                "optimizer": condition.optimizer,
                "optimizer_maxfun": condition.optimizer_maxfun,
                "processes": processes,
                "data_source": (
                    "existing lineage_signal_15leaf first replicates"
                    if condition.dataset == "lineage_signal_15leaf"
                    else "newly generated 15-leaf hyperparameter dataset"
                ),
            }
            for condition in CONDITIONS
        ]
    )
    settings.to_csv(output_dir / "hyperparameter_settings.tsv", sep="\t", index=False)

    runtimes.to_csv(output_dir / "regarch_runtime.tsv", sep="\t", index=False)
    if not runtimes.empty:
        (
            runtimes.loc[runtimes["runtime_seconds"].notna()]
            .groupby("condition", as_index=False)["runtime_seconds"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
            .to_csv(output_dir / "regarch_runtime_summary.tsv", sep="\t", index=False)
        )
    diagnostics.to_csv(
        output_dir / "optimization_diagnostics.tsv", sep="\t", index=False
    )
    if not diagnostics.empty and "success" in diagnostics.columns:
        objective_column = (
            "objective"
            if "objective" in diagnostics.columns
            else "fun"
            if "fun" in diagnostics.columns
            else None
        )
        aggregation = {
            "target_fits": ("success", "size"),
            "successful_fits": ("success", "sum"),
        }
        if objective_column:
            aggregation["mean_objective"] = (objective_column, "mean")
        convergence = diagnostics.groupby(["condition", "dataset"], as_index=False).agg(
            **aggregation
        )
        convergence["success_rate"] = (
            convergence["successful_fits"] / convergence["target_fits"]
        )
        convergence.to_csv(
            output_dir / "optimization_convergence_summary.tsv",
            sep="\t",
            index=False,
        )

    report = [
        "# 15-leaf hyperparameter sensitivity",
        "",
        "This analysis keeps the 15-leaf ultrametric fate map topology fixed.",
        "Optimizer-only conditions reuse the existing `lineage_signal_15leaf`",
        "simulation data. Cell-count and gene-count conditions necessarily use",
        "newly generated 15-leaf datasets with the same simulator and inference",
        "code, because those parameters are properties of the simulated RNA/ATAC",
        "observations and GRN universe.",
        "",
        f"Replicates per condition: {replicates}.",
        f"Processes per RegArch run: {processes}.",
        "",
        "Main tables:",
        "- `hyperparameter_leaf_summary.tsv`",
        "- `hyperparameter_ancestor_summary.tsv`",
        "- `hyperparameter_leaf_delta_vs_baseline.tsv`",
        "- `regarch_runtime.tsv` and `regarch_runtime_summary.tsv`",
    ]
    (output_dir / "HYPERPARAMETER_SENSITIVITY_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 15-leaf RegArch cell/gene/optimizer sensitivity."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("benchmark/configs/lineage_signal_15leaf_10rep.json"),
    )
    parser.add_argument(
        "--existing-run-root",
        type=Path,
        default=Path("benchmark/runs"),
    )
    parser.add_argument(
        "--generated-run-root",
        type=Path,
        default=Path("benchmark/runs"),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("benchmark/configs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/results/hyperparameter_15leaf_sensitivity"),
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260660)
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    all_metrics = []
    runtime_rows = []
    diagnostics = []

    for index, condition in enumerate(CONDITIONS):
        if condition.dataset == "lineage_signal_15leaf":
            replicate_dirs = _replicate_dirs(
                args.existing_run_root,
                condition.dataset,
                args.replicates,
            )
            if len(replicate_dirs) < args.replicates:
                raise RuntimeError(
                    f"Expected {args.replicates} existing 15-leaf replicates; "
                    f"found {len(replicate_dirs)}."
                )
        else:
            replicate_dirs = _ensure_generated(
                condition,
                base_config,
                args.config_dir,
                args.generated_run_root,
                args.replicates,
                args.seed + index,
            )
        for replicate_dir in replicate_dirs:
            runtime_rows.append(
                _run_condition(replicate_dir, condition, args.processes)
            )
            all_metrics.append(
                _evaluate_replicate(replicate_dir, condition, args.output_dir)
            )
            diagnostics.append(_collect_optimizer_diagnostics(replicate_dir, condition))

    metrics = pd.concat(all_metrics, ignore_index=True)
    runtimes = pd.DataFrame(runtime_rows)
    diagnostics_df = pd.concat(diagnostics, ignore_index=True)
    _write_summaries(
        args.output_dir,
        metrics,
        runtimes,
        diagnostics_df,
        args.replicates,
        args.processes,
    )


if __name__ == "__main__":
    main()
