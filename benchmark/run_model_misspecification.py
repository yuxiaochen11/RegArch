from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .ablation import make_wrong_fate_map, map_ancestor_nodes_by_descendants
from .evaluation import build_edge_universe, evaluate_scores
from .fate_map import FateMapTruth
from .standardize_regarch import standardize_regarch


@dataclass(frozen=True)
class MisspecificationCondition:
    name: str
    factor: str
    multiplier: float = 1.0
    topology: str = "correct"


STUDY_SCHEMA_VERSION = 2
METRIC_COLUMNS = ["auprc", "auroc", "signed_precision_at_true_k"]


def resolve_replicate_dirs(
    explicit: list[Path],
    runs_dir: Path | None = None,
) -> list[Path]:
    """Resolve explicit replicates plus config-bearing children of a runs directory."""
    candidates = list(explicit)
    if runs_dir is not None:
        if not runs_dir.is_dir():
            raise ValueError(f"runs directory does not exist: {runs_dir}")
        candidates.extend(
            path
            for path in sorted(runs_dir.iterdir())
            if path.is_dir() and (path / "config.json").is_file()
        )
    unique: list[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        if not (resolved / "config.json").is_file():
            raise ValueError(f"replicate directory lacks config.json: {candidate}")
        seen.add(resolved)
        unique.append(resolved)
    if not unique:
        raise ValueError("provide replicate directories or --runs-dir")
    return unique


def model_misspecification_conditions(
    rate_multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> list[MisspecificationCondition]:
    """Return the one-factor-at-a-time model-misspecification grid."""
    conditions = [MisspecificationCondition("matched", "matched")]
    for factor in ("gain_rate", "loss_rate", "sign_switch_rate"):
        for multiplier in rate_multipliers:
            if multiplier != 1.0:
                conditions.append(
                    MisspecificationCondition(
                        f"{factor}_x{multiplier:g}", factor, multiplier
                    )
                )
    for factor in ("positive_gain_odds", "root_prior_odds"):
        for multiplier in (0.5, 2.0):
            conditions.append(
                MisspecificationCondition(
                    f"{factor}_x{multiplier:g}", factor, multiplier
                )
            )
    for factor in ("branch_length", "terminal_temperature"):
        for multiplier in (0.5, 2.0):
            conditions.append(
                MisspecificationCondition(
                    f"{factor}_x{multiplier:g}", factor, multiplier
                )
            )
    conditions.append(
        MisspecificationCondition("topology_perturbed", "topology", topology="wrong")
    )
    return conditions


def odds_scaled_probability(probability: float, multiplier: float) -> float:
    """Return logistic(logit(probability) + log(multiplier))."""
    if not 0 < probability < 1:
        raise ValueError("probability must lie strictly between 0 and 1")
    if multiplier <= 0:
        raise ValueError("odds multiplier must be positive")
    numerator = multiplier * probability
    return numerator / (1.0 - probability + numerator)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_fingerprint(
    replicate_dir: Path,
    condition: MisspecificationCondition,
    optimizer_maxfun: int,
) -> dict:
    inputs = replicate_dir / "method_inputs/RegArch"
    source_paths = {
        "config": replicate_dir / "config.json",
        "expression": inputs / "expression_data.csv",
        "atac_prior": inputs / "atac_prior_tf_target.tsv",
        "fate_map": inputs / "fate_map_edges.csv",
        "truth": replicate_dir / "truth/true_grn_all_nodes.tsv",
    }
    if condition.topology == "wrong":
        source_paths.update(
            {
                "wrong_fate_map": replicate_dir
                / "ablations/model_misspecification_inputs"
                / "topology_perturbed_fate_map_edges.csv",
                "wrong_mapping": replicate_dir
                / "ablations/model_misspecification_inputs"
                / "topology_perturbed_to_truth_mapping.tsv",
            }
        )
    return {
        "study_schema_version": STUDY_SCHEMA_VERSION,
        "condition": asdict(condition),
        "optimizer_maxfun": optimizer_maxfun,
        "input_sha256": {name: _sha256(path) for name, path in source_paths.items()},
    }


def _cache_is_valid(
    run_dir: Path,
    target_count: int,
    expected_fingerprint: dict,
) -> bool:
    inferred_count = len(list((run_dir / "inferred_grns").glob("target_gene_*.csv")))
    diagnostic_count = len(
        list((run_dir / "optimization_diagnostics").glob("target_gene_*.json"))
    )
    manifest_path = run_dir / "model_misspecification_cache.json"
    if (
        inferred_count != target_count
        or diagnostic_count != target_count
        or not manifest_path.exists()
    ):
        return False
    try:
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return cached == expected_fingerprint


def _scaled_fate_map(replicate_dir: Path, multiplier: float) -> Path:
    source = replicate_dir / "method_inputs/RegArch/fate_map_edges.csv"
    table = pd.read_csv(source)
    if "Length" not in table:
        raise ValueError(f"{source} must contain a Length column")
    table["Length"] = table["Length"].astype(float) * multiplier
    output = (
        replicate_dir
        / "ablations/model_misspecification_inputs"
        / f"fate_map_branch_length_x{multiplier:g}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    return output


def _topology_only_fate_map(replicate_dir: Path) -> tuple[Path, Path]:
    """Build a wrong topology without the short-branch negative control."""
    source = replicate_dir / "method_inputs/RegArch/fate_map_edges.csv"
    edges = pd.read_csv(source).rename(
        columns={"Parent": "parent", "Child": "child", "Length": "branch_length"}
    )
    nodes = pd.read_csv(replicate_dir / "truth/node_metadata.tsv", sep="\t")
    truth_map = FateMapTruth(edges=edges, nodes=nodes, newick="")
    wrong_map = make_wrong_fate_map(truth_map, mode="cross_clade_split")
    output_dir = replicate_dir / "ablations/model_misspecification_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    fate_path = output_dir / "topology_perturbed_fate_map_edges.csv"
    wrong_map.edges[["parent", "child", "branch_length"]].rename(
        columns={"parent": "Parent", "child": "Child", "branch_length": "Length"}
    ).to_csv(fate_path, index=False)
    mapping_path = output_dir / "topology_perturbed_to_truth_mapping.tsv"
    map_ancestor_nodes_by_descendants(wrong_map, truth_map).to_csv(
        mapping_path, sep="\t", index=False
    )
    return fate_path, mapping_path


def build_inference_command(
    replicate_dir: Path,
    condition: MisspecificationCondition,
    processes: int,
    optimizer_maxfun: int,
) -> tuple[list[str], Path | None]:
    config = json.loads((replicate_dir / "config.json").read_text(encoding="utf-8"))
    grn = config["grn"]
    command = [
        sys.executable,
        "-m",
        "benchmark.run_regarch",
        str(replicate_dir),
        "--variant",
        "full",
        "--run-name",
        f"misspec_{condition.name}",
        "--processes",
        str(processes),
        "--lineage-model",
        "ctmc_signed",
        "--root-prior",
        "atac",
        "--optimizer-maxfun",
        str(optimizer_maxfun),
    ]
    fate_map = None
    option_by_factor = {
        "gain_rate": ("--gain-rate", float(grn["gain_rate"])),
        "loss_rate": ("--loss-rate", float(grn["loss_rate"])),
        "sign_switch_rate": (
            "--sign-switch-rate",
            float(grn["sign_switch_rate"]),
        ),
    }
    if condition.factor in option_by_factor:
        option, true_value = option_by_factor[condition.factor]
        value = true_value * condition.multiplier
        command.extend([option, f"{value:.12g}"])
    elif condition.factor == "positive_gain_odds":
        value = odds_scaled_probability(
            float(grn["positive_fraction"]), condition.multiplier
        )
        command.extend(["--positive-fraction", f"{value:.12g}"])
    elif condition.factor == "root_prior_odds":
        command.extend(["--root-prior-odds-multiplier", f"{condition.multiplier:.12g}"])
    elif condition.factor == "branch_length":
        fate_map = _scaled_fate_map(replicate_dir, condition.multiplier)
        command.extend(["--fate-map-file", str(fate_map)])
    elif condition.factor == "terminal_temperature":
        command.extend(
            [
                "--edge-temperature",
                f"{0.03 * condition.multiplier:.12g}",
                "--sign-temperature",
                f"{0.05 * condition.multiplier:.12g}",
            ]
        )
    elif condition.topology == "wrong":
        fate_map, _ = _topology_only_fate_map(replicate_dir)
        command.extend(["--fate-map-file", str(fate_map)])
    return command, fate_map


def _evaluate(
    replicate_dir: Path, condition: MisspecificationCondition
) -> pd.DataFrame:
    run_dir = replicate_dir / f"method_outputs/raw/RegArch_misspec_{condition.name}"
    output = run_dir / "standardized.tsv"
    mapping = None
    if condition.topology == "wrong":
        mapping = (
            replicate_dir
            / "ablations/model_misspecification_inputs"
            / "topology_perturbed_to_truth_mapping.tsv"
        )
    scores = standardize_regarch(
        run_dir / "inferred_grns",
        replicate_dir / "method_inputs/RegArch/regulator_list.txt",
        output,
        method_name=condition.name,
        node_mapping=mapping,
    )
    truth = pd.read_csv(replicate_dir / "truth/true_grn_all_nodes.tsv", sep="\t")
    metrics = evaluate_scores(build_edge_universe(truth), scores)
    metrics = metrics.loc[metrics["cell_type"].ne("__mean__")].copy()
    metrics.insert(0, "condition", condition.name)
    metrics.insert(1, "factor", condition.factor)
    metrics.insert(2, "multiplier", condition.multiplier)
    metrics.insert(3, "replicate", replicate_dir.name)
    node_metadata = pd.read_csv(replicate_dir / "truth/node_metadata.tsv", sep="\t")
    metrics = metrics.merge(
        node_metadata[["node_id", "node_type"]],
        left_on="cell_type",
        right_on="node_id",
        how="left",
        validate="many_to_one",
    ).drop(columns="node_id")
    return metrics


def write_study_tables(
    metrics: pd.DataFrame,
    conditions: list[MisspecificationCondition],
    parameter_rows: list[dict] | pd.DataFrame,
    output_dir: Path,
    convergence_rows: list[dict] | pd.DataFrame | None = None,
) -> None:
    """Write node-, replicate-, and paired summaries in the shared schema."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        output_dir / "model_misspecification_metrics.tsv", sep="\t", index=False
    )
    replicate_metrics = metrics.groupby(
        ["replicate", "condition", "factor", "multiplier", "node_type"],
        as_index=False,
    )[METRIC_COLUMNS].mean()
    replicate_metrics.to_csv(
        output_dir / "model_misspecification_replicate_metrics.tsv",
        sep="\t",
        index=False,
    )
    baseline_replicates = replicate_metrics.loc[
        replicate_metrics["condition"].eq("matched"),
        ["replicate", "node_type", *METRIC_COLUMNS],
    ].rename(columns={metric: f"matched_{metric}" for metric in METRIC_COLUMNS})
    paired = replicate_metrics.merge(
        baseline_replicates,
        on=["replicate", "node_type"],
        how="left",
        validate="many_to_one",
    )
    if paired[[f"matched_{metric}" for metric in METRIC_COLUMNS]].isna().any().any():
        raise ValueError("every condition must have a replicate-matched baseline")
    for metric in METRIC_COLUMNS:
        paired[f"delta_{metric}_vs_matched"] = (
            paired[metric] - paired[f"matched_{metric}"]
        )
    paired.to_csv(
        output_dir / "model_misspecification_paired_deltas.tsv",
        sep="\t",
        index=False,
    )

    aggregations = {}
    for metric in METRIC_COLUMNS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_sd"] = (metric, "std")
        delta = f"delta_{metric}_vs_matched"
        aggregations[f"{delta}_mean"] = (delta, "mean")
        aggregations[f"{delta}_sd"] = (delta, "std")
    summary = paired.groupby(
        ["condition", "factor", "multiplier", "node_type"], as_index=False
    ).agg(**aggregations)
    # Backward-compatible aliases used by older report templates.
    summary["delta_auprc_vs_matched"] = summary["delta_auprc_vs_matched_mean"]
    summary["delta_auroc_vs_matched"] = summary["delta_auroc_vs_matched_mean"]
    summary.to_csv(
        output_dir / "model_misspecification_summary.tsv", sep="\t", index=False
    )
    pd.DataFrame(asdict(condition) for condition in conditions).to_csv(
        output_dir / "model_misspecification_conditions.tsv", sep="\t", index=False
    )
    pd.DataFrame(parameter_rows).to_csv(
        output_dir / "model_misspecification_inference_parameters.tsv",
        sep="\t",
        index=False,
    )
    if convergence_rows is not None:
        convergence = pd.DataFrame(convergence_rows)
        convergence["success_rate"] = convergence["successful_fits"] / convergence[
            "target_fits"
        ].replace(0, float("nan"))
        convergence.to_csv(
            output_dir / "model_misspecification_convergence.tsv",
            sep="\t",
            index=False,
        )


def run_study(
    replicate_dirs: list[Path],
    output_dir: Path,
    processes: int = 1,
    optimizer_maxfun: int = 60000,
    force: bool = False,
) -> None:
    conditions = model_misspecification_conditions()
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_parts = []
    convergence_rows = []
    parameter_rows = []
    for replicate_dir in replicate_dirs:
        target_count = int(
            json.loads((replicate_dir / "config.json").read_text())["grn"][
                "target_count"
            ]
        )
        for condition in conditions:
            run_dir = (
                replicate_dir / f"method_outputs/raw/RegArch_misspec_{condition.name}"
            )
            command, _ = build_inference_command(
                replicate_dir, condition, processes, optimizer_maxfun
            )
            fingerprint = _cache_fingerprint(
                replicate_dir,
                condition,
                optimizer_maxfun,
            )
            if force or not _cache_is_valid(run_dir, target_count, fingerprint):
                print(f"[{replicate_dir.name}] running {condition.name}", flush=True)
                subprocess.run(command, check=True)
                (run_dir / "model_misspecification_cache.json").write_text(
                    json.dumps(fingerprint, indent=2),
                    encoding="utf-8",
                )
            else:
                print(f"[{replicate_dir.name}] reusing {condition.name}", flush=True)
            metric_parts.append(_evaluate(replicate_dir, condition))
            diagnostics_dir = run_dir / "optimization_diagnostics"
            diagnostic_files = list(diagnostics_dir.glob("target_gene_*.json"))
            convergence_rows.append(
                {
                    "replicate": replicate_dir.name,
                    "condition": condition.name,
                    "target_fits": len(diagnostic_files),
                    "successful_fits": sum(
                        bool(json.loads(path.read_text(encoding="utf-8"))["success"])
                        for path in diagnostic_files
                    ),
                }
            )
            run_config = json.loads(
                (run_dir / "run_config.json").read_text(encoding="utf-8")
            )
            parameter_rows.append(
                {
                    "replicate": replicate_dir.name,
                    "condition": condition.name,
                    "factor": condition.factor,
                    "multiplier": condition.multiplier,
                    **{
                        key: run_config[key]
                        for key in (
                            "lineage_gain_rate",
                            "lineage_loss_rate",
                            "lineage_sign_switch_rate",
                            "lineage_positive_fraction",
                            "lineage_edge_temperature",
                            "lineage_sign_temperature",
                            "root_prior",
                            "root_prior_probability",
                            "root_prior_odds_multiplier",
                            "root_prior_mean_before",
                            "root_prior_mean_after",
                            "fate_map_file",
                        )
                    },
                }
            )

    write_study_tables(
        pd.concat(metric_parts, ignore_index=True),
        conditions,
        parameter_rows,
        output_dir,
        convergence_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RegArch CTMC model-misspecification sensitivity."
    )
    parser.add_argument("replicate_dirs", type=Path, nargs="*")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help="Discover every config-bearing replicate directory under this path.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument(
        "--optimizer-maxfun",
        type=int,
        default=180000,
        help="Maximum objective evaluations per target (default: 180000).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        replicate_dirs = resolve_replicate_dirs(args.replicate_dirs, args.runs_dir)
    except ValueError as error:
        parser.error(str(error))
    run_study(
        replicate_dirs,
        args.output_dir,
        args.processes,
        args.optimizer_maxfun,
        args.force,
    )


if __name__ == "__main__":
    main()
