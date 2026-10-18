from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .compare_leaf_methods import compare_replicates
from .diagnose_outputs import diagnose_replicates
from .input_contract import verify_input_contract
from .plot_leaf_benchmark import plot_benchmark
from .run_random_baseline import run_random_baseline
from .run_rna_baselines import run_rna_baseline
from .standardize_external import standardize_method_outputs

METHODS = [
    "RegArch",
    "Random",
    "CellOracle",
    "GENIE3",
    "GRNBoost",
    "SCENIC",
    "SCENICplus",
    "FigR",
    "Pando",
    "GRaNIE",
    "LINGER",
    "Dictys",
    "scMTNI",
]

EXTERNAL_STANDARDIZATION = {
    "CellOracle",
    "SCENIC",
    "SCENICplus",
    "FigR",
    "Pando",
    "GRaNIE",
    "LINGER",
    "Dictys",
    "scMTNI",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _container_replicate(replicate_dir: Path, mount: str) -> str:
    relative = replicate_dir.resolve().relative_to(_repo_root())
    return f"{mount}/{relative.as_posix()}"


def _run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _run_regarch(
    replicate_dir: Path,
    processes: int,
    lineage_weight: float,
    optimizer: str,
    optimizer_maxfun: int,
) -> None:
    _run(
        [
            sys.executable,
            "-m",
            "benchmark.run_regarch",
            str(replicate_dir),
            "--variant",
            "full",
            "--processes",
            str(processes),
            "--lineage-model",
            "ctmc_signed",
            "--lineage-weight",
            str(lineage_weight),
            "--optimizer",
            optimizer,
            "--optimizer-maxfun",
            str(optimizer_maxfun),
        ],
        _repo_root(),
    )
    _run(
        [
            sys.executable,
            "-m",
            "benchmark.standardize_regarch",
            "--inferred-dir",
            str(replicate_dir / "method_outputs/raw/RegArch_full/inferred_grns"),
            "--regulator-list",
            str(replicate_dir / "method_inputs/RegArch/regulator_list.txt"),
            "--method-name",
            "RegArch",
            "--output",
            str(replicate_dir / "evaluation/RegArch_all_nodes.tsv"),
        ],
        _repo_root(),
    )
    all_node_output = replicate_dir / "evaluation/RegArch_all_nodes.tsv"
    output = replicate_dir / "method_outputs/standardized/RegArch.tsv"
    leaves = {
        line.strip()
        for line in (replicate_dir / "truth/leaf_nodes.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    scores = pd.read_csv(all_node_output, sep="\t")
    scores.loc[scores["cell_type"].isin(leaves)].to_csv(output, sep="\t", index=False)


def _run_method(
    method: str,
    replicate_dir: Path,
    seed: int,
    processes: int,
    lineage_weight: float,
    optimizer: str,
    optimizer_maxfun: int,
) -> None:
    repo = _repo_root()
    if method == "RegArch":
        _run_regarch(
            replicate_dir,
            processes,
            lineage_weight,
            optimizer,
            optimizer_maxfun,
        )
    elif method == "Random":
        run_random_baseline(replicate_dir, seed + 100_000)
    elif method in {"GENIE3", "GRNBoost"}:
        run_rna_baseline(replicate_dir, method, seed)
    elif method == "CellOracle":
        _run(
            [
                "docker",
                "run",
                "--rm",
                "-e",
                f"PYTHONHASHSEED={seed}",
                "-v",
                f"{repo}:/workspace",
                "-w",
                "/workspace",
                "kenjikamimoto126/celloracle_ubuntu:0.18.0",
                "python",
                "benchmark/methods/run_celloracle.py",
                _container_replicate(replicate_dir, "/workspace"),
                "--bagging-number",
                "50",
                "--n-jobs",
                "1",
                "--seed",
                str(seed),
            ]
        )
    elif method == "SCENIC":
        _run(
            [
                sys.executable,
                "-m",
                "benchmark.methods.run_scenic",
                str(replicate_dir),
                "--seed",
                str(seed),
            ],
            repo,
        )
    elif method == "SCENICplus":
        _run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{repo}:/repo",
                "regarch-scenicplus:latest",
                _container_replicate(replicate_dir, "/repo"),
                "--source",
                "/opt/scenicplus",
                "--seed",
                str(seed),
            ]
        )
    elif method in {"FigR", "Pando", "GRaNIE"}:
        # One invocation writes all three outputs. Existing outputs make the
        # subsequent two method entries no-ops.
        first_leaf = next(
            leaf.strip()
            for leaf in (replicate_dir / "truth/leaf_nodes.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if leaf.strip()
        )
        expected = replicate_dir / "method_outputs/raw" / method / f"{first_leaf}.tsv"
        if not expected.exists():
            _run(
                [
                    "Rscript",
                    "benchmark/methods/run_r_multiome_methods.R",
                    str(replicate_dir),
                    str(seed),
                ],
                repo,
            )
    elif method == "LINGER":
        _run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{repo}:/workspace",
                "-w",
                "/workspace",
                "regarch-linger:1.105",
                "python",
                "benchmark/methods/run_linger.py",
                _container_replicate(replicate_dir, "/workspace"),
            ]
        )
    elif method == "Dictys":
        _run(
            [
                sys.executable,
                "-m",
                "benchmark.methods.run_dictys",
                str(replicate_dir),
                "--seed",
                str(seed),
            ],
            repo,
        )
    elif method == "scMTNI":
        _run(
            [
                sys.executable,
                "-m",
                "benchmark.methods.run_scmtni",
                str(replicate_dir),
            ],
            repo,
        )
    else:
        raise ValueError(f"Unsupported benchmark method: {method}")


def _standardize(replicate_dir: Path, method: str) -> None:
    if method not in EXTERNAL_STANDARDIZATION:
        return
    standardize_method_outputs(
        replicate_dir,
        method,
        Path(__file__).with_name("method_specs.json"),
        replicate_dir / "method_outputs/standardized" / f"{method}.tsv",
    )


def _implementation_table() -> pd.DataFrame:
    metadata = json.loads(
        Path(__file__)
        .with_name("method_implementations.json")
        .read_text(encoding="utf-8")
    )
    return pd.DataFrame([{"method": method, **metadata[method]} for method in METHODS])


def run_replicate(
    replicate_dir: str | Path,
    processes: int = 1,
    lineage_weight: float = 0.5,
    optimizer: str = "L-BFGS-B",
    optimizer_maxfun: int = 60000,
    force: bool = False,
    continue_on_error: bool = False,
) -> dict[str, dict]:
    replicate_dir = Path(replicate_dir).resolve()
    verify_input_contract(replicate_dir)
    config = json.loads((replicate_dir / "config.json").read_text(encoding="utf-8"))
    seed = int(config["seed"]) + int(replicate_dir.name.rsplit("_", 1)[-1])
    standardized_dir = replicate_dir / "method_outputs/standardized"
    standardized_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, dict] = {}

    if force:
        (replicate_dir / "evaluation/RegArch_all_nodes.tsv").unlink(missing_ok=True)
        for method in METHODS:
            (standardized_dir / f"{method}.tsv").unlink(missing_ok=True)
            raw = replicate_dir / "method_outputs/raw" / method
            if raw.exists():
                shutil.rmtree(raw)
        lineage_raw = replicate_dir / "method_outputs/raw/RegArch_full"
        if lineage_raw.exists():
            shutil.rmtree(lineage_raw)

    for method in METHODS:
        output = standardized_dir / f"{method}.tsv"
        if output.exists() and not force:
            status[method] = {"status": "precomputed", "seconds": 0.0}
            continue

        started = time.perf_counter()
        try:
            _run_method(
                method,
                replicate_dir,
                seed,
                processes,
                lineage_weight,
                optimizer,
                optimizer_maxfun,
            )
            _standardize(replicate_dir, method)
            if not output.exists():
                raise FileNotFoundError(f"{method} produced no standardized output")
            status[method] = {
                "status": "completed",
                "seconds": round(time.perf_counter() - started, 3),
            }
        except Exception as exc:
            status[method] = {
                "status": "failed",
                "seconds": round(time.perf_counter() - started, 3),
                "error": str(exc),
            }
            if not continue_on_error:
                status_path = replicate_dir / "evaluation/method_status.json"
                status_path.parent.mkdir(parents=True, exist_ok=True)
                status_path.write_text(
                    json.dumps(status, indent=2),
                    encoding="utf-8",
                )
                break

        status_path = replicate_dir / "evaluation/method_status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    implementation = _implementation_table()
    implementation.to_csv(
        replicate_dir / "evaluation/method_implementations.tsv",
        sep="\t",
        index=False,
    )
    return status


def aggregate(
    replicate_dirs: list[str | Path],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    replicate_dirs = [Path(path).resolve() for path in replicate_dirs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, comparison = compare_replicates(replicate_dirs)
    metrics.to_csv(output_dir / "leaf_metrics.tsv", sep="\t", index=False)
    comparison.to_csv(
        output_dir / "paired_method_comparison.tsv", sep="\t", index=False
    )
    leaf = metrics.loc[metrics["cell_type"] != "__mean__"]
    summary = (
        leaf.groupby("method")
        .agg(
            auprc_mean=("auprc", "mean"),
            auprc_sd=("auprc", "std"),
            auroc_mean=("auroc", "mean"),
            precision_at_true_k_mean=("precision_at_true_k", "mean"),
            recall_at_true_k_mean=("recall_at_true_k", "mean"),
            paired_leaf_count=("cell_type", "size"),
        )
        .reset_index()
        .sort_values("auprc_mean", ascending=False)
    )
    summary.to_csv(output_dir / "benchmark_summary.tsv", sep="\t", index=False)
    random_reference = leaf.loc[leaf["method"].eq("Random")]
    random_comparisons = []
    if not random_reference.empty:
        for method in sorted(set(leaf["method"]) - {"Random"}):
            for metric in [
                "auprc",
                "auroc",
                "precision_at_true_k",
                "recall_at_true_k",
            ]:
                leaf_level = (
                    leaf.loc[
                        leaf["method"].isin([method, "Random"]),
                        ["replicate", "cell_type", "method", metric],
                    ]
                    .pivot(
                        index=["replicate", "cell_type"],
                        columns="method",
                        values=metric,
                    )
                    .dropna()
                )
                paired = leaf_level.groupby(level="replicate").mean()
                differences = (paired[method] - paired["Random"]).to_numpy()
                try:
                    p_value = (
                        wilcoxon(differences, alternative="greater").pvalue
                        if np.any(differences)
                        else 1.0
                    )
                except ValueError:
                    p_value = np.nan
                random_comparisons.append(
                    {
                        "method": method,
                        "metric": metric,
                        "paired_leaf_count": len(leaf_level),
                        "independent_replicate_count": len(differences),
                        "method_mean": paired[method].mean(),
                        "random_mean": paired["Random"].mean(),
                        "mean_difference": differences.mean(),
                        "win_rate": np.mean(differences > 0),
                        "wilcoxon_one_sided_p": p_value,
                    }
                )
        pd.DataFrame(random_comparisons).to_csv(
            output_dir / "paired_random_comparison.tsv",
            sep="\t",
            index=False,
        )
    random_metrics = random_reference
    if not random_metrics.empty:
        universes = [
            pd.read_csv(
                replicate_dir / "intermediate/evaluation/edge_universe.tsv",
                sep="\t",
            )
            for replicate_dir in replicate_dirs
        ]
        universe = pd.concat(universes, ignore_index=True)
        prevalence = float(universe["true_edge"].mean())
        pd.DataFrame(
            [
                {
                    "metric": "auprc",
                    "observed_mean": random_metrics["auprc"].mean(),
                    "observed_sd": random_metrics["auprc"].std(),
                    "theoretical_reference": prevalence,
                    "reference_interpretation": "edge prevalence; finite-sample AP can be higher",
                },
                {
                    "metric": "auroc",
                    "observed_mean": random_metrics["auroc"].mean(),
                    "observed_sd": random_metrics["auroc"].std(),
                    "theoretical_reference": 0.5,
                    "reference_interpretation": "random ranking",
                },
                {
                    "metric": "precision_at_true_k",
                    "observed_mean": random_metrics["precision_at_true_k"].mean(),
                    "observed_sd": random_metrics["precision_at_true_k"].std(),
                    "theoretical_reference": prevalence,
                    "reference_interpretation": "edge prevalence",
                },
                {
                    "metric": "recall_at_true_k",
                    "observed_mean": random_metrics["recall_at_true_k"].mean(),
                    "observed_sd": random_metrics["recall_at_true_k"].std(),
                    "theoretical_reference": prevalence,
                    "reference_interpretation": "edge prevalence when K equals true edge count",
                },
                {
                    "metric": "top10_regulator_overlap",
                    "observed_mean": random_metrics["top10_regulator_overlap"].mean(),
                    "observed_sd": random_metrics["top10_regulator_overlap"].std(),
                    "theoretical_reference": float("nan"),
                    "reference_interpretation": "not informative with only eight regulators",
                },
            ]
        ).to_csv(
            output_dir / "random_control_summary.tsv",
            sep="\t",
            index=False,
        )
    diagnose_replicates(replicate_dirs).to_csv(
        output_dir / "method_output_diagnostics.tsv", sep="\t", index=False
    )
    scenicplus_metadata = []
    for replicate_dir in replicate_dirs:
        path = replicate_dir / "method_outputs/raw/SCENICplus/run_metadata.tsv"
        if path.exists():
            table = pd.read_csv(path, sep="\t")
            table.insert(0, "replicate", replicate_dir.name)
            scenicplus_metadata.append(table)
    if scenicplus_metadata:
        pd.concat(scenicplus_metadata, ignore_index=True).to_csv(
            output_dir / "scenicplus_run_metadata.tsv",
            sep="\t",
            index=False,
        )
    implementations = _implementation_table()
    implementations.to_csv(
        output_dir / "method_implementations.tsv", sep="\t", index=False
    )
    status = {method: "completed" for method in METHODS}
    status_path = output_dir / "method_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    plot_benchmark(
        output_dir / "leaf_metrics.tsv",
        status_path,
        output_dir / "leaf_benchmark.png",
    )
    return metrics, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete leaf benchmark.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("replicate_dirs", type=Path, nargs="+")
    run_parser.add_argument("--processes", type=int, default=1)
    run_parser.add_argument("--lineage-weight", type=float, default=0.5)
    run_parser.add_argument("--optimizer", type=str, default="L-BFGS-B")
    run_parser.add_argument("--optimizer-maxfun", type=int, default=60000)
    run_parser.add_argument("--force", action="store_true")
    run_parser.add_argument("--continue-on-error", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("replicate_dirs", type=Path, nargs="+")
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "run":
        for replicate_dir in args.replicate_dirs:
            status = run_replicate(
                replicate_dir,
                args.processes,
                args.lineage_weight,
                args.optimizer,
                args.optimizer_maxfun,
                args.force,
                args.continue_on_error,
            )
            print(replicate_dir, json.dumps(status, sort_keys=True))
    else:
        aggregate(args.replicate_dirs, args.output_dir)


if __name__ == "__main__":
    main()
