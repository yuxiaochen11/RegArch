from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .evaluation import evaluate_scores
from .standardize_regarch import standardize_regarch


def calibrate(replicate_dirs: list[Path], output: Path) -> pd.DataFrame:
    rows = []
    settings = itertools.product(
        [0.05, 0.2, 0.5],
        [1.0, 4.0],
        [1.0],
        ["uniform", "atac"],
    )
    for lineage_weight, atac_weight, likelihood_weight, root_prior in settings:
        run_name = (
            f"cal_l{lineage_weight:g}_a{atac_weight:g}_"
            f"r{likelihood_weight:g}_{root_prior}"
        )
        metric_parts = []
        for replicate_dir in replicate_dirs:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmark.run_regarch",
                    str(replicate_dir),
                    "--processes",
                    "4",
                    "--run-name",
                    run_name,
                    "--lineage-model",
                    "ctmc_signed",
                    "--lineage-weight",
                    str(lineage_weight),
                    "--atac-weight",
                    str(atac_weight),
                    "--likelihood-weight",
                    str(likelihood_weight),
                    "--root-prior",
                    root_prior,
                    "--root-prior-probability",
                    "0.08",
                    "--optimizer-maxfun",
                    "60000",
                ],
                check=True,
            )
            scores = standardize_regarch(
                replicate_dir
                / "method_outputs/raw"
                / f"RegArch_{run_name}"
                / "inferred_grns",
                replicate_dir / "method_inputs/RegArch/regulator_list.txt",
                replicate_dir / "evaluation/calibration" / f"{run_name}.tsv",
                method_name="RegArch",
            )
            leaves = set((replicate_dir / "truth/leaf_nodes.txt").read_text().split())
            scores = scores.loc[scores["cell_type"].isin(leaves)]
            universe = pd.read_csv(
                replicate_dir / "intermediate/evaluation/edge_universe.tsv",
                sep="\t",
            )
            metrics = evaluate_scores(universe, scores)
            metric_parts.append(metrics.loc[metrics["cell_type"].ne("__mean__")])
        metrics = pd.concat(metric_parts, ignore_index=True)
        rows.append(
            {
                "run_name": run_name,
                "lineage_weight": lineage_weight,
                "atac_weight": atac_weight,
                "likelihood_weight": likelihood_weight,
                "root_prior": root_prior,
                "leaf_auprc_mean": metrics["auprc"].mean(),
                "leaf_auroc_mean": metrics["auroc"].mean(),
                "signed_precision_at_true_k_mean": metrics[
                    "signed_precision_at_true_k"
                ].mean(),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["leaf_auprc_mean", "leaf_auroc_mean"],
        ascending=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, sep="\t", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replicate_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(calibrate(args.replicate_dirs, args.output).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
