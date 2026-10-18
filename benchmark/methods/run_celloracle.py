from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from celloracle.network.net_core import Net


def _candidate_regulators(base_grn: pd.DataFrame) -> dict[str, np.ndarray]:
    candidates = base_grn.loc[base_grn["base_edge"].astype(bool)].copy()
    return {
        target: group["regulator"].astype(str).unique()
        for target, group in candidates.groupby("target", sort=False)
    }


def run_celloracle(
    replicate_dir: str | Path,
    bagging_number: int = 50,
    n_jobs: int = 1,
    seed: int = 20250612,
) -> None:
    np.random.seed(seed)
    replicate_dir = Path(replicate_dir)
    input_dir = replicate_dir / "method_inputs" / "CellOracle"
    output_dir = replicate_dir / "method_outputs" / "raw" / "CellOracle"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_grn = pd.read_csv(input_dir / "base_grn_linklist.tsv", sep="\t")
    tf_dict = _candidate_regulators(base_grn)
    leaves = [
        line.strip()
        for line in (replicate_dir / "truth" / "leaf_nodes.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    for leaf in leaves:
        expression = pd.read_csv(
            input_dir / "leaf_inputs" / f"{leaf}_rna_gene_by_cell.tsv",
            sep="\t",
            index_col=0,
        ).T
        expression = expression.loc[:, expression.var(axis=0) > 0]
        leaf_tf_dict = {
            target: regulators[np.isin(regulators, expression.columns)]
            for target, regulators in tf_dict.items()
            if target in expression.columns
        }
        leaf_tf_dict = {
            target: regulators
            for target, regulators in leaf_tf_dict.items()
            if len(regulators)
        }

        network = Net(
            gene_expression_matrix=expression,
            TFinfo_dic=leaf_tf_dict,
            verbose=False,
        )
        network.fit_genes(
            target_genes=list(leaf_tf_dict),
            bagging_number=bagging_number,
            scaling=True,
            model_method="bagging_ridge",
            save_coefs=False,
            command_line_mode=False,
            alpha=1,
            verbose=False,
            n_jobs=n_jobs,
        )
        network.updateLinkList(verbose=False)
        links = network.linkList.copy()
        links.to_csv(output_dir / f"{leaf}.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CellOracle on benchmark leaves.")
    parser.add_argument("replicate_dir", type=Path)
    parser.add_argument("--bagging-number", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20250612)
    args = parser.parse_args()
    run_celloracle(
        args.replicate_dir,
        args.bagging_number,
        args.n_jobs,
        args.seed,
    )


if __name__ == "__main__":
    main()
