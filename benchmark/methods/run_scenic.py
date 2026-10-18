from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _leaf_prior(replicate_dir: Path, leaf: str) -> pd.DataFrame:
    motif = pd.read_csv(replicate_dir / "intermediate/atac/motif_hits.tsv", sep="\t")
    links = pd.read_csv(
        replicate_dir / "intermediate/atac/region_to_gene_links.tsv", sep="\t"
    )
    atac = pd.read_csv(
        replicate_dir
        / "method_inputs/SCENIC/leaf_inputs"
        / f"{leaf}_atac_region_by_cell.tsv",
        sep="\t",
        index_col=0,
    )
    accessibility = np.log1p(atac).mean(axis=1).rename("accessibility")
    motif = motif.loc[motif["motif_hit"].astype(bool)]
    links = links.loc[links["is_candidate_link"].astype(bool)]
    prior = motif.merge(links, on="region_id").merge(
        accessibility, left_on="region_id", right_index=True
    )
    motif_score = next(column for column in ["motif_score", "score"] if column in prior)
    link_score = next(
        column
        for column in ["link_score", "score_y", "candidate_score"]
        if column in prior
    )
    prior["cis_score"] = (
        prior[motif_score].clip(lower=0)
        * prior[link_score].clip(lower=0)
        * prior["accessibility"].clip(lower=0)
    )
    return prior.groupby(["regulator", "target"], as_index=False).agg(
        cis_score=("cis_score", "max")
    )


def run_scenic(replicate_dir: str | Path, seed: int = 20250612) -> None:
    replicate_dir = Path(replicate_dir)
    input_dir = replicate_dir / "method_inputs/SCENIC/leaf_inputs"
    output_dir = replicate_dir / "method_outputs/raw/SCENIC"
    output_dir.mkdir(parents=True, exist_ok=True)
    leaves = [
        line.strip()
        for line in (replicate_dir / "truth/leaf_nodes.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    for leaf in leaves:
        expression = pd.read_csv(
            input_dir / f"{leaf}_rna_gene_by_cell.tsv",
            sep="\t",
            index_col=0,
        ).T
        adjacency = pd.read_csv(
            replicate_dir / "method_outputs/raw/GRNBoost" / f"{leaf}.tsv",
            sep="\t",
        )
        prior = _leaf_prior(replicate_dir, leaf)
        result = adjacency.merge(
            prior,
            left_on=["TF", "target"],
            right_on=["regulator", "target"],
            how="inner",
        )
        result["score"] = result["importance"] * result["cis_score"]
        result["correlation"] = [
            expression[tf].corr(expression[target])
            for tf, target in zip(result["TF"], result["target"])
        ]
        result.to_csv(output_dir / f"{leaf}.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pySCENIC GRN inference with synthetic cis pruning."
    )
    parser.add_argument("replicate_dir", type=Path)
    parser.add_argument("--seed", type=int, default=20250612)
    args = parser.parse_args()
    run_scenic(args.replicate_dir, args.seed)


if __name__ == "__main__":
    main()
