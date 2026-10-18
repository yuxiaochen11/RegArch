from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from LingerGRN import LL_net


def _safe_correlation(x: pd.Series, y: pd.Series) -> float:
    value = x.corr(y)
    return float(value) if np.isfinite(value) else 0.0


def run_linger(replicate_dir: str | Path) -> None:
    replicate_dir = Path(replicate_dir)
    input_dir = replicate_dir / "method_inputs/LINGER/leaf_inputs"
    work_dir = replicate_dir / "method_outputs/raw/LINGER/work"
    output_dir = replicate_dir / "method_outputs/raw/LINGER"
    work_dir.mkdir(parents=True, exist_ok=True)
    motif = pd.read_csv(replicate_dir / "intermediate/atac/motif_hits.tsv", sep="\t")
    motif = motif.loc[motif["motif_hit"].astype(bool)]
    links = pd.read_csv(
        replicate_dir / "intermediate/atac/region_to_gene_links.tsv", sep="\t"
    )
    links = links.loc[links["is_candidate_link"].astype(bool)]
    leaves = [
        line.strip()
        for line in (replicate_dir / "truth/leaf_nodes.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    for leaf in leaves:
        rna = pd.read_csv(
            input_dir / f"{leaf}_rna_gene_by_cell.tsv",
            sep="\t",
            index_col=0,
        )
        atac = pd.read_csv(
            input_dir / f"{leaf}_atac_region_by_cell.tsv",
            sep="\t",
            index_col=0,
        )
        common_cells = rna.columns.intersection(atac.columns)
        rna = np.log1p(rna.loc[:, common_cells])
        atac = np.log1p(atac.loc[:, common_cells])

        leaf_work = work_dir / leaf
        leaf_work.mkdir(parents=True, exist_ok=True)
        prefix = str(leaf_work.resolve()).replace("\\", "/") + "/"

        binding_rows = []
        for row in motif.itertuples():
            if row.region_id not in atac.index or row.regulator not in rna.index:
                continue
            score = (
                row.motif_score
                * atac.loc[row.region_id].mean()
                * rna.loc[row.regulator].mean()
            )
            binding_rows.append((row.region_id, row.regulator, score))
        binding_long = pd.DataFrame(
            binding_rows, columns=["region", "regulator", "score"]
        )
        binding = binding_long.pivot_table(
            index="region",
            columns="regulator",
            values="score",
            aggfunc="max",
            fill_value=0,
        )
        binding.to_csv(
            leaf_work / f"cell_type_specific_TF_RE_binding_{leaf}.txt",
            sep="\t",
        )

        cis_rows = []
        for row in links.itertuples():
            if row.region_id not in atac.index or row.target not in rna.index:
                continue
            correlation = _safe_correlation(
                atac.loc[row.region_id], rna.loc[row.target]
            )
            cis_rows.append(
                (row.region_id, row.target, row.link_score * abs(correlation))
            )
        pd.DataFrame(cis_rows).to_csv(
            leaf_work / f"cell_type_specific_cis_regulatory_{leaf}.txt",
            sep="\t",
            header=False,
            index=False,
        )

        dummy = ad.AnnData(np.zeros((len(common_cells), 1)))
        dummy.obs["label"] = leaf
        LL_net.cell_type_specific_trans_reg("", dummy, leaf, prefix)
        matrix = pd.read_csv(
            leaf_work / f"cell_type_specific_trans_regulatory_{leaf}.txt",
            sep="\t",
            index_col=0,
        )
        long = (
            matrix.rename_axis("target")
            .reset_index()
            .melt(id_vars="target", var_name="TF", value_name="Score")
        )
        long.to_csv(output_dir / f"{leaf}.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LINGER baseline trans-regulatory inference."
    )
    parser.add_argument("replicate_dir", type=Path)
    args = parser.parse_args()
    run_linger(args.replicate_dir)


if __name__ == "__main__":
    main()
