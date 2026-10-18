from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def prepare_dictys(replicate_dir: str | Path) -> None:
    replicate_dir = Path(replicate_dir)
    method_dir = replicate_dir / "method_inputs/Dictys"
    work_dir = method_dir / "prepared"
    work_dir.mkdir(parents=True, exist_ok=True)
    motif = pd.read_csv(replicate_dir / "intermediate/atac/motif_hits.tsv", sep="\t")
    links = pd.read_csv(
        replicate_dir / "intermediate/atac/region_to_gene_links.tsv", sep="\t"
    )
    candidates = motif.loc[motif["motif_hit"].astype(bool)].merge(
        links.loc[links["is_candidate_link"].astype(bool)], on="region_id"
    )
    leaves = [
        line.strip()
        for line in (replicate_dir / "truth/leaf_nodes.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    for leaf in leaves:
        leaf_dir = work_dir / leaf
        leaf_dir.mkdir(parents=True, exist_ok=True)
        expression = pd.read_csv(
            method_dir / "leaf_inputs" / f"{leaf}_rna_gene_by_cell.tsv",
            sep="\t",
            index_col=0,
        )
        expression.to_csv(leaf_dir / "expression.tsv", sep="\t")
        genes = expression.index.astype(str)
        mask = pd.DataFrame(0, index=genes, columns=genes, dtype=int)
        for edge in candidates.itertuples():
            if (
                edge.regulator in mask.columns
                and edge.target in mask.index
                and edge.regulator != edge.target
            ):
                mask.loc[edge.target, edge.regulator] = 1
        mask.to_csv(leaf_dir / "mask.tsv", sep="\t")


def collect_dictys(replicate_dir: str | Path) -> None:
    replicate_dir = Path(replicate_dir)
    prepared = replicate_dir / "method_inputs/Dictys/prepared"
    output_dir = replicate_dir / "method_outputs/raw/Dictys"
    output_dir.mkdir(parents=True, exist_ok=True)
    regulators = {
        line.strip()
        for line in (replicate_dir / "intermediate/metadata/regulator_list.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    for leaf_dir in sorted(path for path in prepared.iterdir() if path.is_dir()):
        weights = pd.read_csv(leaf_dir / "weight.tsv", sep="\t", index_col=0)
        long = (
            weights.rename_axis("target")
            .reset_index()
            .melt(id_vars="target", var_name="regulator", value_name="weight")
        )
        long = long.loc[long["regulator"].isin(regulators) & long["weight"].ne(0)]
        long.to_csv(output_dir / f"{leaf_dir.name}.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Dictys leaf inputs.")
    parser.add_argument("replicate_dir", type=Path)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        collect_dictys(args.replicate_dir)
    else:
        prepare_dictys(args.replicate_dir)


if __name__ == "__main__":
    main()
