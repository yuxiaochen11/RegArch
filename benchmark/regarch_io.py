from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .dynamic_grn import DynamicGRNTruth
from .fate_map import FateMapTruth
from .multiome import MultiomeData
from .regions import RegionTruth


def derive_atac_prior(
    multiome: MultiomeData,
    region_truth: RegionTruth,
    fate_map: FateMapTruth,
) -> pd.DataFrame:
    cell_types = multiome.cell_metadata.set_index("cell_id")["cell_type"]
    accessibility_parts = []
    for leaf in fate_map.leaf_nodes:
        cells = cell_types.index[cell_types == leaf]
        mean_accessibility = (
            multiome.atac_binary.loc[:, cells].mean(axis=1).rename("accessibility")
        )
        part = mean_accessibility.reset_index().rename(columns={"index": "region_id"})
        part.insert(0, "node", leaf)
        accessibility_parts.append(part)
    accessibility = pd.concat(accessibility_parts, ignore_index=True)

    motif = region_truth.motif_links.loc[
        region_truth.motif_links["motif_hit"] == 1,
        ["region_id", "regulator", "motif_score"],
    ]
    links = region_truth.region_to_gene.loc[
        region_truth.region_to_gene["is_candidate_link"] == 1,
        ["region_id", "target", "link_score"],
    ]
    joined = accessibility.merge(motif, on="region_id").merge(links, on="region_id")
    joined["raw_score"] = (
        joined["accessibility"] * joined["motif_score"] * joined["link_score"]
    )
    prior = (
        joined.groupby(["node", "regulator", "target"], as_index=False)["raw_score"]
        .max()
        .rename(columns={"raw_score": "prior_score"})
    )
    if prior.empty:
        raise RuntimeError(
            "No ATAC prior links were derived from simulated observations."
        )
    prior["prior_score"] = prior.groupby("node")["prior_score"].transform(
        lambda values: values / values.max() if values.max() > 0 else values
    )
    return prior


def complete_prior_universe(
    prior: pd.DataFrame,
    nodes: list[str],
    regulators: list[str],
    targets: list[str],
) -> pd.DataFrame:
    universe = pd.MultiIndex.from_product(
        [nodes, regulators, targets],
        names=["node", "regulator", "target"],
    ).to_frame(index=False)
    return universe.merge(prior, how="left").fillna({"prior_score": 0.0})


def export_regarch_inputs(
    output_root: str | Path,
    fate_map: FateMapTruth,
    grn_truth: DynamicGRNTruth,
    multiome: MultiomeData,
    region_truth: RegionTruth,
) -> pd.DataFrame:
    output_root = Path(output_root)
    output_dir = output_root / "method_inputs" / "RegArch"
    output_dir.mkdir(parents=True, exist_ok=True)

    prior = derive_atac_prior(multiome, region_truth, fate_map)
    prior = complete_prior_universe(
        prior,
        fate_map.leaf_nodes,
        grn_truth.regulators,
        grn_truth.targets,
    )
    root_prior = (
        prior.groupby(["regulator", "target"], as_index=False)["prior_score"]
        .mean()
        .assign(node="Root")
    )
    full_prior = pd.concat([prior, root_prior], ignore_index=True)
    full_prior.to_csv(output_dir / "atac_prior_tf_target.tsv", sep="\t", index=False)

    with (output_dir / "atac_data.csv").open("w", encoding="utf-8") as handle:
        for row in full_prior.itertuples(index=False):
            handle.write(
                f"{row.regulator},{row.target},{float(row.prior_score):.8g},{row.node}\n"
            )

    metadata = multiome.cell_metadata.set_index("cell_id")
    with (output_dir / "expression_data.csv").open("w", encoding="utf-8") as handle:
        for node in fate_map.leaf_nodes:
            cells = metadata.index[metadata["cell_type"] == node]
            for gene in grn_truth.targets:
                gene_type_rows = ["target_gene"]
                if gene in set(grn_truth.regulators):
                    gene_type_rows.append("regulator_gene")
                for gene_type in gene_type_rows:
                    for cell_id, value in multiome.rna_counts.loc[gene, cells].items():
                        handle.write(
                            f"{gene},{cell_id},{int(value)},{node},{gene_type}\n"
                        )

    fate_map.edges[["parent", "child", "branch_length"]].rename(
        columns={"parent": "Parent", "child": "Child", "branch_length": "Length"}
    ).to_csv(output_dir / "fate_map_edges.csv", index=False)
    (output_dir / "fate_map.nwk").write_text(fate_map.newick + "\n", encoding="utf-8")
    (output_dir / "regulator_list.txt").write_text(
        "\n".join(grn_truth.regulators) + "\n", encoding="utf-8"
    )
    (output_dir / "target_list.txt").write_text(
        "\n".join(grn_truth.targets) + "\n", encoding="utf-8"
    )
    return full_prior


def corrupt_prior(
    prior: pd.DataFrame,
    truth_leaf_edges: pd.DataFrame,
    coverage: float,
    correctness: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if not 0 <= coverage <= 1 or not 0 < correctness <= 1:
        raise ValueError("coverage must be in [0, 1] and correctness in (0, 1].")
    result = prior.copy()
    truth = truth_leaf_edges[["node", "regulator", "target", "true_edge"]]
    result = result.merge(truth, how="left").fillna({"true_edge": 0})
    positive_mask = result["true_edge"].eq(1) & result["prior_score"].gt(0)
    keep = rng.random(len(result)) < coverage
    result.loc[positive_mask & ~keep, "prior_score"] = 0.0

    retained_true = int((positive_mask & keep).sum())
    desired_total = int(round(retained_true / correctness))
    false_needed = max(0, desired_total - retained_true)
    false_candidates = result.index[
        result["true_edge"].eq(0) & result["prior_score"].eq(0)
    ].to_numpy()
    if false_needed and len(false_candidates):
        selected = rng.choice(
            false_candidates,
            size=min(false_needed, len(false_candidates)),
            replace=False,
        )
        result.loc[selected, "prior_score"] = rng.uniform(0.25, 1.0, size=len(selected))
    return result.drop(columns="true_edge")
