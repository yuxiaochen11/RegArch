from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dynamic_grn import DynamicGRNTruth
from .fate_map import FateMapTruth
from .multiome import MultiomeData

METHODS = [
    "CellOracle",
    "SCENIC",
    "SCENICplus",
    "GENIE3",
    "GRNBoost",
    "FigR",
    "Pando",
    "GRaNIE",
    "LINGER",
    "Dictys",
    "scMTNI",
]

LEAF_BENCHMARK_METHODS = METHODS

METHOD_NOTES = {
    "CellOracle": [
        "Use base_grn_linklist.tsv as the ATAC/motif-derived base GRN.",
        "Infer one context-specific network per leaf cell type.",
    ],
    "SCENIC": [
        "Build a custom synthetic cisTarget ranking resource before motif pruning.",
        "Split regulon evidence by leaf cluster for cell-type-specific evaluation.",
    ],
    "SCENICplus": [
        "Create version-compatible AnnData and cisTopic objects from shared matrices.",
        "Collapse eRegulon region-level evidence to TF-target scores.",
    ],
    "GENIE3": [
        "Run each leaf expression matrix independently with the shared TF list."
    ],
    "GRNBoost": [
        "Run each leaf expression matrix independently with the shared TF list."
    ],
    "FigR": ["Keep RNA and ATAC cell identifiers in exactly the same order."],
    "Pando": [
        "Build a Seurat/Signac object without changing synthetic feature identifiers."
    ],
    "GRaNIE": ["Use the generated metacell RNA and ATAC matrices."],
    "LINGER": [
        "Use matched-information mode without external atlas data in the primary benchmark."
    ],
    "Dictys": [
        "The primary leaf benchmark contains terminal observations only.",
        "Any trajectory-support-cell analysis must be reported separately.",
    ],
    "scMTNI": ["Use the same ATAC-derived prior supplied to RegArch."],
}


def _write_manifest(
    output_dir: Path,
    method: str,
    files: dict[str, str],
    notes: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "files": files,
        "notes": notes,
        "standard_output_columns": [
            "method",
            "cell_type",
            "regulator",
            "target",
            "score",
            "sign_optional",
        ],
    }
    (output_dir / "input_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _export_leaf_expression(
    method_dir: Path,
    multiome: MultiomeData,
    fate_map: FateMapTruth,
    orientation: str,
) -> dict[str, str]:
    metadata = multiome.cell_metadata.set_index("cell_id")
    paths = {}
    for leaf in fate_map.leaf_nodes:
        cells = metadata.index[metadata["cell_type"] == leaf]
        matrix = multiome.rna_counts.loc[:, cells]
        if orientation == "cell_by_gene":
            matrix = matrix.T
        path = method_dir / f"{leaf}_expression_{orientation}.tsv"
        matrix.to_csv(path, sep="\t")
        paths[leaf] = path.name
    return paths


def _export_granie_metacells(
    method_dir: Path,
    multiome: MultiomeData,
    fate_map: FateMapTruth,
    metacells_per_leaf: int,
) -> None:
    metadata = multiome.cell_metadata.set_index("cell_id")
    rna_parts = []
    atac_parts = []
    meta_rows = []
    for leaf in fate_map.leaf_nodes:
        cells = metadata.index[metadata["cell_type"] == leaf].to_numpy()
        groups = np.array_split(cells, min(metacells_per_leaf, len(cells)))
        for index, group in enumerate(groups, start=1):
            metacell = f"{leaf}_MC{index:02d}"
            rna_parts.append(
                multiome.rna_counts.loc[:, group].sum(axis=1).rename(metacell)
            )
            atac_parts.append(
                multiome.atac_counts.loc[:, group].sum(axis=1).rename(metacell)
            )
            meta_rows.append(
                {"metacell": metacell, "cell_type": leaf, "cell_count": len(group)}
            )
    pd.concat(rna_parts, axis=1).to_csv(method_dir / "metacell_rna.tsv", sep="\t")
    pd.concat(atac_parts, axis=1).to_csv(method_dir / "metacell_atac.tsv", sep="\t")
    pd.DataFrame(meta_rows).to_csv(
        method_dir / "metacell_metadata.tsv", sep="\t", index=False
    )


def prepare_method_inputs(
    output_root: str | Path,
    fate_map: FateMapTruth,
    grn_truth: DynamicGRNTruth,
    multiome: MultiomeData,
    regarch_prior: pd.DataFrame,
    metacells_per_leaf: int,
) -> None:
    output_root = Path(output_root)
    base = output_root / "method_inputs"
    shared = {
        "cell_metadata": "../../intermediate/metadata/cell_metadata.tsv",
        "regulator_list": "../../intermediate/metadata/regulator_list.txt",
        "target_list": "../../intermediate/metadata/target_list.txt",
        "rna_gene_by_cell": "../../intermediate/rna/rna_counts_gene_by_cell.tsv",
        "rna_cell_by_gene": "../../intermediate/rna/rna_counts_cell_by_gene.tsv",
        "atac_region_by_cell": "../../intermediate/atac/atac_counts_region_by_cell.tsv",
        "atac_cell_by_region": "../../intermediate/atac/atac_counts_cell_by_region.tsv",
        "peak_bed": "../../intermediate/atac/peak_metadata.bed",
        "motif_hits": "../../intermediate/atac/motif_hits.tsv",
        "region_to_gene": "../../intermediate/atac/region_to_gene_links.tsv",
        "fate_map": "../../truth/fate_map.nwk",
    }

    for method in METHODS:
        method_dir = base / method
        files = dict(shared)
        notes = [
            "All methods must use this replicate's shared observations and edge universe.",
            "The benchmark runner records whether execution is native end-to-end or uses a synthetic-resource adapter.",
            "Evaluate leaf-node TF-target scores only; use unsigned scores for the primary comparison.",
        ] + METHOD_NOTES[method]
        _write_manifest(method_dir, method, files, notes)

    metadata = multiome.cell_metadata.set_index("cell_id")
    for method in LEAF_BENCHMARK_METHODS:
        method_dir = base / method
        leaf_dir = method_dir / "leaf_inputs"
        leaf_dir.mkdir(parents=True, exist_ok=True)
        leaf_files = {}
        for leaf in fate_map.leaf_nodes:
            cells = metadata.index[metadata["cell_type"] == leaf]
            rna_path = leaf_dir / f"{leaf}_rna_gene_by_cell.tsv"
            atac_path = leaf_dir / f"{leaf}_atac_region_by_cell.tsv"
            multiome.rna_counts.loc[:, cells].to_csv(rna_path, sep="\t")
            multiome.atac_counts.loc[:, cells].to_csv(atac_path, sep="\t")
            leaf_files[leaf] = {
                "rna_gene_by_cell": str(rna_path.relative_to(method_dir)),
                "atac_region_by_cell": str(atac_path.relative_to(method_dir)),
            }
        manifest_path = method_dir / "input_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["leaf_inputs"] = leaf_files
        manifest["shared_input_contract"] = "../shared_input_contract.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    celloracle_dir = base / "CellOracle"
    global_prior = (
        regarch_prior.loc[regarch_prior["node"] != "Root"]
        .groupby(["regulator", "target"], as_index=False)["prior_score"]
        .max()
        .rename(columns={"prior_score": "base_score"})
    )
    global_prior["base_edge"] = (global_prior["base_score"] > 0).astype(int)
    global_prior.to_csv(celloracle_dir / "base_grn_linklist.tsv", sep="\t", index=False)

    for method in ("GENIE3", "GRNBoost"):
        method_dir = base / method
        paths = _export_leaf_expression(
            method_dir,
            multiome,
            fate_map,
            "gene_by_cell" if method == "GENIE3" else "cell_by_gene",
        )
        manifest = json.loads((method_dir / "input_manifest.json").read_text())
        manifest["leaf_expression_files"] = paths
        (method_dir / "input_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    granie_dir = base / "GRaNIE"
    _export_granie_metacells(granie_dir, multiome, fate_map, metacells_per_leaf)

    scmtni_dir = base / "scMTNI"
    expression_dir = scmtni_dir / "celltype_expression"
    prior_dir = scmtni_dir / "celltype_prior_networks"
    expression_dir.mkdir(parents=True, exist_ok=True)
    prior_dir.mkdir(parents=True, exist_ok=True)
    for leaf in fate_map.leaf_nodes:
        cells = metadata.index[metadata["cell_type"] == leaf]
        multiome.rna_counts.loc[:, cells].to_csv(
            expression_dir / f"{leaf}_expression.tsv", sep="\t"
        )
        regarch_prior.loc[regarch_prior["node"] == leaf].to_csv(
            prior_dir / f"{leaf}_prior.tsv", sep="\t", index=False
        )

    linger_dir = base / "LINGER"
    multiome.rna_counts.to_csv(linger_dir / "RNA.txt", sep="\t")
    multiome.atac_counts.to_csv(linger_dir / "ATAC.txt", sep="\t")
    multiome.cell_metadata[["cell_id", "cell_type"]].rename(
        columns={"cell_id": "barcode", "cell_type": "Annotation"}
    ).to_csv(linger_dir / "label.txt", sep="\t", index=False)
