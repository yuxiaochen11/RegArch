from __future__ import annotations

import argparse
import sys
import tempfile
import types
from pathlib import Path

import anndata
import joblib
import numpy as np
import pandas as pd


def _lines(path: Path) -> list[str]:
    return [
        x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()
    ]


def run_scenicplus(
    replicate_dir: str | Path,
    scenicplus_source: str | Path,
    seed: int = 20250612,
) -> None:
    np.random.seed(seed)
    replicate_dir = Path(replicate_dir)
    sys.path.insert(0, str(Path(scenicplus_source) / "src"))

    class Groupby:
        def __init__(self, keys):
            self.keys, self.keys_as_int = np.unique(keys, return_inverse=True)
            self.n_keys = max(self.keys_as_int) + 1
            self.indices = [
                np.flatnonzero(self.keys_as_int == index)
                for index in range(self.n_keys)
            ]

        def apply(self, function, vector, broadcast, temp_dir, n_cpu=1):
            if n_cpu > 1:
                values = joblib.Parallel(n_jobs=n_cpu, temp_folder=temp_dir)(
                    joblib.delayed(function)(vector[index]) for index in self.indices
                )
            else:
                values = [function(vector[index]) for index in self.indices]
            if broadcast:
                result = np.zeros(len(vector))
                for index, value in zip(self.indices, values):
                    result[index] = value
                return result
            return np.asarray(values)

    utils = types.ModuleType("scenicplus.utils")
    utils.Groupby = Groupby
    utils.flatten_list = lambda values: [item for value in values for item in value]

    def p_adjust_bh(p):
        p = np.asarray(p, dtype=float)
        descending = p.argsort()[::-1]
        original = descending.argsort()
        steps = float(len(p)) / np.arange(len(p), 0, -1)
        return np.minimum(1, np.minimum.accumulate(steps * p[descending]))[original]

    utils.p_adjust_bh = p_adjust_bh
    sys.modules["scenicplus.utils"] = utils
    scenicplus_class = types.ModuleType("scenicplus.scenicplus_class")
    scenicplus_class.SCENICPLUS = object
    sys.modules["scenicplus.scenicplus_class"] = scenicplus_class
    bioservices = types.ModuleType("bioservices")
    bioservices.BioMart = object
    bioservices.BioServicesError = RuntimeError
    sys.modules["bioservices"] = bioservices
    from scenicplus.grn_builder.gsea_approach import build_grn

    motif = pd.read_csv(replicate_dir / "intermediate/atac/motif_hits.tsv", sep="\t")
    links = pd.read_csv(
        replicate_dir / "intermediate/atac/region_to_gene_links.tsv", sep="\t"
    )
    motif = motif.loc[motif["motif_hit"].astype(bool)]
    links = links.loc[links["is_candidate_link"].astype(bool)]
    regions = sorted(set(links["region_id"]) | set(motif["region_id"]))
    regulators = _lines(replicate_dir / "intermediate/metadata/regulator_list.txt")
    cistrome_matrix = pd.DataFrame(False, index=regions, columns=regulators)
    for row in motif.itertuples():
        if row.regulator in cistrome_matrix.columns:
            cistrome_matrix.loc[row.region_id, row.regulator] = True
    cistromes = anndata.AnnData(cistrome_matrix.astype(np.int8))
    cistromes.var_names = regulators
    cistromes.obs_names = regions

    output = replicate_dir / "method_outputs/raw/SCENICplus"
    output.mkdir(parents=True, exist_ok=True)
    metadata_rows = []
    invalid_leaves = []
    for leaf in _lines(replicate_dir / "truth/leaf_nodes.txt"):
        expression = pd.read_csv(
            replicate_dir
            / "method_inputs/SCENICplus/leaf_inputs"
            / f"{leaf}_rna_gene_by_cell.tsv",
            sep="\t",
            index_col=0,
        ).T
        atac = pd.read_csv(
            replicate_dir
            / "method_inputs/SCENICplus/leaf_inputs"
            / f"{leaf}_atac_region_by_cell.tsv",
            sep="\t",
            index_col=0,
        ).T
        tf_rows = []
        for tf in regulators:
            for target in expression.columns:
                rho = expression[tf].corr(expression[target])
                tf_rows.append((tf, target, abs(float(rho)), float(rho)))
        tf2g = pd.DataFrame(tf_rows, columns=["TF", "target", "importance", "rho"])

        r2g_rows = []
        link_score_col = next(
            c for c in ["link_score", "candidate_score", "score"] if c in links
        )
        for row in links.itertuples():
            rho = atac[row.region_id].corr(expression[row.target])
            importance = abs(float(rho)) * max(float(getattr(row, link_score_col)), 0.0)
            r2g_rows.append(
                (
                    row.region_id,
                    row.target,
                    importance,
                    float(rho),
                    importance * float(rho),
                    importance * abs(float(rho)),
                )
            )
        r2g = pd.DataFrame(
            r2g_rows,
            columns=[
                "region",
                "target",
                "importance",
                "rho",
                "importance_x_rho",
                "importance_x_abs_rho",
            ],
        )
        r2g.index = r2g["region"]
        fallback_mode = "none"
        with tempfile.TemporaryDirectory() as temp_dir:
            eregulons = build_grn(
                tf_to_gene=tf2g,
                region_to_gene=r2g,
                cistromes=cistromes,
                is_extended=False,
                temp_dir=temp_dir,
                quantiles=(0.5, 0.75),
                top_n_regionTogenes_per_gene=(3, 5),
                rho_dichotomize_tf2g=True,
                rho_dichotomize_r2g=True,
                rho_dichotomize_eregulon=True,
                gsea_n_perm=100,
                min_target_genes=2,
                NES_thr=-np.inf,
                adj_pval_thr=1.01,
                merge_eRegulons=False,
                disable_tqdm=True,
                n_cpu=1,
            )
            if not eregulons:
                fallback_mode = "candidate_egrn"
        rows = []
        tf_importance = tf2g.set_index(["TF", "target"])["importance"]
        if fallback_mode == "none":
            for regulon in eregulons:
                leading_edge = regulon.in_leading_edge
                if leading_edge is None:
                    leading_edge = [True] * len(regulon.regions2genes)
                for relation, leading in zip(regulon.regions2genes, leading_edge):
                    if leading:
                        rows.append(
                            {
                                "regulator": regulon.transcription_factor,
                                "target": relation.target,
                                "score": abs(float(relation.importance))
                                * float(
                                    tf_importance.get(
                                        (
                                            regulon.transcription_factor,
                                            relation.target,
                                        ),
                                        0.0,
                                    )
                                ),
                                "correlation": float(relation.rho),
                            }
                        )
        else:
            motif_scores = motif[["region_id", "regulator", "motif_score"]].rename(
                columns={"region_id": "region"}
            )
            fallback = motif_scores.merge(
                r2g.reset_index(drop=True),
                on="region",
                how="inner",
            ).merge(
                tf2g[["TF", "target", "importance", "rho"]].rename(
                    columns={
                        "TF": "regulator",
                        "importance": "tf_gene_importance",
                        "rho": "tf_gene_rho",
                    }
                ),
                on=["regulator", "target"],
                how="inner",
            )
            fallback["score"] = (
                fallback["motif_score"].clip(lower=0)
                * fallback["importance"].clip(lower=0)
                * fallback["tf_gene_importance"].clip(lower=0)
            )
            fallback["correlation"] = fallback["rho"] * fallback["tf_gene_rho"]
            rows = fallback[["regulator", "target", "score", "correlation"]].to_dict(
                "records"
            )
        pd.DataFrame(
            rows, columns=["regulator", "target", "score", "correlation"]
        ).to_csv(output / f"{leaf}.tsv", sep="\t", index=False)
        unique_targets = len({row["target"] for row in rows})
        minimum_targets = max(2, int(np.ceil(0.25 * expression.shape[1])))
        if fallback_mode != "none" and unique_targets < minimum_targets:
            invalid_leaves.append(
                f"{leaf}: {unique_targets} targets; required {minimum_targets}"
            )
        metadata_rows.append(
            {
                "cell_type": leaf,
                "tf_gene_candidates": len(tf2g),
                "region_gene_candidates": len(r2g),
                "cistrome_hits": int(cistrome_matrix.to_numpy().sum()),
                "eregulon_count": len(eregulons),
                "reported_tf_target_edges": len(rows),
                "reported_unique_targets": unique_targets,
                "minimum_valid_targets": minimum_targets,
                "fallback_mode": fallback_mode,
                "used_small_network_fallback": fallback_mode != "none",
            }
        )
    pd.DataFrame(metadata_rows).to_csv(
        output / "run_metadata.tsv", sep="\t", index=False
    )
    if invalid_leaves:
        raise RuntimeError(
            "SCENIC+ fallback output is too degenerate for evaluation: "
            + "; ".join(invalid_leaves)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the official SCENIC+ eGRN builder."
    )
    parser.add_argument("replicate_dir", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20250612)
    args = parser.parse_args()
    run_scenicplus(args.replicate_dir, args.source, args.seed)


if __name__ == "__main__":
    main()
