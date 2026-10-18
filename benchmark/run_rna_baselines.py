from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .input_contract import verify_input_contract

ALGORITHMS = {"GENIE3": "RF", "GRNBoost": "GBM"}


def run_rna_baseline(
    replicate_dir: str | Path,
    method: str,
    seed: int = 20250611,
) -> pd.DataFrame:
    if method not in ALGORITHMS:
        raise ValueError(f"Unsupported RNA baseline: {method}")
    try:
        from arboreto.algo import RF_KWARGS, SGBM_KWARGS
        from arboreto.core import infer_partial_network
    except ImportError as exc:
        raise ImportError(
            "RNA baselines require the benchmark extra: "
            "python -m pip install -e '.[benchmark]'"
        ) from exc
    algorithm_settings = {
        "GENIE3": ("RF", RF_KWARGS),
        "GRNBoost": ("GBM", SGBM_KWARGS),
    }
    replicate_dir = Path(replicate_dir)
    verify_input_contract(replicate_dir)
    regulators = [
        line.strip()
        for line in (replicate_dir / "intermediate/metadata/regulator_list.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    leaves = [
        line.strip()
        for line in (replicate_dir / "truth/leaf_nodes.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    metadata = pd.read_csv(
        replicate_dir / "intermediate/metadata/cell_metadata.tsv",
        sep="\t",
    ).set_index("cell_id")
    expression = pd.read_csv(
        replicate_dir / "intermediate/rna/rna_lognorm_cell_by_gene.tsv",
        sep="\t",
        index_col=0,
    )
    output_dir = replicate_dir / "method_outputs/raw" / method
    output_dir.mkdir(parents=True, exist_ok=True)
    standardized = []
    for leaf in leaves:
        cells = metadata.index[metadata["cell_type"] == leaf]
        matrix = expression.loc[cells]
        regulator_matrix = matrix.loc[:, regulators].to_numpy()
        network_parts = []
        regressor_type, regressor_kwargs = algorithm_settings[method]
        for target in matrix.columns:
            network_parts.append(
                infer_partial_network(
                    regressor_type=regressor_type,
                    regressor_kwargs=regressor_kwargs,
                    tf_matrix=regulator_matrix,
                    tf_matrix_gene_names=regulators,
                    target_gene_name=target,
                    target_gene_expression=matrix[target].to_numpy(),
                    seed=seed,
                )
            )
        network = pd.concat(network_parts, ignore_index=True)
        network.to_csv(output_dir / f"{leaf}.tsv", sep="\t", index=False)
        part = network.rename(
            columns={"TF": "regulator", "target": "target", "importance": "score"}
        )[["regulator", "target", "score"]]
        part.insert(0, "cell_type", leaf)
        part.insert(0, "method", method)
        part["sign_optional"] = 0
        standardized.append(part)
    result = pd.concat(standardized, ignore_index=True)
    standardized_path = replicate_dir / "method_outputs/standardized" / f"{method}.tsv"
    standardized_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(standardized_path, sep="\t", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GENIE3 or GRNBoost2.")
    parser.add_argument("replicate_dir", type=Path)
    parser.add_argument("--method", choices=sorted(ALGORITHMS), required=True)
    parser.add_argument("--seed", type=int, default=20250611)
    args = parser.parse_args()
    run_rna_baseline(args.replicate_dir, args.method, args.seed)


if __name__ == "__main__":
    main()
