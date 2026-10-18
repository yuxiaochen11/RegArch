from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd


def standardize_regarch(
    inferred_dir: str | Path,
    regulator_list: str | Path,
    output_path: str | Path,
    method_name: str = "RegArch",
    node_mapping: str | Path | None = None,
) -> pd.DataFrame:
    inferred_dir = Path(inferred_dir)
    regulators = [
        line.strip()
        for line in Path(regulator_list).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for path in sorted(inferred_dir.glob("target_gene_*.csv")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            values = ast.literal_eval(line)
            target, node = values[0], values[1]
            weights = values[3:]
            if len(weights) != len(regulators):
                raise ValueError(
                    f"{path} contains {len(weights)} weights; expected {len(regulators)}."
                )
            for regulator, weight in zip(regulators, weights):
                value = float(weight)
                rows.append(
                    {
                        "method": method_name,
                        "cell_type": node,
                        "regulator": regulator,
                        "target": target,
                        "score": abs(value),
                        "sign_optional": int(value > 0) - int(value < 0),
                    }
                )
    result = pd.DataFrame(rows)
    if node_mapping is not None:
        mapping = pd.read_csv(node_mapping, sep="\t")
        rename = dict(zip(mapping["inferred_node"], mapping["truth_node"]))
        result["cell_type"] = result["cell_type"].replace(rename)
        result = result.groupby(
            ["method", "cell_type", "regulator", "target"],
            as_index=False,
        ).agg({"score": "mean", "sign_optional": "first"})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, sep="\t", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize RegArch outputs.")
    parser.add_argument("--inferred-dir", type=Path, required=True)
    parser.add_argument("--regulator-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method-name", default="RegArch")
    parser.add_argument("--node-mapping", type=Path)
    args = parser.parse_args()
    standardize_regarch(
        args.inferred_dir,
        args.regulator_list,
        args.output,
        args.method_name,
        args.node_mapping,
    )


if __name__ == "__main__":
    main()
