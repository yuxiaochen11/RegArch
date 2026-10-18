from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _first_present(columns: pd.Index, candidates: list[str], field: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Could not find a {field} column. Tried {candidates}; "
        f"available columns are {columns.tolist()}."
    )


def standardize_table(
    table: pd.DataFrame,
    method: str,
    cell_type: str,
    spec: dict,
) -> pd.DataFrame:
    columns = spec["columns"]
    regulator = _first_present(table.columns, columns["regulator"], "regulator")
    target = _first_present(table.columns, columns["target"], "target")
    score = _first_present(table.columns, columns["score"], "score")
    sign_candidates = columns.get("sign", [])
    sign = next((item for item in sign_candidates if item in table.columns), score)

    result = pd.DataFrame(
        {
            "method": method,
            "cell_type": cell_type,
            "regulator": table[regulator].astype(str),
            "target": table[target].astype(str),
            "score": pd.to_numeric(table[score], errors="coerce").abs(),
            "sign_optional": np.sign(pd.to_numeric(table[sign], errors="coerce"))
            .fillna(0)
            .astype(int),
        }
    ).dropna(subset=["score"])
    edge_columns = ["method", "cell_type", "regulator", "target"]
    # Preserve the sign attached to the raw row that supplies the maximum
    # score when region-level evidence collapses to a TF-target edge.
    selected = result.groupby(edge_columns, sort=False)["score"].idxmax()
    return result.loc[selected].reset_index(drop=True)


def standardize_method_outputs(
    replicate_dir: str | Path,
    method: str,
    specs_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    replicate_dir = Path(replicate_dir)
    specs = json.loads(Path(specs_path).read_text(encoding="utf-8"))
    if method not in specs:
        raise ValueError(f"No output specification for method {method}.")
    spec = specs[method]
    leaves = [
        line.strip()
        for line in (replicate_dir / "truth" / "leaf_nodes.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    parts = []
    for leaf in leaves:
        pattern = spec["output_glob"].format(cell_type=leaf)
        paths = sorted(replicate_dir.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"No {method} output for leaf {leaf}; expected {pattern}."
            )
        for path in paths:
            separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
            parts.append(
                standardize_table(
                    pd.read_csv(path, sep=separator),
                    method,
                    leaf,
                    spec,
                )
            )
    result = pd.concat(parts, ignore_index=True)
    edge_columns = ["method", "cell_type", "regulator", "target"]
    selected = result.groupby(edge_columns, sort=False)["score"].idxmax()
    result = result.loc[selected].reset_index(drop=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, sep="\t", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize external GRN outputs.")
    parser.add_argument("replicate_dir", type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--specs",
        type=Path,
        default=Path(__file__).with_name("method_specs.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    standardize_method_outputs(args.replicate_dir, args.method, args.specs, args.output)


if __name__ == "__main__":
    main()
