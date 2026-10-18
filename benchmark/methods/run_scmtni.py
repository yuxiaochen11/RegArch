from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree


def _lines(path: Path) -> list[str]:
    return [
        x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()
    ]


def _prepare(replicate_dir: Path) -> Path:
    method_dir = replicate_dir / "method_inputs/scMTNI"
    work = method_dir / "prepared"
    work.mkdir(parents=True, exist_ok=True)
    leaves = _lines(replicate_dir / "truth/leaf_nodes.txt")
    genes = _lines(replicate_dir / "intermediate/metadata/target_list.txt")
    regulators = _lines(replicate_dir / "intermediate/metadata/regulator_list.txt")

    og = {gene: f"OG{i + 1}_1" for i, gene in enumerate(genes)}
    (work / "celltype_order.txt").write_text("\n".join(leaves) + "\n", encoding="utf-8")
    pd.DataFrame(
        {
            "Gene_OGID": [og[g] for g in genes],
            "NAME": [",".join(f"{g}_{leaf}" for leaf in leaves) for g in genes],
        }
    ).to_csv(work / "ogids.txt", sep="\t", index=False)
    (work / "TFs_OGs.txt").write_text(
        "\n".join(str(genes.index(g) + 1) for g in regulators) + "\n",
        encoding="utf-8",
    )
    (work / "AllGenes.txt").write_text(
        "\n".join(str(i + 1) for i in range(len(genes))) + "\n", encoding="utf-8"
    )

    config_rows = []
    for leaf in leaves:
        expression = pd.read_csv(
            method_dir / "celltype_expression" / f"{leaf}_expression.tsv",
            sep="\t",
            index_col=0,
        )
        expression = np.log1p(expression.astype(float))
        expression.index = [f"{g}_{leaf}" for g in expression.index]
        expression.index.name = "Gene"
        expression.to_csv(work / f"{leaf}.table", sep="\t")

        prior = pd.read_csv(
            method_dir / "celltype_prior_networks" / f"{leaf}_prior.tsv", sep="\t"
        )
        prior = prior.loc[
            prior["prior_score"] > 0, ["regulator", "target", "prior_score"]
        ]
        prior["regulator"] += f"_{leaf}"
        prior["target"] += f"_{leaf}"
        prior.to_csv(work / f"{leaf}_network.txt", sep="\t", index=False, header=False)
        (work / f"{leaf}_regulators.txt").write_text(
            "\n".join(f"{g}_{leaf}" for g in regulators) + "\n", encoding="utf-8"
        )
        (work / f"{leaf}_genes.txt").write_text(
            "\n".join(f"{g}_{leaf}" for g in genes) + "\n", encoding="utf-8"
        )
        config_rows.append(
            [
                leaf,
                f"/work/{leaf}",
                f"/work/results/{leaf}",
                f"/work/{leaf}_regulators.txt",
                f"/work/{leaf}_genes.txt",
                f"/work/{leaf}_network.txt",
            ]
        )
    pd.DataFrame(config_rows).to_csv(
        work / "config.txt", sep="\t", index=False, header=False
    )

    cfg = __import__("json").loads((replicate_dir / "config.json").read_text())
    gain_rate = float(cfg["grn"]["gain_rate"])
    loss_rate = float(cfg["grn"]["loss_rate"])
    # A leaf-only projection is required because scMTNI requires expression for
    # every tree node. Build a deterministic minimum-spanning tree from exact
    # leaf-to-leaf path lengths so arbitrary ultrametric trees are supported.
    edges = pd.read_csv(
        replicate_dir / "truth/fate_map_edges.tsv",
        sep="\t",
    )
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for row in edges.itertuples(index=False):
        adjacency.setdefault(row.parent, []).append(
            (row.child, float(row.branch_length))
        )
        adjacency.setdefault(row.child, []).append(
            (row.parent, float(row.branch_length))
        )

    def distance(start: str, end: str) -> float:
        stack = [(start, "", 0.0)]
        while stack:
            node, parent, total = stack.pop()
            if node == end:
                return total
            for neighbor, length in adjacency[node]:
                if neighbor != parent:
                    stack.append((neighbor, node, total + length))
        raise ValueError(f"No fate-map path between {start} and {end}.")

    distances = np.zeros((len(leaves), len(leaves)), dtype=float)
    for left in range(len(leaves)):
        for right in range(left + 1, len(leaves)):
            value = distance(leaves[left], leaves[right])
            distances[left, right] = distances[right, left] = value
    undirected = minimum_spanning_tree(distances).toarray()
    neighbors: dict[int, list[tuple[int, float]]] = {
        index: [] for index in range(len(leaves))
    }
    for left, right in zip(*np.nonzero(undirected)):
        length = float(undirected[left, right])
        neighbors[left].append((right, length))
        neighbors[right].append((left, length))
    projected = []
    stack = [(0, -1)]
    while stack:
        parent, previous = stack.pop()
        for child, length in sorted(neighbors[parent], reverse=True):
            if child == previous:
                continue
            projected.append((leaves[child], leaves[parent], length))
            stack.append((child, parent))
    tree_rows = [
        [
            child,
            parent,
            1 - math.exp(-gain_rate * length),
            1 - math.exp(-loss_rate * length),
        ]
        for child, parent, length in projected
    ]
    pd.DataFrame(tree_rows).to_csv(
        work / "leaf_projected_tree.txt", sep="\t", index=False, header=False
    )
    # The official parser does not trim carriage returns and otherwise looks
    # up a species named "T1\r", causing a null dereference on Windows inputs.
    for path in work.glob("*.txt"):
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    for path in work.glob("*.table"):
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    return work


def run_scmtni(replicate_dir: str | Path, image: str = "regarch-scmtni:latest") -> None:
    replicate_dir = Path(replicate_dir).resolve()
    work = _prepare(replicate_dir)
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{work}:/work",
            image,
            "-f",
            "/work/config.txt",
            "-x8",
            "-l",
            "/work/TFs_OGs.txt",
            "-n",
            "/work/AllGenes.txt",
            "-d",
            "/work/leaf_projected_tree.txt",
            "-m",
            "/work/ogids.txt",
            "-s",
            "/work/celltype_order.txt",
            "-p",
            "0.08",
            "-c",
            "yes",
            "-b",
            "-0.9",
            "-q",
            "2",
        ],
        check=True,
    )
    output = replicate_dir / "method_outputs/raw/scMTNI"
    output.mkdir(parents=True, exist_ok=True)
    for leaf in _lines(replicate_dir / "truth/leaf_nodes.txt"):
        files = sorted((work / "results" / leaf / "fold0").glob("var_mb_pw_k*.txt"))
        if not files:
            raise FileNotFoundError(f"scMTNI produced no network for {leaf}")
        table = pd.read_csv(files[-1], sep="\t", header=None, comment="#")
        table = table.iloc[:, :3]
        table.columns = ["regulator", "target", "weight"]
        table["regulator"] = table["regulator"].astype(str).str.removesuffix(f"_{leaf}")
        table["target"] = table["target"].astype(str).str.removesuffix(f"_{leaf}")
        table.to_csv(output / f"{leaf}.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official scMTNI in Docker.")
    parser.add_argument("replicate_dir", type=Path)
    parser.add_argument("--image", default="regarch-scmtni:latest")
    args = parser.parse_args()
    run_scmtni(args.replicate_dir, args.image)


if __name__ == "__main__":
    main()
