from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_NODE_TIMES = {
    "Root": 0.0,
    "P1": 0.2,
    "P2": 0.8,
    "T1": 1.0,
    "T2": 1.0,
    "T3": 1.0,
    "T4": 1.0,
}


def retime_fate_map(
    topology_edges: pd.DataFrame,
    node_times: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {"parent", "child"}
    missing = required_columns - set(topology_edges.columns)
    if missing:
        raise ValueError(f"Fate map is missing columns: {sorted(missing)}")

    nodes = set(topology_edges["parent"]) | set(topology_edges["child"])
    if nodes != set(node_times):
        missing_times = sorted(nodes - set(node_times))
        extra_times = sorted(set(node_times) - nodes)
        raise ValueError(
            f"Node-time mismatch; missing={missing_times}, extra={extra_times}."
        )
    if node_times.get("Root") != 0:
        raise ValueError("Root developmental time must be 0.")

    parent_of = dict(zip(topology_edges["child"], topology_edges["parent"]))
    topological_depth = {"Root": 0}
    unresolved = nodes - {"Root"}
    while unresolved:
        progressed = False
        for node in list(unresolved):
            parent = parent_of.get(node)
            if parent in topological_depth:
                topological_depth[node] = topological_depth[parent] + 1
                unresolved.remove(node)
                progressed = True
        if not progressed:
            raise ValueError("Fate map must be a connected rooted tree.")

    edge_rows = []
    for row in topology_edges.itertuples(index=False):
        parent_time = float(node_times[row.parent])
        child_time = float(node_times[row.child])
        branch_length = round(child_time - parent_time, 12)
        if branch_length <= 0:
            raise ValueError(
                f"Non-positive branch {row.parent}->{row.child}: {branch_length}."
            )
        edge_rows.append(
            {
                "parent": row.parent,
                "child": row.child,
                "branch_length": branch_length,
                "depth": topological_depth[row.child],
                "parent_time": parent_time,
                "child_time": child_time,
            }
        )

    node_rows = []
    for node in sorted(nodes, key=lambda value: (topological_depth[value], value)):
        is_leaf = node not in set(topology_edges["parent"])
        node_rows.append(
            {
                "node_id": node,
                "node_type": (
                    "root" if node == "Root" else "leaf" if is_leaf else "internal"
                ),
                "is_leaf": is_leaf,
                "depth": topological_depth[node],
                "developmental_time": float(node_times[node]),
            }
        )
    return pd.DataFrame(edge_rows), pd.DataFrame(node_rows)


def prepare_retimed_fate_map(
    replicate_dir: str | Path,
    output_name: str,
    node_times: dict[str, float] | None = None,
) -> tuple[Path, Path]:
    replicate_dir = Path(replicate_dir)
    topology = pd.read_csv(
        replicate_dir / "truth/fate_map_edges.tsv",
        sep="\t",
    )
    edges, nodes = retime_fate_map(topology, node_times or DEFAULT_NODE_TIMES)
    output_dir = replicate_dir / "method_inputs" / "RegArch"
    edge_path = output_dir / f"{output_name}.csv"
    node_path = output_dir / f"{output_name}_nodes.tsv"
    edges[["parent", "child", "branch_length"]].rename(
        columns={"parent": "Parent", "child": "Child", "branch_length": "Length"}
    ).to_csv(edge_path, index=False)
    nodes.to_csv(node_path, sep="\t", index=False)
    edges.to_csv(output_dir / f"{output_name}_edges.tsv", sep="\t", index=False)
    return edge_path, node_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retain a fate-map topology while assigning calibrated node times."
    )
    parser.add_argument("replicate_dirs", type=Path, nargs="+")
    parser.add_argument("--output-name", default="fate_map_sampling_time_1")
    args = parser.parse_args()
    for replicate_dir in args.replicate_dirs:
        edge_path, node_path = prepare_retimed_fate_map(
            replicate_dir,
            args.output_name,
        )
        print(f"{edge_path}\t{node_path}")


if __name__ == "__main__":
    main()
