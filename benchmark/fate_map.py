from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FateMapTruth:
    edges: pd.DataFrame
    nodes: pd.DataFrame
    newick: str

    @property
    def leaf_nodes(self) -> list[str]:
        return self.nodes.loc[self.nodes["is_leaf"], "node_id"].tolist()

    @property
    def internal_nodes(self) -> list[str]:
        return self.nodes.loc[~self.nodes["is_leaf"], "node_id"].tolist()


def _node_name(index: int, leaf_start: int) -> str:
    if index == 0:
        return "Root"
    if index >= leaf_start:
        return f"T{index - leaf_start + 1}"
    return f"P{index}"


def generate_balanced_fate_map(
    node_count: int,
    branch_lengths: list[float] | None = None,
) -> FateMapTruth:
    if node_count < 3 or node_count % 2 == 0:
        raise ValueError("node_count must be an odd full-binary-tree size.")

    leaf_start = (node_count - 1) // 2
    edge_count = node_count - 1
    if branch_lengths is None:
        branch_lengths = [0.3 + 0.02 * (i % 5) for i in range(edge_count)]
    if len(branch_lengths) != edge_count:
        raise ValueError(
            f"Expected {edge_count} branch lengths for {node_count} nodes, "
            f"received {len(branch_lengths)}."
        )
    if any(length <= 0 for length in branch_lengths):
        raise ValueError("All branch lengths must be positive.")

    depths = {0: 0}
    times = {0: 0.0}
    edge_rows = []
    for child_index in range(1, node_count):
        parent_index = (child_index - 1) // 2
        length = float(branch_lengths[child_index - 1])
        depths[child_index] = depths[parent_index] + 1
        times[child_index] = times[parent_index] + length
        edge_rows.append(
            {
                "parent": _node_name(parent_index, leaf_start),
                "child": _node_name(child_index, leaf_start),
                "branch_length": length,
                "depth": depths[child_index],
                "parent_time": times[parent_index],
                "child_time": times[child_index],
            }
        )

    node_rows = []
    for index in range(node_count):
        is_leaf = index >= leaf_start
        node_rows.append(
            {
                "node_id": _node_name(index, leaf_start),
                "node_type": "root"
                if index == 0
                else ("leaf" if is_leaf else "internal"),
                "is_leaf": is_leaf,
                "depth": depths[index],
                "developmental_time": times[index],
            }
        )

    edge_df = pd.DataFrame(edge_rows)
    node_df = pd.DataFrame(node_rows)
    children = {
        parent: group.sort_values("child")["child"].tolist()
        for parent, group in edge_df.groupby("parent")
    }
    lengths = {
        (row.parent, row.child): row.branch_length
        for row in edge_df.itertuples(index=False)
    }

    def build(node: str) -> str:
        if node not in children:
            return node
        child_text = ",".join(
            f"{build(child)}:{lengths[(node, child)]:.6f}" for child in children[node]
        )
        return f"({child_text}){node}"

    return FateMapTruth(edges=edge_df, nodes=node_df, newick=build("Root") + ";")


def save_fate_map(fate_map: FateMapTruth, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fate_map.edges.to_csv(output_dir / "fate_map_edges.tsv", sep="\t", index=False)
    fate_map.nodes.to_csv(output_dir / "node_metadata.tsv", sep="\t", index=False)
    (output_dir / "fate_map.nwk").write_text(fate_map.newick + "\n", encoding="utf-8")
    (output_dir / "leaf_nodes.txt").write_text(
        "\n".join(fate_map.leaf_nodes) + "\n", encoding="utf-8"
    )
    (output_dir / "internal_nodes.txt").write_text(
        "\n".join(fate_map.internal_nodes) + "\n", encoding="utf-8"
    )


def descendant_leaf_sets(fate_map: FateMapTruth) -> dict[str, frozenset[str]]:
    children = {
        parent: group["child"].tolist()
        for parent, group in fate_map.edges.groupby("parent")
    }

    def visit(node: str) -> frozenset[str]:
        if node not in children:
            return frozenset([node])
        leaves = frozenset().union(*(visit(child) for child in children[node]))
        result[node] = leaves
        return leaves

    result: dict[str, frozenset[str]] = {}
    for leaf in fate_map.leaf_nodes:
        result[leaf] = frozenset([leaf])
    visit("Root")
    return result
