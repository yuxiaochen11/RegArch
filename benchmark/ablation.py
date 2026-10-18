from __future__ import annotations

from pathlib import Path

import pandas as pd

from .fate_map import FateMapTruth, descendant_leaf_sets


def make_wrong_fate_map(
    fate_map: FateMapTruth,
    mode: str = "paired_rotation",
) -> FateMapTruth:
    if len(fate_map.leaf_nodes) < 4:
        raise ValueError(
            "Wrong-topology perturbation requires at least four leaf nodes."
        )
    edges = fate_map.edges.copy()
    leaf_edges = edges[edges["child"].isin(fate_map.leaf_nodes)].copy()
    leaf_edges["_leaf_number"] = (
        leaf_edges["child"].str.extract(r"(\d+)$")[0].astype(int)
    )
    leaf_edges = leaf_edges.sort_values("_leaf_number")
    indices = leaf_edges.index.tolist()
    parents = edges.loc[indices, "parent"].tolist()
    compress_branch_lengths = False
    if mode == "paired_rotation":
        # Rotate complete sibling pairs. This preserves the number of children
        # per parent while assigning every leaf to a different terminal clade.
        rotated = parents[2:] + parents[:2]
    elif mode in {"cross_clade_split", "cross_clade_short_branches"}:
        compress_branch_lengths = mode == "cross_clade_short_branches"
        terminal_parents = list(dict.fromkeys(parents))
        if len(terminal_parents) < 3:
            raise ValueError(
                "cross_clade_split requires at least three terminal parents."
            )
        group_sizes = pd.Series(parents).value_counts()
        if group_sizes.nunique() == 1 and int(group_sizes.iloc[0]) == 2:
            parent_rank = {
                parent: index for index, parent in enumerate(terminal_parents)
            }
            seen_within_parent = {parent: 0 for parent in terminal_parents}
            rotated = []
            for parent in parents:
                within_parent = seen_within_parent[parent]
                seen_within_parent[parent] += 1
                rotated.append(
                    terminal_parents[
                        (parent_rank[parent] + within_parent + 1)
                        % len(terminal_parents)
                    ]
                )
        else:
            # Fallback for uneven terminal groups: rotate the multiset of
            # terminal parents while avoiding fixed points.
            rotated = []
            for shift in range(1, len(parents)):
                candidate = parents[shift:] + parents[:shift]
                if all(left != right for left, right in zip(parents, candidate)):
                    rotated = candidate
                    break
        if not rotated:
            raise ValueError(
                "Unable to construct a fixed-point-free wrong fate map for "
                f"terminal parent assignment: {parents}"
            )
    else:
        raise ValueError(f"Unsupported wrong fate-map mode: {mode}")
    edges.loc[indices, "parent"] = rotated
    parent_times = fate_map.nodes.set_index("node_id")["developmental_time"].to_dict()
    leaf_time = float(
        fate_map.nodes.loc[fate_map.nodes["is_leaf"], "developmental_time"].iloc[0]
    )
    edges.loc[indices, "branch_length"] = [
        leaf_time - float(parent_times[parent]) for parent in rotated
    ]
    if compress_branch_lengths:
        # Strong negative control: the topology is already wrong, and a very
        # short time scale forces unrelated terminal states to share GRNs.
        edges["branch_length"] = 0.001

    children = {
        parent: group.sort_values("child")["child"].tolist()
        for parent, group in edges.groupby("parent")
    }
    lengths = {
        (row.parent, row.child): row.branch_length
        for row in edges.itertuples(index=False)
    }

    def build(node: str) -> str:
        if node not in children:
            return node
        inner = ",".join(
            f"{build(child)}:{lengths[(node, child)]:.6f}" for child in children[node]
        )
        return f"({inner}){node}"

    return FateMapTruth(
        edges=edges, nodes=fate_map.nodes.copy(), newick=build("Root") + ";"
    )


def map_ancestor_nodes_by_descendants(
    inferred_map: FateMapTruth,
    truth_map: FateMapTruth,
) -> pd.DataFrame:
    inferred_sets = descendant_leaf_sets(inferred_map)
    truth_sets = descendant_leaf_sets(truth_map)
    rows = []
    for inferred_node in inferred_map.internal_nodes:
        candidates = []
        for truth_node in truth_map.internal_nodes:
            intersection = len(inferred_sets[inferred_node] & truth_sets[truth_node])
            union = len(inferred_sets[inferred_node] | truth_sets[truth_node])
            candidates.append((intersection / union if union else 0.0, truth_node))
        score, truth_node = max(candidates)
        rows.append(
            {
                "inferred_node": inferred_node,
                "truth_node": truth_node,
                "descendant_leaf_jaccard": score,
            }
        )
    return pd.DataFrame(rows)


def save_wrong_fate_map(
    fate_map: FateMapTruth,
    output_dir: str | Path,
    mode: str = "paired_rotation",
) -> FateMapTruth:
    wrong = make_wrong_fate_map(fate_map, mode=mode)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wrong.edges[["parent", "child", "branch_length"]].rename(
        columns={"parent": "Parent", "child": "Child", "branch_length": "Length"}
    ).to_csv(output_dir / "wrong_fate_map_edges.csv", index=False)
    (output_dir / "wrong_fate_map.nwk").write_text(
        wrong.newick + "\n", encoding="utf-8"
    )
    map_ancestor_nodes_by_descendants(wrong, fate_map).to_csv(
        output_dir / "wrong_to_truth_ancestor_mapping.tsv", sep="\t", index=False
    )
    return wrong
