from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GRNConfig
from .dynamic_grn import DynamicGRNTruth, transition_matrix
from .fate_map import FateMapTruth, descendant_leaf_sets


def _edge_presence(matrix: pd.DataFrame) -> np.ndarray:
    return (matrix.to_numpy().ravel() != 0).astype(np.int8)


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def lineage_signal_table(
    fate_map: FateMapTruth,
    grn_truth: DynamicGRNTruth,
    config: GRNConfig,
) -> pd.DataFrame:
    """Summarize whether the simulated leaves contain usable tree signal."""
    descendants = descendant_leaf_sets(fate_map)
    sibling_pairs = set()
    for leaves in descendants.values():
        if len(leaves) == 2:
            sibling_pairs.add(tuple(sorted(leaves)))

    rows = []
    for left, right in combinations(fate_map.leaf_nodes, 2):
        left_state = _edge_presence(grn_truth.matrices[left])
        right_state = _edge_presence(grn_truth.matrices[right])
        rows.append(
            {
                "record_type": "leaf_pair",
                "left": left,
                "right": right,
                "relationship": (
                    "sibling"
                    if tuple(sorted((left, right))) in sibling_pairs
                    else "cross_clade"
                ),
                "jaccard": _jaccard(left_state, right_state),
                "agreement": float(np.mean(left_state == right_state)),
                "parent": "",
                "child": "",
                "branch_length": np.nan,
                "observed_loss": np.nan,
                "observed_gain": np.nan,
                "expected_loss": np.nan,
                "expected_gain": np.nan,
            }
        )

    for edge in fate_map.edges.itertuples(index=False):
        parent = _edge_presence(grn_truth.matrices[edge.parent])
        child = _edge_presence(grn_truth.matrices[edge.child])
        parent_present = parent == 1
        parent_absent = parent == 0
        transition = transition_matrix(float(edge.branch_length), config)
        rows.append(
            {
                "record_type": "branch",
                "left": "",
                "right": "",
                "relationship": "",
                "jaccard": np.nan,
                "agreement": float(np.mean(parent == child)),
                "parent": edge.parent,
                "child": edge.child,
                "branch_length": edge.branch_length,
                "observed_loss": (
                    float(np.mean(child[parent_present] == 0))
                    if parent_present.any()
                    else np.nan
                ),
                "observed_gain": (
                    float(np.mean(child[parent_absent] == 1))
                    if parent_absent.any()
                    else np.nan
                ),
                "expected_loss": float(transition[0, 1]),
                "expected_gain": float(transition[1, 0] + transition[1, 2]),
            }
        )
    return pd.DataFrame(rows)


def save_lineage_diagnostics(
    fate_map: FateMapTruth,
    grn_truth: DynamicGRNTruth,
    config: GRNConfig,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = lineage_signal_table(fate_map, grn_truth, config)
    table.to_csv(output_dir / "lineage_signal.tsv", sep="\t", index=False)

    pairs = table.loc[table["record_type"] == "leaf_pair"]
    summary = pairs.groupby("relationship", as_index=False).agg(
        mean_jaccard=("jaccard", "mean"), mean_agreement=("agreement", "mean")
    )
    summary.to_csv(output_dir / "lineage_signal_summary.tsv", sep="\t", index=False)
