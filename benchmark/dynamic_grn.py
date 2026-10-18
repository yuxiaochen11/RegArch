from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

from .config import GRNConfig
from .fate_map import FateMapTruth

STATE_ORDER = np.array([-1, 0, 1], dtype=int)


@dataclass(frozen=True)
class DynamicGRNTruth:
    matrices: dict[str, pd.DataFrame]
    transition_log: pd.DataFrame
    regulators: list[str]
    targets: list[str]


def transition_generator(config: GRNConfig) -> np.ndarray:
    positive_gain = config.gain_rate * config.positive_fraction
    negative_gain = config.gain_rate * (1.0 - config.positive_fraction)
    q = np.array(
        [
            [
                -(config.loss_rate + config.sign_switch_rate),
                config.loss_rate,
                config.sign_switch_rate,
            ],
            [negative_gain, -config.gain_rate, positive_gain],
            [
                config.sign_switch_rate,
                config.loss_rate,
                -(config.loss_rate + config.sign_switch_rate),
            ],
        ],
        dtype=float,
    )
    return q


def transition_matrix(branch_length: float, config: GRNConfig) -> np.ndarray:
    matrix = expm(transition_generator(config) * branch_length)
    matrix[matrix < 0] = 0
    return matrix / matrix.sum(axis=1, keepdims=True)


def _sample_magnitude(rng: np.random.Generator, config: GRNConfig) -> float:
    return float(
        np.clip(
            abs(rng.normal(config.weight_mean, config.weight_sd)),
            config.weight_min,
            config.weight_max,
        )
    )


def _gene_names(config: GRNConfig) -> tuple[list[str], list[str]]:
    regulators = [f"TF{i + 1:03d}" for i in range(config.regulator_count)]
    targets = regulators + [
        f"Gene{i + 1:03d}" for i in range(config.target_count - config.regulator_count)
    ]
    return regulators, targets


def generate_root_grn(
    config: GRNConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if config.regulator_count > config.target_count:
        raise ValueError("regulator_count must not exceed target_count.")
    regulators, targets = _gene_names(config)
    shape = (config.target_count, config.regulator_count)
    matrix = np.zeros(shape, dtype=float)
    allowed = np.ones(shape, dtype=bool)
    if not config.allow_self_regulation:
        for regulator_index in range(config.regulator_count):
            allowed[regulator_index, regulator_index] = False
    allowed_flat = np.flatnonzero(allowed.ravel())
    edge_count = max(1, int(round(len(allowed_flat) * config.density)))
    selected = rng.choice(
        allowed_flat,
        size=min(edge_count, len(allowed_flat)),
        replace=False,
    )
    for flat_index in selected:
        target_index, regulator_index = np.unravel_index(flat_index, shape)
        sign = 1 if rng.random() < config.positive_fraction else -1
        matrix[target_index, regulator_index] = sign * _sample_magnitude(rng, config)

    if config.ensure_each_target:
        for target_index in range(config.target_count):
            if not np.any(matrix[target_index]):
                candidates = np.flatnonzero(allowed[target_index])
                regulator_index = int(rng.choice(candidates))
                sign = 1 if rng.random() < config.positive_fraction else -1
                matrix[target_index, regulator_index] = sign * _sample_magnitude(
                    rng, config
                )

    return pd.DataFrame(matrix, index=targets, columns=regulators), regulators, targets


def _sample_child(
    parent: pd.DataFrame,
    branch_length: float,
    config: GRNConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    probabilities = transition_matrix(branch_length, config)
    parent_values = parent.to_numpy()
    parent_states = np.sign(parent_values).astype(int)
    child_values = np.zeros_like(parent_values)

    for state_index, state in enumerate(STATE_ORDER):
        positions = np.argwhere(parent_states == state)
        if not len(positions):
            continue
        child_states = rng.choice(
            STATE_ORDER,
            size=len(positions),
            p=probabilities[state_index],
        )
        for (target_index, regulator_index), child_state in zip(
            positions, child_states
        ):
            if child_state == 0:
                continue
            if state == 0:
                magnitude = _sample_magnitude(rng, config)
            else:
                drift = rng.normal(0, config.weight_drift_sd * np.sqrt(branch_length))
                magnitude = float(
                    np.clip(
                        abs(parent_values[target_index, regulator_index]) + drift,
                        config.weight_min,
                        config.weight_max,
                    )
                )
            child_values[target_index, regulator_index] = child_state * magnitude

    if not config.allow_self_regulation:
        diagonal = min(config.regulator_count, child_values.shape[0])
        child_values[np.arange(diagonal), np.arange(diagonal)] = 0.0

    return pd.DataFrame(child_values, index=parent.index, columns=parent.columns)


def generate_dynamic_grns(
    fate_map: FateMapTruth,
    config: GRNConfig,
    rng: np.random.Generator,
) -> DynamicGRNTruth:
    root, regulators, targets = generate_root_grn(config, rng)
    matrices = {"Root": root}
    log_rows = []

    ordered_edges = fate_map.edges.sort_values(["depth", "parent", "child"])
    for row in ordered_edges.itertuples(index=False):
        child = _sample_child(
            matrices[row.parent],
            float(row.branch_length),
            config,
            rng,
        )
        matrices[row.child] = child
        p = transition_matrix(float(row.branch_length), config)
        log_rows.append(
            {
                "parent": row.parent,
                "child": row.child,
                "branch_length": row.branch_length,
                "parent_edges": int(np.count_nonzero(matrices[row.parent].to_numpy())),
                "child_edges": int(np.count_nonzero(child.to_numpy())),
                "p_negative_to_zero": p[0, 1],
                "p_zero_to_negative": p[1, 0],
                "p_zero_to_positive": p[1, 2],
                "p_positive_to_zero": p[2, 1],
                "p_sign_switch": (p[0, 2] + p[2, 0]) / 2,
            }
        )

    return DynamicGRNTruth(
        matrices=matrices,
        transition_log=pd.DataFrame(log_rows),
        regulators=regulators,
        targets=targets,
    )


def grns_to_long(
    truth: DynamicGRNTruth, nodes: list[str] | None = None
) -> pd.DataFrame:
    rows = []
    selected_nodes = nodes or list(truth.matrices)
    for node in selected_nodes:
        matrix = truth.matrices[node]
        stacked = matrix.stack(future_stack=True).rename("true_weight").reset_index()
        stacked.columns = ["target", "regulator", "true_weight"]
        stacked.insert(0, "node", node)
        stacked["true_edge"] = (stacked["true_weight"] != 0).astype(int)
        stacked["true_sign"] = np.sign(stacked["true_weight"]).astype(int)
        rows.append(stacked)
    return pd.concat(rows, ignore_index=True)


def save_dynamic_grns(
    truth: DynamicGRNTruth,
    fate_map: FateMapTruth,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    matrix_dir = output_dir / "grn_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for node, matrix in truth.matrices.items():
        matrix.to_csv(matrix_dir / f"{node}.csv")
    grns_to_long(truth).to_csv(
        output_dir / "true_grn_all_nodes.tsv", sep="\t", index=False
    )
    grns_to_long(truth, fate_map.leaf_nodes).to_csv(
        output_dir / "true_grn_leaf_nodes.tsv", sep="\t", index=False
    )
    truth.transition_log.to_csv(
        output_dir / "rewiring_transition_log.tsv", sep="\t", index=False
    )
