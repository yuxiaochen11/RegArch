from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ObservationConfig
from .dynamic_grn import DynamicGRNTruth
from .fate_map import FateMapTruth
from .regions import RegionTruth


@dataclass(frozen=True)
class MultiomeData:
    cell_metadata: pd.DataFrame
    rna_counts: pd.DataFrame
    rna_lognorm: pd.DataFrame
    atac_counts: pd.DataFrame
    atac_binary: pd.DataFrame


def _negative_binomial(
    means: np.ndarray,
    dispersion: float,
    rng: np.random.Generator,
) -> np.ndarray:
    shape = np.full_like(means, dispersion, dtype=float)
    scale = means / dispersion
    rates = rng.gamma(shape=shape, scale=np.maximum(scale, 1e-10))
    return rng.poisson(rates)


def _simulate_rna_for_leaf(
    node: str,
    cell_ids: list[str],
    grn: pd.DataFrame,
    config: ObservationConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    cell_count = len(cell_ids)
    regulator_count = grn.shape[1]
    target_count = grn.shape[0]

    node_shift = rng.normal(0, 0.35, size=regulator_count)
    regulator_noise = rng.normal(
        node_shift,
        config.regulator_noise_sd,
        size=(cell_count, regulator_count),
    )
    regulator_network = grn.iloc[:regulator_count].to_numpy()
    spectral_radius = max(
        np.max(np.abs(np.linalg.eigvals(regulator_network))),
        1e-8,
    )
    effective_scale = min(
        config.regulatory_scale,
        0.8 / spectral_radius,
    )
    regulator_system = np.eye(regulator_count) - effective_scale * regulator_network.T
    regulator_expression = np.linalg.solve(
        regulator_system.T,
        regulator_noise.T,
    ).T
    nonzero_per_target = np.maximum((grn.to_numpy() != 0).sum(axis=1), 1)
    regulatory_signal = (regulator_expression @ grn.to_numpy().T) / np.sqrt(
        nonzero_per_target
    )[None, :]
    baseline = rng.normal(0.0, 0.55, size=target_count)
    log_means = (
        baseline[None, :]
        + effective_scale * regulatory_signal
        + rng.normal(
            0,
            config.target_noise_sd,
            size=(cell_count, target_count),
        )
    )
    # Regulator genes can also be targets. Their incoming regulatory signal
    # must remain in the observation model or signed TF-to-TF edges become
    # unidentifiable from the simulated RNA.
    log_means[:, :regulator_count] += regulator_expression
    relative = np.exp(np.clip(log_means, -5, 5))
    relative /= relative.sum(axis=1, keepdims=True)
    libraries = np.exp(
        rng.normal(
            np.log(config.rna_library_mean),
            config.rna_library_sd,
            size=cell_count,
        )
    )
    means = relative * libraries[:, None]
    counts = _negative_binomial(means, config.rna_dispersion, rng)
    mean_scale = np.median(means[means > 0]) if np.any(means > 0) else 1.0
    dropout_probability = config.rna_dropout_rate * np.exp(
        -means / max(mean_scale, 1e-8)
    )
    counts[rng.random(counts.shape) < dropout_probability] = 0
    return pd.DataFrame(counts.T, index=grn.index, columns=cell_ids)


def _simulate_atac_for_leaf(
    node: str,
    cell_ids: list[str],
    region_truth: RegionTruth,
    config: ObservationConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    peak_ids = region_truth.peaks["region_id"].tolist()
    active = set(
        region_truth.regulatory_assignments.loc[
            region_truth.regulatory_assignments["node"] == node,
            "region_id",
        ]
    )
    region_types = region_truth.peaks.set_index("region_id")["region_type"]
    base_probabilities = []
    for region_id in peak_ids:
        if region_id in active:
            probability = rng.beta(config.active_peak_alpha, config.active_peak_beta)
        elif region_types[region_id] == "decoy":
            probability = rng.beta(config.decoy_peak_alpha, config.decoy_peak_beta)
        else:
            probability = rng.beta(
                config.inactive_peak_alpha, config.inactive_peak_beta
            )
        base_probabilities.append(probability)
    base_probabilities = np.asarray(base_probabilities)

    depth_factor = np.exp(rng.normal(0, config.atac_depth_sd, size=len(cell_ids)))
    probabilities = 1.0 - np.power(
        1.0 - base_probabilities[:, None],
        depth_factor[None, :],
    )
    binary = rng.binomial(1, np.clip(probabilities, 0, 1))
    return pd.DataFrame(binary, index=peak_ids, columns=cell_ids)


def simulate_paired_multiome(
    fate_map: FateMapTruth,
    grn_truth: DynamicGRNTruth,
    region_truth: RegionTruth,
    config: ObservationConfig,
    rng: np.random.Generator,
) -> MultiomeData:
    metadata_rows = []
    rna_parts = []
    atac_parts = []
    cell_offset = 0

    parent_by_child = fate_map.edges.set_index("child")["parent"].to_dict()
    for leaf in fate_map.leaf_nodes:
        cell_ids = [
            f"Cell{cell_offset + i + 1:06d}" for i in range(config.cells_per_leaf)
        ]
        cell_offset += config.cells_per_leaf
        path = [leaf]
        while path[-1] != "Root":
            path.append(parent_by_child[path[-1]])
        path.reverse()
        for cell_id in cell_ids:
            metadata_rows.append(
                {
                    "cell_id": cell_id,
                    "cell_type": leaf,
                    "leaf_node": leaf,
                    "branch_id": "-".join(path),
                    "pseudotime": 1.0,
                    "developmental_time": float(
                        fate_map.nodes.set_index("node_id").loc[
                            leaf, "developmental_time"
                        ]
                    ),
                    "is_leaf_observation": 1,
                }
            )
        rna_parts.append(
            _simulate_rna_for_leaf(
                leaf, cell_ids, grn_truth.matrices[leaf], config, rng
            )
        )
        atac_parts.append(
            _simulate_atac_for_leaf(leaf, cell_ids, region_truth, config, rng)
        )

    rna_counts = pd.concat(rna_parts, axis=1)
    atac_counts = pd.concat(atac_parts, axis=1)
    library_sizes = rna_counts.sum(axis=0).replace(0, 1)
    rna_lognorm = np.log1p(rna_counts.div(library_sizes, axis=1) * 1_000_000)

    return MultiomeData(
        cell_metadata=pd.DataFrame(metadata_rows),
        rna_counts=rna_counts,
        rna_lognorm=rna_lognorm,
        atac_counts=atac_counts,
        atac_binary=(atac_counts > 0).astype(int),
    )


def save_multiome(
    data: MultiomeData,
    grn_truth: DynamicGRNTruth,
    output_root: str | Path,
) -> None:
    output_root = Path(output_root)
    metadata_dir = output_root / "intermediate" / "metadata"
    rna_dir = output_root / "intermediate" / "rna"
    atac_dir = output_root / "intermediate" / "atac"
    for directory in (metadata_dir, rna_dir, atac_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data.cell_metadata.to_csv(metadata_dir / "cell_metadata.tsv", sep="\t", index=False)
    gene_metadata = pd.DataFrame(
        {
            "gene": grn_truth.targets,
            "is_regulator": [
                int(gene in set(grn_truth.regulators)) for gene in grn_truth.targets
            ],
            "is_target": 1,
        }
    )
    gene_metadata.to_csv(metadata_dir / "gene_metadata.tsv", sep="\t", index=False)
    (metadata_dir / "regulator_list.txt").write_text(
        "\n".join(grn_truth.regulators) + "\n", encoding="utf-8"
    )
    (metadata_dir / "target_list.txt").write_text(
        "\n".join(grn_truth.targets) + "\n", encoding="utf-8"
    )

    data.rna_counts.to_csv(rna_dir / "rna_counts_gene_by_cell.tsv", sep="\t")
    data.rna_counts.T.to_csv(rna_dir / "rna_counts_cell_by_gene.tsv", sep="\t")
    data.rna_lognorm.to_csv(rna_dir / "rna_lognorm_gene_by_cell.tsv", sep="\t")
    data.rna_lognorm.T.to_csv(rna_dir / "rna_lognorm_cell_by_gene.tsv", sep="\t")
    data.atac_counts.to_csv(atac_dir / "atac_counts_region_by_cell.tsv", sep="\t")
    data.atac_counts.T.to_csv(atac_dir / "atac_counts_cell_by_region.tsv", sep="\t")
    data.atac_binary.to_csv(atac_dir / "atac_binary_region_by_cell.tsv", sep="\t")
