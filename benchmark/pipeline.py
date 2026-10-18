from __future__ import annotations

from pathlib import Path

import numpy as np

from .ablation import save_wrong_fate_map
from .adapters import prepare_method_inputs
from .config import BenchmarkConfig, save_config
from .diagnostics import save_lineage_diagnostics
from .dynamic_grn import generate_dynamic_grns, grns_to_long, save_dynamic_grns
from .evaluation import build_edge_universe
from .fate_map import generate_balanced_fate_map, save_fate_map
from .input_contract import write_input_contract
from .multiome import save_multiome, simulate_paired_multiome
from .regarch_io import corrupt_prior, export_regarch_inputs
from .regions import generate_region_truth, save_region_truth


def generate_dataset(
    config: BenchmarkConfig,
    output_dir: str | Path,
    replicate_index: int = 1,
) -> Path:
    output_dir = Path(output_dir)
    replicate_dir = output_dir / f"{config.name}_rep_{replicate_index:03d}"
    replicate_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, replicate_dir / "config.json")

    seed_sequence = np.random.SeedSequence([config.seed, replicate_index])
    grn_rng, region_rng, observation_rng, ablation_rng = [
        np.random.default_rng(seed) for seed in seed_sequence.spawn(4)
    ]

    fate_map = generate_balanced_fate_map(
        config.fate_map.node_count,
        config.fate_map.branch_lengths,
    )
    save_fate_map(fate_map, replicate_dir / "truth")

    grn_truth = generate_dynamic_grns(fate_map, config.grn, grn_rng)
    save_dynamic_grns(grn_truth, fate_map, replicate_dir / "truth")
    save_lineage_diagnostics(
        fate_map,
        grn_truth,
        config.grn,
        replicate_dir / "diagnostics",
    )

    region_truth = generate_region_truth(
        grn_truth, fate_map, config.regions, region_rng
    )
    save_region_truth(region_truth, replicate_dir)

    multiome = simulate_paired_multiome(
        fate_map,
        grn_truth,
        region_truth,
        config.observations,
        observation_rng,
    )
    save_multiome(multiome, grn_truth, replicate_dir)

    true_leaf = grns_to_long(grn_truth, fate_map.leaf_nodes)
    edge_universe = build_edge_universe(true_leaf)
    evaluation_dir = replicate_dir / "intermediate" / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    edge_universe.to_csv(evaluation_dir / "edge_universe.tsv", sep="\t", index=False)
    all_universe = build_edge_universe(grns_to_long(grn_truth))
    all_universe.to_csv(
        evaluation_dir / "edge_universe_all_nodes.tsv", sep="\t", index=False
    )
    all_universe.loc[~all_universe["cell_type"].isin(fate_map.leaf_nodes)].to_csv(
        evaluation_dir / "edge_universe_ancestor_nodes.tsv",
        sep="\t",
        index=False,
    )

    prior = export_regarch_inputs(
        replicate_dir, fate_map, grn_truth, multiome, region_truth
    )
    prepare_method_inputs(
        replicate_dir,
        fate_map,
        grn_truth,
        multiome,
        prior,
        config.observations.metacells_per_leaf,
    )
    write_input_contract(replicate_dir)

    ablation_dir = replicate_dir / "ablations"
    save_wrong_fate_map(
        fate_map,
        ablation_dir / "wrong_fate_map",
        mode=config.fate_map.wrong_fate_map_mode,
    )
    leaf_prior = prior.loc[prior["node"].isin(fate_map.leaf_nodes)].copy()
    for coverage, correctness in [(1.0, 1.0), (0.75, 0.9), (0.5, 0.75), (0.25, 0.5)]:
        variant = corrupt_prior(
            leaf_prior,
            true_leaf,
            coverage=coverage,
            correctness=correctness,
            rng=ablation_rng,
        )
        variant.to_csv(
            ablation_dir
            / f"atac_prior_coverage_{coverage:.2f}_correctness_{correctness:.2f}.tsv",
            sep="\t",
            index=False,
        )
    leaf_prior.assign(prior_score=0.0).to_csv(
        ablation_dir / "atac_prior_no_atac.tsv", sep="\t", index=False
    )
    return replicate_dir
