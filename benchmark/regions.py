from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RegionConfig
from .dynamic_grn import DynamicGRNTruth
from .fate_map import FateMapTruth


@dataclass(frozen=True)
class RegionTruth:
    peaks: pd.DataFrame
    region_to_gene: pd.DataFrame
    motif_links: pd.DataFrame
    regulatory_assignments: pd.DataFrame


def generate_region_truth(
    grn_truth: DynamicGRNTruth,
    fate_map: FateMapTruth,
    config: RegionConfig,
    rng: np.random.Generator,
) -> RegionTruth:
    peak_rows = []
    gene_link_rows = []
    gene_regions: dict[str, list[str]] = {}
    chromosome_count = 5
    peak_index = 0

    for target_index, target in enumerate(grn_truth.targets):
        enhancer_count = int(rng.integers(config.enhancer_min, config.enhancer_max + 1))
        decoy_count = int(rng.integers(config.decoy_min, config.decoy_max + 1))
        region_types = (
            ["promoter"] * config.promoter_per_target
            + ["enhancer"] * enhancer_count
            + ["decoy"] * decoy_count
        )
        candidate_regions = []
        for local_index, region_type in enumerate(region_types):
            peak_index += 1
            region_id = f"Peak{peak_index:06d}"
            chrom = f"chr{target_index % chromosome_count + 1}"
            gene_anchor = 100_000 + target_index * 50_000
            if region_type == "promoter":
                distance = int(rng.integers(-500, 501))
            elif region_type == "enhancer":
                distance = int(rng.choice([-1, 1]) * rng.integers(2_000, 100_001))
            else:
                distance = int(rng.choice([-1, 1]) * rng.integers(100_000, 500_001))
            start = max(1, gene_anchor + distance)
            end = start + 500
            is_candidate = region_type != "decoy"
            if not is_candidate and rng.random() < config.link_false_positive_rate:
                is_candidate = True
            link_score = (
                1.0
                if region_type == "promoter"
                else float(np.exp(-abs(distance) / 75_000))
            )
            if region_type == "decoy":
                link_score *= 0.1
            peak_rows.append(
                {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "region_id": region_id,
                    "region_type": region_type,
                    "linked_target": target,
                }
            )
            gene_link_rows.append(
                {
                    "region_id": region_id,
                    "target": target,
                    "link_score": link_score,
                    "distance": distance,
                    "is_promoter": int(region_type == "promoter"),
                    "is_candidate_link": int(is_candidate),
                }
            )
            if region_type != "decoy":
                candidate_regions.append(region_id)
        gene_regions[target] = candidate_regions

    all_long = []
    for node, matrix in grn_truth.matrices.items():
        edges = matrix.stack()
        edges = edges[edges != 0]
        for (target, regulator), weight in edges.items():
            region_count = int(
                rng.integers(
                    config.regulatory_regions_min,
                    config.regulatory_regions_max + 1,
                )
            )
            available = gene_regions[target]
            selected = rng.choice(
                available,
                size=min(region_count, len(available)),
                replace=False,
            )
            for region_id in selected:
                all_long.append(
                    {
                        "node": node,
                        "regulator": regulator,
                        "region_id": region_id,
                        "target": target,
                        "true_weight": float(weight),
                        "true_sign": int(np.sign(weight)),
                    }
                )
    assignments = pd.DataFrame(all_long).drop_duplicates(
        ["node", "regulator", "region_id", "target"]
    )

    active_pairs = set(zip(assignments["region_id"], assignments["regulator"]))
    motif_rows = []
    for peak in peak_rows:
        region_id = peak["region_id"]
        for regulator in grn_truth.regulators:
            is_true = (region_id, regulator) in active_pairs
            hit_probability = (
                config.motif_true_positive_rate
                if is_true
                else config.motif_false_positive_rate
            )
            motif_hit = int(rng.random() < hit_probability)
            if motif_hit:
                motif_score = float(
                    rng.gamma(4.0, 1.5) if is_true else rng.gamma(1.5, 0.8)
                )
            else:
                motif_score = float(rng.uniform(0, 0.25))
            motif_rows.append(
                {
                    "region_id": region_id,
                    "regulator": regulator,
                    "motif_hit": motif_hit,
                    "motif_score": motif_score,
                }
            )

    return RegionTruth(
        peaks=pd.DataFrame(peak_rows),
        region_to_gene=pd.DataFrame(gene_link_rows),
        motif_links=pd.DataFrame(motif_rows),
        regulatory_assignments=assignments,
    )


def save_region_truth(region_truth: RegionTruth, output_root: str | Path) -> None:
    output_root = Path(output_root)
    truth_dir = output_root / "truth"
    atac_dir = output_root / "intermediate" / "atac"
    truth_dir.mkdir(parents=True, exist_ok=True)
    atac_dir.mkdir(parents=True, exist_ok=True)

    region_truth.regulatory_assignments.to_csv(
        truth_dir / "true_tf_region_gene_edges.tsv", sep="\t", index=False
    )
    region_truth.peaks[["chrom", "start", "end", "region_id"]].to_csv(
        atac_dir / "peak_metadata.bed",
        sep="\t",
        header=False,
        index=False,
    )
    region_truth.peaks.to_csv(atac_dir / "peak_metadata.tsv", sep="\t", index=False)
    region_truth.region_to_gene.to_csv(
        atac_dir / "region_to_gene_links.tsv", sep="\t", index=False
    )
    region_truth.motif_links.to_csv(
        atac_dir / "region_to_tf_links.tsv", sep="\t", index=False
    )
    region_truth.motif_links.to_csv(atac_dir / "motif_hits.tsv", sep="\t", index=False)
