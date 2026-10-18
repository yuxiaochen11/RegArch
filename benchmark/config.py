from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FateMapConfig:
    node_count: int = 7
    branch_lengths: list[float] = field(
        default_factory=lambda: [0.32, 0.38, 0.28, 0.36, 0.34, 0.30]
    )
    wrong_fate_map_mode: str = "paired_rotation"


@dataclass
class GRNConfig:
    regulator_count: int = 100
    target_count: int = 150
    density: float = 0.03
    positive_fraction: float = 0.60
    weight_mean: float = 1.0
    weight_sd: float = 0.25
    weight_min: float = 0.2
    weight_max: float = 1.0
    loss_rate: float = 0.45
    gain_rate: float = 0.015
    sign_switch_rate: float = 0.04
    weight_drift_sd: float = 0.08
    ensure_each_target: bool = True
    allow_self_regulation: bool = False


@dataclass
class RegionConfig:
    promoter_per_target: int = 1
    enhancer_min: int = 4
    enhancer_max: int = 8
    decoy_min: int = 1
    decoy_max: int = 2
    regulatory_regions_min: int = 1
    regulatory_regions_max: int = 3
    motif_true_positive_rate: float = 0.85
    motif_false_positive_rate: float = 0.03
    link_false_positive_rate: float = 0.03


@dataclass
class ObservationConfig:
    cells_per_leaf: int = 300
    rna_library_mean: float = 12000.0
    rna_library_sd: float = 0.25
    rna_dispersion: float = 12.0
    regulatory_scale: float = 0.35
    regulator_noise_sd: float = 0.65
    target_noise_sd: float = 0.35
    rna_dropout_rate: float = 0.30
    atac_depth_sd: float = 0.30
    active_peak_alpha: float = 8.0
    active_peak_beta: float = 2.0
    inactive_peak_alpha: float = 2.0
    inactive_peak_beta: float = 8.0
    decoy_peak_alpha: float = 1.0
    decoy_peak_beta: float = 12.0
    metacells_per_leaf: int = 10


@dataclass
class BenchmarkConfig:
    name: str = "main"
    seed: int = 20250611
    replicates: int = 1
    fate_map: FateMapConfig = field(default_factory=FateMapConfig)
    grn: GRNConfig = field(default_factory=GRNConfig)
    regions: RegionConfig = field(default_factory=RegionConfig)
    observations: ObservationConfig = field(default_factory=ObservationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct_config(raw: dict[str, Any]) -> BenchmarkConfig:
    return BenchmarkConfig(
        name=raw.get("name", "main"),
        seed=int(raw.get("seed", 20250611)),
        replicates=int(raw.get("replicates", 1)),
        fate_map=FateMapConfig(**raw.get("fate_map", {})),
        grn=GRNConfig(**raw.get("grn", {})),
        regions=RegionConfig(**raw.get("regions", {})),
        observations=ObservationConfig(**raw.get("observations", {})),
    )


def load_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return _construct_config(json.loads(path.read_text(encoding="utf-8")))
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML configuration requires PyYAML; use JSON instead."
            ) from exc
        return _construct_config(yaml.safe_load(path.read_text(encoding="utf-8")))
    raise ValueError(f"Unsupported configuration format: {path.suffix}")


def save_config(config: BenchmarkConfig, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
