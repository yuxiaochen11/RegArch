from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from .config import BenchmarkConfig, load_config, save_config


def _balanced_branch_lengths(node_count: int) -> list[float]:
    return [0.26 + 0.02 * (index % 7) for index in range(node_count - 1)]


def write_sensitivity_configs(
    base_config: BenchmarkConfig,
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = []

    for node_count in (7, 15):
        config = deepcopy(base_config)
        config.name = f"sensitivity_nodes_{node_count}"
        config.replicates = min(config.replicates, 5)
        config.fate_map.node_count = node_count
        config.fate_map.branch_lengths = _balanced_branch_lengths(node_count)
        variants.append(config)

    for label, loss, gain, switch in [
        ("low", 0.20, 0.005, 0.01),
        ("medium", 0.45, 0.015, 0.04),
        ("high", 0.80, 0.030, 0.08),
    ]:
        config = deepcopy(base_config)
        config.name = f"sensitivity_rewiring_{label}"
        config.replicates = min(config.replicates, 5)
        config.grn.loss_rate = loss
        config.grn.gain_rate = gain
        config.grn.sign_switch_rate = switch
        variants.append(config)

    paths = []
    for config in variants:
        path = output_dir / f"{config.name}.json"
        save_config(config, path)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Create sensitivity configurations.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in write_sensitivity_configs(load_config(args.base_config), args.output):
        print(path)


if __name__ == "__main__":
    main()
