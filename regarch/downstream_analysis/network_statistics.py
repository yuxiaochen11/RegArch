"""Query regulator and target values from inferred node-specific networks."""

from pathlib import Path

from regarch.utils.data_io import read_target_network


def get_regulators_for_target_gene(
    target_gene_id: str,
    node_id: str,
    input_path: str,
    regulator_names: list[str],
) -> dict[str, float]:
    """Return regulator strengths for one target gene at one fate-map node."""
    table = read_target_network(
        Path(input_path) / f"target_gene_{target_gene_id}.csv", regulator_names
    )
    selected = table.loc[table["node_id"] == str(node_id), regulator_names]
    if selected.empty:
        raise KeyError(f"Node {node_id!r} is absent for target gene {target_gene_id!r}")
    return {name: float(value) for name, value in selected.iloc[0].items()}


def get_targets_for_regulator_gene(
    regulator_id: str,
    node_id: str,
    input_path: str,
    regulator_names: list[str],
) -> dict[str, float]:
    """Return target-gene strengths for one regulator at one fate-map node."""
    if regulator_id not in regulator_names:
        raise KeyError(f"Unknown regulator: {regulator_id!r}")

    target_values = {}
    for file_path in sorted(Path(input_path).glob("target_gene_*.csv")):
        table = read_target_network(file_path, regulator_names)
        selected = table.loc[table["node_id"] == str(node_id)]
        if selected.empty:
            raise KeyError(f"Node {node_id!r} is absent from {file_path}")
        target_gene_id = str(selected.iloc[0]["target_gene_id"])
        target_values[target_gene_id] = float(selected.iloc[0][regulator_id])
    if not target_values:
        raise FileNotFoundError(
            f"No target_gene_*.csv files were found in {input_path}"
        )
    return target_values
