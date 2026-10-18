from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.special import expit

from .evaluation import build_edge_universe, evaluate_scores
from .run_model_misspecification import (
    MisspecificationCondition,
    _topology_only_fate_map,
    model_misspecification_conditions,
    odds_scaled_probability,
    resolve_replicate_dirs,
    write_study_tables,
)

EDGE_THRESHOLD = 0.10
EDGE_TEMPERATURE = 0.03
SIGN_TEMPERATURE = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_edges(path: Path) -> pd.DataFrame:
    edges = pd.read_csv(path).rename(
        columns={"Parent": "parent", "Child": "child", "Length": "branch_length"}
    )
    required = ["parent", "child", "branch_length"]
    missing = set(required) - set(edges)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    edges = edges[required].copy()
    edges["branch_length"] = edges["branch_length"].astype(float)
    if (edges["branch_length"] <= 0).any():
        raise ValueError(f"{path} contains non-positive branch lengths")
    return edges


def _transition_generator(
    gain_rate: float,
    loss_rate: float,
    sign_switch_rate: float,
    positive_fraction: float,
) -> np.ndarray:
    if min(gain_rate, loss_rate, sign_switch_rate) <= 0:
        raise ValueError("all CTMC rates must be positive")
    if not 0 < positive_fraction < 1:
        raise ValueError("positive fraction must lie strictly between 0 and 1")
    negative_gain = gain_rate * (1.0 - positive_fraction)
    positive_gain = gain_rate * positive_fraction
    return np.array(
        [
            [-(loss_rate + sign_switch_rate), loss_rate, sign_switch_rate],
            [negative_gain, -gain_rate, positive_gain],
            [sign_switch_rate, loss_rate, -(loss_rate + sign_switch_rate)],
        ],
        dtype=float,
    )


def _state_evidence(
    weights: np.ndarray,
    edge_temperature: float,
    sign_temperature: float,
) -> np.ndarray:
    if edge_temperature <= 0 or sign_temperature <= 0:
        raise ValueError("terminal-evidence temperatures must be positive")
    values = np.asarray(weights, dtype=float)
    magnitudes = np.sqrt(np.square(values) + 1e-8)
    present = expit((magnitudes - EDGE_THRESHOLD) / edge_temperature)
    positive_given_present = expit(values / sign_temperature)
    probabilities = np.stack(
        (
            present * (1.0 - positive_given_present),
            1.0 - present,
            present * positive_given_present,
        ),
        axis=-1,
    )
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def _root_state_prior(
    edge_probabilities: np.ndarray,
    positive_fraction: float,
) -> np.ndarray:
    probabilities = np.asarray(edge_probabilities, dtype=float)
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("root-edge prior probabilities must lie in [0, 1]")
    return np.stack(
        (
            probabilities * (1.0 - positive_fraction),
            1.0 - probabilities,
            probabilities * positive_fraction,
        ),
        axis=-1,
    )


def exact_tree_posteriors(
    edges: pd.DataFrame,
    leaf_evidence: dict[str, np.ndarray],
    root_prior: np.ndarray,
    generator: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run vectorized exact sum-product for independent three-state edges."""
    children: dict[str, list[tuple[str, np.ndarray]]] = {}
    parent_by_child = {}
    all_nodes = set()
    for row in edges.itertuples(index=False):
        transition = expm(generator * float(row.branch_length))
        transition[transition < 0] = 0
        transition /= transition.sum(axis=1, keepdims=True)
        children.setdefault(row.parent, []).append((row.child, transition))
        children.setdefault(row.child, [])
        if row.child in parent_by_child:
            raise ValueError(f"node {row.child} has more than one parent")
        parent_by_child[row.child] = row.parent
        all_nodes.update((row.parent, row.child))
    roots = all_nodes - set(parent_by_child)
    if len(roots) != 1:
        raise ValueError(f"expected one root, found {sorted(roots)}")
    root = next(iter(roots))
    missing_leaves = set(leaf_evidence) - all_nodes
    if missing_leaves:
        raise ValueError(
            f"terminal evidence contains unknown leaves: {sorted(missing_leaves)}"
        )
    inferred_leaves = {
        node for node, descendants in children.items() if not descendants
    }
    if inferred_leaves != set(leaf_evidence):
        raise ValueError(
            "tree leaves and terminal evidence differ: "
            f"tree={sorted(inferred_leaves)}, evidence={sorted(leaf_evidence)}"
        )

    edge_count = root_prior.shape[0]
    postorder = []
    visiting = set()
    visited = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("fate map contains a directed cycle")
        if node in visited:
            return
        visiting.add(node)
        for child, _ in children[node]:
            visit(child)
        visiting.remove(node)
        visited.add(node)
        postorder.append(node)

    visit(root)
    if visited != all_nodes:
        raise ValueError("fate map contains nodes disconnected from the root")

    evidence = {
        node: leaf_evidence.get(node, np.ones((edge_count, 3), dtype=float))
        for node in all_nodes
    }
    upward = {}
    child_messages = {}
    for node in postorder:
        value = evidence[node].copy()
        for child, transition in children[node]:
            message = upward[child] @ transition.T
            child_messages[(node, child)] = message
            value *= message
        upward[node] = value

    downward = {root: root_prior}
    posteriors = {}
    stack = [root]
    while stack:
        node = stack.pop()
        marginal = downward[node] * evidence[node]
        for child, _ in children[node]:
            marginal *= child_messages[(node, child)]
        denominator = marginal.sum(axis=1, keepdims=True)
        if np.any(denominator <= 0):
            raise FloatingPointError(f"zero posterior normalizer at node {node}")
        posteriors[node] = marginal / denominator

        for child, transition in children[node]:
            context = downward[node] * evidence[node]
            for sibling, _ in children[node]:
                if sibling != child:
                    context *= child_messages[(node, sibling)]
            downward[child] = context @ transition
            stack.append(child)
    return posteriors


def _edge_index(truth: pd.DataFrame) -> pd.DataFrame:
    root = truth.loc[truth["node"].eq("Root"), ["target", "regulator"]].copy()
    if root.empty or root.duplicated().any():
        raise ValueError("truth must contain one complete Root edge universe")
    return root.reset_index(drop=True)


def _leaf_evidence(
    truth: pd.DataFrame,
    edge_index: pd.DataFrame,
    leaf_nodes: list[str],
    edge_temperature: float,
    sign_temperature: float,
) -> dict[str, np.ndarray]:
    result = {}
    for leaf in leaf_nodes:
        weights = edge_index.merge(
            truth.loc[truth["node"].eq(leaf), ["target", "regulator", "true_weight"]],
            on=["target", "regulator"],
            how="left",
            validate="one_to_one",
        )["true_weight"]
        if weights.isna().any():
            raise ValueError(f"truth is incomplete for leaf {leaf}")
        result[leaf] = _state_evidence(
            weights.to_numpy(), edge_temperature, sign_temperature
        )
    return result


def _root_probabilities(
    prior: pd.DataFrame,
    edge_index: pd.DataFrame,
    odds_multiplier: float,
) -> tuple[np.ndarray, float, float]:
    root = prior.loc[prior["node"].eq("Root"), ["target", "regulator", "prior_score"]]
    merged = edge_index.merge(
        root,
        on=["target", "regulator"],
        how="left",
        validate="one_to_one",
    )
    probabilities = merged["prior_score"].fillna(0.0).to_numpy(dtype=float)
    self_mask = edge_index["target"].eq(edge_index["regulator"]).to_numpy()
    eligible = ~self_mask
    if not eligible.any():
        raise ValueError("root prior contains no eligible non-self edges")
    before = float(probabilities[eligible].mean())
    numerator = odds_multiplier * probabilities
    probabilities = numerator / (1.0 - probabilities + numerator)
    after = float(probabilities[eligible].mean())
    probabilities[self_mask] = 0.0
    return probabilities, before, after


def _condition_parameters(
    grn: dict,
    condition: MisspecificationCondition,
) -> dict[str, float]:
    parameters = {
        "gain_rate": float(grn["gain_rate"]),
        "loss_rate": float(grn["loss_rate"]),
        "sign_switch_rate": float(grn["sign_switch_rate"]),
        "positive_fraction": float(grn["positive_fraction"]),
        "root_prior_odds_multiplier": 1.0,
        "branch_length_multiplier": 1.0,
        "edge_temperature": EDGE_TEMPERATURE,
        "sign_temperature": SIGN_TEMPERATURE,
    }
    if condition.factor in {"gain_rate", "loss_rate", "sign_switch_rate"}:
        parameters[condition.factor] *= condition.multiplier
    elif condition.factor == "positive_gain_odds":
        parameters["positive_fraction"] = odds_scaled_probability(
            parameters["positive_fraction"], condition.multiplier
        )
    elif condition.factor == "root_prior_odds":
        parameters["root_prior_odds_multiplier"] = condition.multiplier
    elif condition.factor == "branch_length":
        parameters["branch_length_multiplier"] = condition.multiplier
    elif condition.factor == "terminal_temperature":
        parameters["edge_temperature"] *= condition.multiplier
        parameters["sign_temperature"] *= condition.multiplier
    return parameters


def _scores_from_posteriors(
    posteriors: dict[str, np.ndarray],
    edge_index: pd.DataFrame,
    condition: MisspecificationCondition,
    node_mapping: Path | None,
) -> pd.DataFrame:
    parts = []
    for node, probabilities in posteriors.items():
        presence = probabilities[:, 0] + probabilities[:, 2]
        sign = np.where(probabilities[:, 2] >= probabilities[:, 0], 1, -1)
        part = edge_index.copy()
        part["cell_type"] = node
        part["signed_score"] = presence * sign
        parts.append(part)
    scores = pd.concat(parts, ignore_index=True)
    if node_mapping is not None:
        mapping = pd.read_csv(node_mapping, sep="\t")
        scores["cell_type"] = scores["cell_type"].replace(
            dict(zip(mapping["inferred_node"], mapping["truth_node"]))
        )
        scores = scores.groupby(["cell_type", "regulator", "target"], as_index=False)[
            "signed_score"
        ].mean()
    scores.insert(0, "method", condition.name)
    scores["score"] = scores["signed_score"].abs()
    scores["sign_optional"] = np.sign(scores["signed_score"]).astype(int)
    return scores[
        ["method", "cell_type", "regulator", "target", "score", "sign_optional"]
    ]


def evaluate_condition(
    replicate_dir: Path,
    condition: MisspecificationCondition,
) -> tuple[pd.DataFrame, dict]:
    inputs = replicate_dir / "method_inputs/RegArch"
    config = json.loads((replicate_dir / "config.json").read_text(encoding="utf-8"))
    truth = pd.read_csv(replicate_dir / "truth/true_grn_all_nodes.tsv", sep="\t")
    node_metadata = pd.read_csv(replicate_dir / "truth/node_metadata.tsv", sep="\t")
    leaf_nodes = node_metadata.loc[
        node_metadata["node_type"].eq("leaf"), "node_id"
    ].tolist()
    edge_index = _edge_index(truth)
    prior = pd.read_csv(inputs / "atac_prior_tf_target.tsv", sep="\t")
    parameters = _condition_parameters(config["grn"], condition)

    fate_path = inputs / "fate_map_edges.csv"
    fate_map_file = "method_inputs/RegArch/fate_map_edges.csv"
    mapping_path = None
    if condition.topology == "wrong":
        fate_path, mapping_path = _topology_only_fate_map(replicate_dir)
        fate_map_file = (
            "ablations/model_misspecification_inputs/"
            "topology_perturbed_fate_map_edges.csv"
        )
    edges = _normalize_edges(fate_path)
    edges["branch_length"] *= parameters["branch_length_multiplier"]
    leaf_evidence = _leaf_evidence(
        truth,
        edge_index,
        leaf_nodes,
        parameters["edge_temperature"],
        parameters["sign_temperature"],
    )
    root_probabilities, root_mean_before, root_mean_after = _root_probabilities(
        prior,
        edge_index,
        parameters["root_prior_odds_multiplier"],
    )
    root_prior = _root_state_prior(
        root_probabilities,
        parameters["positive_fraction"],
    )
    generator = _transition_generator(
        parameters["gain_rate"],
        parameters["loss_rate"],
        parameters["sign_switch_rate"],
        parameters["positive_fraction"],
    )
    posteriors = exact_tree_posteriors(edges, leaf_evidence, root_prior, generator)
    scores = _scores_from_posteriors(
        posteriors,
        edge_index,
        condition,
        mapping_path,
    )
    metrics = evaluate_scores(build_edge_universe(truth), scores)
    # sklearn metrics can exceed one by a few ulps after weighted summation.
    # Keep probability-like publication fields inside their exact schema bounds.
    metrics[["auprc", "auroc"]] = metrics[["auprc", "auroc"]].clip(0.0, 1.0)
    metrics = metrics.loc[metrics["cell_type"].ne("__mean__")].copy()
    metrics.insert(0, "condition", condition.name)
    metrics.insert(1, "factor", condition.factor)
    metrics.insert(2, "multiplier", condition.multiplier)
    metrics.insert(3, "replicate", replicate_dir.name)
    metrics = metrics.merge(
        node_metadata[["node_id", "node_type"]],
        left_on="cell_type",
        right_on="node_id",
        how="left",
        validate="many_to_one",
    ).drop(columns="node_id")
    if metrics["node_type"].isna().any():
        raise ValueError("posterior scores contain nodes absent from truth metadata")

    parameter_row = {
        "replicate": replicate_dir.name,
        "condition": condition.name,
        "factor": condition.factor,
        "multiplier": condition.multiplier,
        "lineage_gain_rate": parameters["gain_rate"],
        "lineage_loss_rate": parameters["loss_rate"],
        "lineage_sign_switch_rate": parameters["sign_switch_rate"],
        "lineage_positive_fraction": parameters["positive_fraction"],
        "lineage_edge_temperature": parameters["edge_temperature"],
        "lineage_sign_temperature": parameters["sign_temperature"],
        "root_prior": "atac",
        "root_prior_odds_multiplier": parameters["root_prior_odds_multiplier"],
        "root_prior_mean_before": root_mean_before,
        "root_prior_mean_after": root_mean_after,
        "branch_length_multiplier": parameters["branch_length_multiplier"],
        # Store a replicate-relative logical path so published result tables do
        # not depend on the machine or temporary staging directory used to run
        # the benchmark.
        "fate_map_file": fate_map_file,
    }
    return metrics, parameter_row


def run_study(replicate_dirs: list[Path], output_dir: Path) -> None:
    conditions = model_misspecification_conditions()
    metric_parts = []
    parameter_rows = []
    input_manifest_rows = []
    for replicate_dir in replicate_dirs:
        inputs = replicate_dir / "method_inputs/RegArch"
        for logical_name, path in {
            "config": replicate_dir / "config.json",
            "truth_grn": replicate_dir / "truth/true_grn_all_nodes.tsv",
            "node_metadata": replicate_dir / "truth/node_metadata.tsv",
            "atac_prior": inputs / "atac_prior_tf_target.tsv",
            "fate_map": inputs / "fate_map_edges.csv",
        }.items():
            input_manifest_rows.append(
                {
                    "replicate": replicate_dir.name,
                    "logical_input": logical_name,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        for condition in conditions:
            print(f"[{replicate_dir.name}] {condition.name}", flush=True)
            metrics, parameters = evaluate_condition(replicate_dir, condition)
            metric_parts.append(metrics)
            parameter_rows.append(parameters)
        derived_dir = replicate_dir / "ablations/model_misspecification_inputs"
        for logical_name, path in {
            "derived_topology_fate_map": (
                derived_dir / "topology_perturbed_fate_map_edges.csv"
            ),
            "derived_topology_node_mapping": (
                derived_dir / "topology_perturbed_to_truth_mapping.tsv"
            ),
        }.items():
            if not path.exists():
                raise FileNotFoundError(f"missing derived sensitivity artifact: {path}")
            input_manifest_rows.append(
                {
                    "replicate": replicate_dir.name,
                    "logical_input": logical_name,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    write_study_tables(
        pd.concat(metric_parts, ignore_index=True),
        conditions,
        parameter_rows,
        output_dir,
    )
    pd.DataFrame(input_manifest_rows).to_csv(
        output_dir / "model_misspecification_input_manifest.tsv",
        sep="\t",
        index=False,
    )
    metadata = {
        "schema_version": 2,
        "analysis": "fixed-leaf-weight ancestral model-misspecification",
        "terminal_evidence_source": "simulated true leaf GRN weights",
        "terminal_evidence_mapping": {
            "edge_threshold": EDGE_THRESHOLD,
            "reference_edge_temperature": EDGE_TEMPERATURE,
            "reference_sign_temperature": SIGN_TEMPERATURE,
        },
        "reference_condition": (
            "true Q and branch lengths with the unperturbed edge-specific "
            "mean-leaf ATAC root prior"
        ),
        "root_prior": "edge-specific mean-leaf ATAC prior",
        "root_prior_mean_definition": "mean over eligible non-self regulator-target edges",
        "condition_count": len(conditions),
        "replicate_count": len(replicate_dirs),
        "interpretation": (
            "This analysis isolates P(S_ancestors | D_leaves, T, Q, pi_root); "
            "it does not refit RNA regression coefficients."
        ),
        "undefined_metric_note": (
            "Sign accuracy or coverage is structurally undefined when a node-condition "
            "pair recovers no true edge with an available sign; primary AUPRC and "
            "signed precision at true K remain defined."
        ),
    }
    (output_dir / "model_misspecification_design.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact ancestral sensitivity with fixed simulated leaf GRN weights."
        )
    )
    parser.add_argument("replicate_dirs", type=Path, nargs="*")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help="Discover every config-bearing replicate directory under this path.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        replicate_dirs = resolve_replicate_dirs(args.replicate_dirs, args.runs_dir)
    except ValueError as error:
        parser.error(str(error))
    run_study(replicate_dirs, args.output_dir)


if __name__ == "__main__":
    main()
