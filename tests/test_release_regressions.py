import json

import numpy as np
import pandas as pd
import pytest

from benchmark.plot_model_misspecification import _paired_deltas, _quality_problems
from benchmark.run_fixed_evidence_sensitivity import (
    _root_probabilities,
    exact_tree_posteriors,
)
from benchmark.run_model_misspecification import (
    build_inference_command,
    model_misspecification_conditions,
    odds_scaled_probability,
    resolve_replicate_dirs,
)
from benchmark.run_regarch import _odds_scale, _uniform_root_probability
from regarch.cell_fate_map import FateMap, generate_newick, parse_edge_dict
from regarch.simulation import perturb_branch_lengths


def test_branch_perturbation_preserves_fate_map_nodes():
    fate_map = FateMap(
        parse_edge_dict(
            {
                "root->left": 0.4,
                "root->right": 0.2,
                "left->a": 0.6,
                "left->b": 0.6,
                "right->c": 0.8,
                "right->d": 0.8,
            }
        )
    )

    perturbed = perturb_branch_lengths(generate_newick(fate_map), sigma=0.05)

    assert set(perturbed.nodes) == set(fate_map.nodes)
    original_root = next(
        node.node_id
        for node in fate_map.nodes.values()
        if node.upstream_node_id is None
    )
    perturbed_root = next(
        node.node_id
        for node in perturbed.nodes.values()
        if node.upstream_node_id is None
    )
    assert perturbed_root == original_root


def test_model_misspecification_grid_is_compact_and_one_factor_at_a_time(tmp_path):
    conditions = model_misspecification_conditions()
    assert len(conditions) == 22
    assert conditions[0].name == "matched"
    assert len({condition.name for condition in conditions}) == len(conditions)
    assert {condition.factor for condition in conditions} >= {
        "gain_rate",
        "loss_rate",
        "sign_switch_rate",
        "positive_gain_odds",
        "root_prior_odds",
        "branch_length",
        "topology",
        "terminal_temperature",
    }

    replicate = tmp_path / "replicate"
    inputs = replicate / "method_inputs" / "RegArch"
    inputs.mkdir(parents=True)
    (replicate / "config.json").write_text(
        json.dumps(
            {
                "grn": {
                    "gain_rate": 0.02,
                    "loss_rate": 0.4,
                    "sign_switch_rate": 0.05,
                    "positive_fraction": 0.6,
                    "density": 0.1,
                }
            }
        ),
        encoding="utf-8",
    )
    (inputs / "fate_map_edges.csv").write_text(
        "Parent,Child,Length\nRoot,A,0.4\nRoot,B,0.4\n",
        encoding="utf-8",
    )
    gain = next(
        condition for condition in conditions if condition.name == "gain_rate_x2"
    )
    command, fate_map = build_inference_command(replicate, gain, 2, 100)
    assert fate_map is None
    assert command[command.index("--gain-rate") + 1] == "0.04"

    low_gain = next(
        condition for condition in conditions if condition.name == "gain_rate_x0.25"
    )
    command, _ = build_inference_command(replicate, low_gain, 2, 100)
    assert command[command.index("--gain-rate") + 1] == "0.005"

    positive_low = next(
        condition
        for condition in conditions
        if condition.name == "positive_gain_odds_x0.5"
    )
    command, _ = build_inference_command(replicate, positive_low, 2, 100)
    assert float(command[command.index("--positive-fraction") + 1]) == pytest.approx(
        3 / 7
    )

    positive_high = next(
        condition
        for condition in conditions
        if condition.name == "positive_gain_odds_x2"
    )
    command, _ = build_inference_command(replicate, positive_high, 2, 100)
    assert float(command[command.index("--positive-fraction") + 1]) == pytest.approx(
        0.75
    )

    root_prior = next(
        condition for condition in conditions if condition.name == "root_prior_odds_x2"
    )
    command, generated_input = build_inference_command(replicate, root_prior, 2, 100)
    assert generated_input is None
    assert command[command.index("--root-prior") + 1] == "atac"
    assert command[command.index("--root-prior-odds-multiplier") + 1] == "2"

    branch = next(
        condition for condition in conditions if condition.name == "branch_length_x2"
    )
    command, fate_map = build_inference_command(replicate, branch, 2, 100)
    assert "--fate-map-file" in command
    assert fate_map is not None
    assert set(pd.read_csv(fate_map)["Length"]) == {0.8}


def test_probability_odds_scaling_is_symmetric_and_rank_preserving():
    baseline = 0.6
    low = odds_scaled_probability(baseline, 0.5)
    high = odds_scaled_probability(baseline, 2.0)
    assert low == pytest.approx(3 / 7)
    assert high == pytest.approx(0.75)
    baseline_odds = baseline / (1 - baseline)
    assert low / (1 - low) == pytest.approx(0.5 * baseline_odds)
    assert high / (1 - high) == pytest.approx(2.0 * baseline_odds)

    values = pd.Series([0.0, 0.1, 0.4, 0.9, 1.0])
    scaled = _odds_scale(values, 2.0)
    assert np.all(np.diff(scaled) >= 0)
    assert scaled.iloc[0] == 0
    assert scaled.iloc[-1] == 1
    pd.testing.assert_series_equal(_odds_scale(values, 1.0), values)


def test_replicate_discovery_is_cross_platform_and_deduplicated(tmp_path):
    runs_dir = tmp_path / "runs"
    first = runs_dir / "scenario_rep_001"
    second = runs_dir / "scenario_rep_002"
    ignored = runs_dir / "notes"
    for path in (first, second, ignored):
        path.mkdir(parents=True)
    for path in (first, second):
        (path / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_replicate_dirs([first], runs_dir) == [
        first.resolve(),
        second.resolve(),
    ]
    with pytest.raises(ValueError, match="provide replicate directories"):
        resolve_replicate_dirs([], ignored)


def test_explicit_uniform_root_probability_takes_precedence():
    config = {"grn": {"density": 0.08}}
    assert _uniform_root_probability(config, None) == pytest.approx(0.08)
    assert _uniform_root_probability(config, 0.05) == pytest.approx(0.05)
    for invalid in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError):
            _uniform_root_probability(config, invalid)


def test_fixed_evidence_sum_product_matches_direct_root_calculation():
    edges = pd.DataFrame(
        {
            "parent": ["Root", "Root"],
            "child": ["A", "B"],
            "branch_length": [0.2, 0.2],
        }
    )
    generator = np.array(
        [
            [-0.5, 0.4, 0.1],
            [0.06, -0.1, 0.04],
            [0.1, 0.4, -0.5],
        ]
    )
    prior = np.array([[0.2, 0.5, 0.3]])
    evidence = {
        "A": np.array([[0.8, 0.1, 0.1]]),
        "B": np.array([[0.1, 0.2, 0.7]]),
    }
    posterior = exact_tree_posteriors(edges, evidence, prior, generator)
    from scipy.linalg import expm

    transition = expm(generator * 0.2)
    expected_root = (
        prior[0] * (transition @ evidence["A"][0]) * (transition @ evidence["B"][0])
    )
    expected_root /= expected_root.sum()
    np.testing.assert_allclose(posterior["Root"][0], expected_root)
    for probabilities in posterior.values():
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_root_prior_summary_uses_only_eligible_non_self_edges():
    edge_index = pd.DataFrame(
        {
            "target": ["g1", "g2", "g2"],
            "regulator": ["g1", "g1", "g2"],
        }
    )
    prior = pd.DataFrame(
        {
            "node": ["Root", "Root", "Root"],
            "target": ["g1", "g2", "g2"],
            "regulator": ["g1", "g1", "g2"],
            "prior_score": [0.9, 0.2, 0.8],
        }
    )
    probabilities, before, after = _root_probabilities(prior, edge_index, 2.0)
    assert before == pytest.approx(0.2)
    assert after == pytest.approx(1 / 3)
    np.testing.assert_allclose(probabilities, [0.0, 1 / 3, 0.0])


def test_plotting_uses_replicate_paired_deltas_and_quality_gate():
    rows = []
    for replicate, matched, perturbed in (("r1", 0.2, 0.3), ("r2", 0.8, 0.7)):
        rows.extend(
            [
                {
                    "replicate": replicate,
                    "condition": "matched",
                    "factor": "matched",
                    "multiplier": 1.0,
                    "node_type": "root",
                    "auprc": matched,
                    "signed_precision_at_true_k": matched,
                },
                {
                    "replicate": replicate,
                    "condition": "gain_rate_x2",
                    "factor": "gain_rate",
                    "multiplier": 2.0,
                    "node_type": "root",
                    "auprc": perturbed,
                    "signed_precision_at_true_k": perturbed,
                },
            ]
        )
    metrics = pd.DataFrame(rows)
    paired = _paired_deltas(metrics, "auprc")
    observed = paired.loc[paired["condition"].eq("gain_rate_x2"), "delta"]
    np.testing.assert_allclose(observed, [0.1, -0.1])
    assert "only 2 independent replicate(s)" in _quality_problems(metrics, None)[0]
