from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import regarch.gene_regulatory_network as grn_module
from benchmark.ablation import make_wrong_fate_map
from benchmark.config import load_config
from benchmark.diagnostics import lineage_signal_table
from benchmark.dynamic_grn import generate_dynamic_grns, transition_matrix
from benchmark.evaluation import build_edge_universe, evaluate_scores
from benchmark.fate_map import generate_balanced_fate_map
from benchmark.pipeline import generate_dataset
from benchmark.plot_leaf_benchmark import METHOD_ORDER
from benchmark.retime_fate_map import retime_fate_map
from benchmark.run_ablation_study import _quality_prior
from benchmark.run_benchmark import METHODS
from benchmark.run_random_baseline import run_random_baseline
from benchmark.standardize_external import standardize_table
from regarch.cell_fate_map import (
    FateMap,
    FateMapEdge,
    _compute_alpha,
    cell_types_distance,
    mrca_time_to_distance,
    upgma_to_edge_dict,
)
from regarch.gene_regulatory_network import _self_regulator_index
from regarch.utils.data_io import RNAData

CURRENT_CONFIG = Path("benchmark/configs/lineage_signal_10leaf_10rep.json")


class BenchmarkTests(unittest.TestCase):
    def test_mrca_times_are_converted_to_patristic_distances(self) -> None:
        mrca = np.array([[10.0, 8.0, 3.0], [8.0, 10.0, 4.0], [3.0, 4.0, 10.0]])
        observed = mrca_time_to_distance(mrca, t_S=10.0)
        expected = np.array([[0.0, 4.0, 14.0], [4.0, 0.0, 12.0], [14.0, 12.0, 0.0]])
        np.testing.assert_allclose(observed, expected)

    def test_cell_type_distance_averages_leaf_distances(self) -> None:
        mrca = np.array([[10.0, 9.0, 4.0], [9.0, 10.0, 6.0], [4.0, 6.0, 10.0]])
        cell_types = {"Cell0": "A", "Cell1": "A", "Cell2": "B"}
        observed = cell_types_distance(mrca, cell_types, t_S=10.0)
        self.assertAlmostEqual(observed.loc["A", "B"], 10.0)
        self.assertEqual(observed.loc["A", "A"], 0.0)

    def test_compute_alpha_uses_only_distinct_cell_pairs(self) -> None:
        sc_mat = np.array([[0], [1]])
        mutation_probability = np.array([[0.0], [1.0]])
        mrca = np.array([[2.0, 1.0], [1.0, 2.0]])
        observed = _compute_alpha(2.0, sc_mat, mutation_probability, 1.0, mrca)
        self.assertGreater(observed[0], 0.0)
        self.assertTrue(np.isfinite(observed[0]))

    def test_upgma_preserves_short_branch_precision(self) -> None:
        distance = pd.DataFrame(
            [[0.0, 0.252, 0.268], [0.252, 0.0, 0.268], [0.268, 0.268, 0.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        edges = upgma_to_edge_dict(distance)
        internal_edge = edges["IP2->IP1"]
        self.assertGreater(internal_edge, 0.0)
        self.assertAlmostEqual(internal_edge, (0.134 - 0.126) / 0.134)

    def test_fate_map_is_order_independent_and_assigns_node_metadata(self) -> None:
        edges = [
            FateMapEdge("R", "I", 0.25),
            FateMapEdge("R", "C", 1.0),
            FateMapEdge("I", "A", 0.75),
            FateMapEdge("I", "B", 0.75),
        ]
        expression = {
            "R": ["root_gene"],
            "I": ["internal_gene"],
            "A": ["a_gene"],
            "B": ["b_gene"],
            "C": ["c_gene"],
        }
        observed = FateMap(list(reversed(edges)), expression)
        self.assertEqual(observed.node_root, "R")
        self.assertEqual(set(observed.node_internals), {"I"})
        self.assertEqual(set(observed.node_leaves), {"A", "B", "C"})
        self.assertEqual(
            observed.nodes["R"].high_expression_genes_in_leaf,
            ["root_gene"],
        )
        self.assertEqual(
            observed.nodes["I"].high_expression_genes_in_leaf,
            ["internal_gene"],
        )

    def test_fate_map_rejects_invalid_tree_structures(self) -> None:
        invalid_cases = [
            [FateMapEdge("R", "A", 1.0)],
            [
                FateMapEdge("R", "A", 1.0),
                FateMapEdge("R", "B", 1.0),
                FateMapEdge("X", "A", 1.0),
                FateMapEdge("X", "C", 1.0),
            ],
            [
                FateMapEdge("R", "A", -1.0),
                FateMapEdge("R", "B", 1.0),
            ],
        ]
        for edges in invalid_cases:
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                FateMap(edges)

    def test_upgma_rejects_invalid_distance_matrices(self) -> None:
        invalid = [
            [[0.0, 1.0], [2.0, 0.0]],
            [[0.0, -1.0], [-1.0, 0.0]],
            [[0.0, np.nan], [np.nan, 0.0]],
            [[1.0, 2.0], [2.0, 0.0]],
        ]
        for values in invalid:
            frame = pd.DataFrame(values, index=["A", "B"], columns=["A", "B"])
            with self.subTest(values=values), self.assertRaises(ValueError):
                upgma_to_edge_dict(frame)

    def test_mrca_distance_rejects_asymmetric_input(self) -> None:
        mrca = np.array([[10.0, 8.0], [7.0, 10.0]])
        with self.assertRaisesRegex(ValueError, "symmetric"):
            mrca_time_to_distance(mrca, t_S=10.0)

    def test_cross_clade_wrong_map_splits_all_sibling_pairs(self) -> None:
        config = load_config(CURRENT_CONFIG)
        fate_map = generate_balanced_fate_map(
            config.fate_map.node_count,
            config.fate_map.branch_lengths,
        )
        wrong = make_wrong_fate_map(fate_map, mode="cross_clade_split")
        true_leaf_edges = fate_map.edges.loc[
            fate_map.edges["child"].isin(fate_map.leaf_nodes),
            ["parent", "child"],
        ]
        wrong_leaf_edges = wrong.edges.loc[
            wrong.edges["child"].isin(fate_map.leaf_nodes),
            ["parent", "child"],
        ]
        true_parent = true_leaf_edges.set_index("child")["parent"]
        wrong_parent = wrong_leaf_edges.set_index("child")["parent"]
        self.assertTrue((true_parent != wrong_parent).all())
        self.assertTrue((wrong_leaf_edges.groupby("parent").size() == 2).all())
        for _, siblings in true_leaf_edges.groupby("parent"):
            assigned = wrong_parent.loc[siblings["child"]]
            self.assertEqual(assigned.nunique(), 2)

    def test_cross_clade_short_branch_wrong_map_compresses_lengths(self) -> None:
        config = load_config(CURRENT_CONFIG)
        fate_map = generate_balanced_fate_map(
            config.fate_map.node_count,
            config.fate_map.branch_lengths,
        )
        wrong = make_wrong_fate_map(fate_map, mode="cross_clade_short_branches")
        self.assertTrue((wrong.edges["branch_length"] == 0.001).all())

    def test_controlled_atac_prior_matches_requested_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            replicate = Path(temporary)
            inputs = replicate / "method_inputs/RegArch"
            truth_dir = replicate / "truth"
            inputs.mkdir(parents=True)
            truth_dir.mkdir()
            rows = []
            truth_rows = []
            for regulator in ["TF1", "TF2"]:
                for target in ["G1", "G2"]:
                    rows.append(
                        {
                            "node": "T1",
                            "regulator": regulator,
                            "target": target,
                            "prior_score": 0.5,
                        }
                    )
                    truth_rows.append(
                        {
                            "node": "T1",
                            "regulator": regulator,
                            "target": target,
                            "true_edge": int(regulator == "TF1"),
                        }
                    )
            pd.DataFrame(rows).to_csv(
                inputs / "atac_prior_tf_target.tsv", sep="\t", index=False
            )
            pd.DataFrame(truth_rows).to_csv(
                truth_dir / "true_grn_all_nodes.tsv", sep="\t", index=False
            )
            _, statistics = _quality_prior(
                replicate, coverage=0.5, correctness=0.5, seed=7
            )
            self.assertEqual(statistics["actual_coverage"], 0.5)
            self.assertEqual(statistics["actual_correctness"], 0.5)

    def test_self_regulator_index_enforces_structural_exclusion(self) -> None:
        self.assertEqual(
            _self_regulator_index(["TF1", "TF2"], "TF2"),
            1,
        )
        self.assertIsNone(_self_regulator_index(["TF1", "TF2"], "Gene1"))

    def test_transition_matrix_is_stochastic(self) -> None:
        config = load_config(CURRENT_CONFIG)
        matrix = transition_matrix(0.4, config.grn)
        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(3))
        self.assertTrue((matrix >= 0).all())

    def test_generated_grns_exclude_self_regulation_by_default(self) -> None:
        config = load_config(CURRENT_CONFIG)
        fate_map = generate_balanced_fate_map(
            config.fate_map.node_count, config.fate_map.branch_lengths
        )
        truth = generate_dynamic_grns(fate_map, config.grn, np.random.default_rng(123))
        for matrix in truth.matrices.values():
            diagonal = matrix.to_numpy()[
                np.arange(config.grn.regulator_count),
                np.arange(config.grn.regulator_count),
            ]
            np.testing.assert_array_equal(diagonal, 0)

    def test_edge_universe_excludes_self_regulation(self) -> None:
        truth = pd.DataFrame(
            {
                "node": ["T1", "T1"],
                "regulator": ["TF001", "TF001"],
                "target": ["TF001", "Gene001"],
                "true_edge": [0, 1],
            }
        )
        universe = build_edge_universe(truth)
        self.assertEqual(universe["target"].tolist(), ["Gene001"])

    def test_presence_transition_matches_ctmc_rates(self) -> None:
        config = load_config(CURRENT_CONFIG)
        signed = transition_matrix(0.4, config.grn)
        presence = np.array(
            [
                [signed[1, 1], signed[1, 0] + signed[1, 2]],
                [signed[0, 1], signed[0, 0] + signed[0, 2]],
            ]
        )
        generator = np.array(
            [
                [-config.grn.gain_rate, config.grn.gain_rate],
                [config.grn.loss_rate, -config.grn.loss_rate],
            ]
        )
        from scipy.linalg import expm

        np.testing.assert_allclose(presence, expm(generator * 0.4))

    def test_retimed_fate_map_is_ultrametric(self) -> None:
        topology = pd.DataFrame(
            {
                "parent": ["Root", "Root", "P1", "P1", "P2", "P2"],
                "child": ["P1", "P2", "T1", "T2", "T3", "T4"],
            }
        )
        node_times = {
            "Root": 0.0,
            "P1": 0.2,
            "P2": 0.8,
            "T1": 1.0,
            "T2": 1.0,
            "T3": 1.0,
            "T4": 1.0,
        }
        edges, nodes = retime_fate_map(topology, node_times)
        leaf_times = nodes.loc[nodes["is_leaf"], "developmental_time"]
        self.assertTrue((leaf_times == 1.0).all())
        self.assertTrue((edges["branch_length"] > 0).all())
        self.assertEqual(
            edges["branch_length"].round(8).tolist(),
            [0.2, 0.8, 0.8, 0.8, 0.2, 0.2],
        )

    def test_lineage_diagnostics_include_leaf_pairs(self) -> None:
        config = load_config(CURRENT_CONFIG)
        fate_map = generate_balanced_fate_map(
            config.fate_map.node_count, config.fate_map.branch_lengths
        )
        truth = generate_dynamic_grns(fate_map, config.grn, np.random.default_rng(123))
        diagnostics = lineage_signal_table(fate_map, truth, config.grn)
        pairs = diagnostics.loc[diagnostics["record_type"] == "leaf_pair"]
        leaf_count = len(fate_map.leaf_nodes)
        self.assertEqual(len(pairs), leaf_count * (leaf_count - 1) // 2)
        self.assertEqual(set(pairs["relationship"]), {"sibling", "cross_clade"})

    def test_matched_lineage_prior_is_smooth(self) -> None:
        class DummyFateMap:
            node_leaves = ["T1", "T2"]
            node_root = "Root"
            node_internals = []

        inference = object.__new__(grn_module.GRNInference)
        inference.fate_map = DummyFateMap()
        ancestor_space = np.array([[0], [1]], dtype=int)
        transition = np.array([[0.98, 0.02], [0.15, 0.85]])
        transitions = [(0, 1, transition), (0, 2, transition)]
        root = np.array([0.5])
        low = inference._lineage_prior_ctmc(
            np.array([0.09, 0.09]), root, ancestor_space, transitions
        )
        high = inference._lineage_prior_ctmc(
            np.array([0.11, 0.11]), root, ancestor_space, transitions
        )
        self.assertNotEqual(low, high)

    def test_atac_prior_is_smooth_near_edge_threshold(self) -> None:
        class DummyFateMap:
            node_leaves = ["T1"]

        inference = object.__new__(grn_module.GRNInference)
        inference.fate_map = DummyFateMap()
        inference.rna_data = SimpleNamespace(regulator_genes=["TF1"])
        log_probabilities = np.array([[[-0.1, -2.0]]])
        low = inference._atac_prior(np.array([0.099]), log_probabilities)
        high = inference._atac_prior(np.array([0.101]), log_probabilities)
        self.assertNotEqual(low, high)
        self.assertLess(low, high)

    def test_worker_initializer_preserves_runner_settings(self) -> None:
        original_model = grn_module.LINEAGE_TRANSITION_MODEL
        original_gain = grn_module.LINEAGE_GAIN_RATE
        original_maxfun = grn_module.OPTIMIZE_MAX_FUNCTION_EVALUATIONS
        try:
            grn_module._initialize_inference_worker(
                {
                    "LINEAGE_TRANSITION_MODEL": "ctmc_presence",
                    "LINEAGE_GAIN_RATE": 0.123,
                    "OPTIMIZE_MAX_FUNCTION_EVALUATIONS": 30000,
                }
            )
            self.assertEqual(
                grn_module.LINEAGE_TRANSITION_MODEL,
                "ctmc_presence",
            )
            self.assertEqual(grn_module.LINEAGE_GAIN_RATE, 0.123)
            self.assertEqual(
                grn_module.OPTIMIZE_MAX_FUNCTION_EVALUATIONS,
                30000,
            )
        finally:
            grn_module.LINEAGE_TRANSITION_MODEL = original_model
            grn_module.LINEAGE_GAIN_RATE = original_gain
            grn_module.OPTIMIZE_MAX_FUNCTION_EVALUATIONS = original_maxfun

    def test_parent_inference_uses_both_branch_probabilities(self) -> None:
        edges = [
            SimpleNamespace(end="T1", weight=0.2),
            SimpleNamespace(end="T2", weight=0.8),
        ]
        values = {
            "T1": [0.4, 0.2, 0.5],
            "T2": [0.6, 0.8, -0.4],
        }
        inferred = grn_module._parent_grn_inference(edges, values, 2)
        probability1 = np.exp(-0.4 * 0.2)
        probability2 = np.exp(-0.6 * 0.8)
        expected = (probability1 * 0.2 + probability2 * 0.8) / (
            probability1 + probability2
        )
        self.assertEqual(len(inferred), 3)
        self.assertAlmostEqual(inferred[1], expected)
        self.assertEqual(inferred[2], 0.5)

    def test_ctmc_ancestor_posterior_uses_leaf_evidence(self) -> None:
        class DummyFateMap:
            node_leaves = ["T1", "T2"]
            node_root = "Root"
            node_internals = []

        inference = object.__new__(grn_module.GRNInference)
        inference.fate_map = DummyFateMap()
        inference.rna_data = SimpleNamespace(regulator_genes=["TF1"])
        ancestor_space = np.array([[0], [1]], dtype=int)
        transition = np.array([[0.98, 0.02], [0.15, 0.85]])
        transitions = [(0, 1, transition), (0, 2, transition)]
        original_mode = grn_module.LINEAGE_ROOT_PRIOR_MODE
        try:
            grn_module.LINEAGE_ROOT_PRIOR_MODE = "uniform"
            low = inference._ancestor_posteriors_ctmc(
                np.array([0.0, 0.0]),
                np.array([0.9]),
                ancestor_space,
                transitions,
            )
            high = inference._ancestor_posteriors_ctmc(
                np.array([0.8, 0.8]),
                np.array([0.1]),
                ancestor_space,
                transitions,
            )
        finally:
            grn_module.LINEAGE_ROOT_PRIOR_MODE = original_mode
        self.assertGreater(high[0, 0], low[0, 0])
        self.assertTrue(0 <= low[0, 0] <= 1)
        self.assertTrue(0 <= high[0, 0] <= 1)

    def test_signed_ctmc_transition_matches_simulator(self) -> None:
        class DummyFateMap:
            node_root = "Root"
            node_leaves = ["T1"]
            node_internals = []
            nodes = {"Root": object(), "T1": object()}
            edges = [SimpleNamespace(start="Root", end="T1", weight=0.4)]

        inference = object.__new__(grn_module.GRNInference)
        inference.fate_map = DummyFateMap()
        original = {
            "gain": grn_module.LINEAGE_GAIN_RATE,
            "loss": grn_module.LINEAGE_LOSS_RATE,
            "switch": grn_module.LINEAGE_SIGN_SWITCH_RATE,
            "positive": grn_module.LINEAGE_POSITIVE_FRACTION,
        }
        config = load_config(CURRENT_CONFIG)
        try:
            grn_module.LINEAGE_GAIN_RATE = config.grn.gain_rate
            grn_module.LINEAGE_LOSS_RATE = config.grn.loss_rate
            grn_module.LINEAGE_SIGN_SWITCH_RATE = config.grn.sign_switch_rate
            grn_module.LINEAGE_POSITIVE_FRACTION = config.grn.positive_fraction
            inferred = inference._reshape_lineage_signed_ctmc_input()
            observed = inferred["children"]["Root"][0][1]
            expected = transition_matrix(0.4, config.grn)
            np.testing.assert_allclose(observed, expected)
        finally:
            grn_module.LINEAGE_GAIN_RATE = original["gain"]
            grn_module.LINEAGE_LOSS_RATE = original["loss"]
            grn_module.LINEAGE_SIGN_SWITCH_RATE = original["switch"]
            grn_module.LINEAGE_POSITIVE_FRACTION = original["positive"]

    def test_signed_global_posterior_recovers_leaf_polarity(self) -> None:
        class DummyFateMap:
            node_root = "Root"
            node_leaves = ["T1", "T2"]
            node_internals = []
            nodes = {"Root": object(), "T1": object(), "T2": object()}
            edges = []

        inference = object.__new__(grn_module.GRNInference)
        inference.fate_map = DummyFateMap()
        transition = np.array(
            [
                [0.94, 0.05, 0.01],
                [0.01, 0.97, 0.02],
                [0.01, 0.05, 0.94],
            ]
        )
        tree = {
            "root": "Root",
            "children": {
                "Root": [("T1", transition), ("T2", transition)],
                "T1": [],
                "T2": [],
            },
            "parents": {},
        }
        original_mode = grn_module.LINEAGE_ROOT_PRIOR_MODE
        try:
            grn_module.LINEAGE_ROOT_PRIOR_MODE = "uniform"
            posteriors = inference._signed_node_posteriors_ctmc(
                np.array([0.8, 0.9]),
                np.array([0.1]),
                tree,
            )
        finally:
            grn_module.LINEAGE_ROOT_PRIOR_MODE = original_mode
        self.assertGreater(
            posteriors["Root"][0, 2],
            posteriors["Root"][0, 0],
        )
        self.assertGreater(
            posteriors["T1"][0, 2],
            posteriors["T1"][0, 0],
        )
        for node_posterior in posteriors.values():
            np.testing.assert_allclose(node_posterior.sum(axis=1), 1.0)

    def test_pilot_generation(self) -> None:
        config = load_config(CURRENT_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            replicate = generate_dataset(config, tmp)
            self.assertTrue((replicate / "truth/true_grn_all_nodes.tsv").exists())
            self.assertTrue(
                (replicate / "intermediate/evaluation/edge_universe.tsv").exists()
            )
            self.assertTrue(
                (replicate / "method_inputs/RegArch/expression_data.csv").exists()
            )
            self.assertTrue(
                (replicate / "diagnostics/lineage_signal_summary.tsv").exists()
            )
            metadata = pd.read_csv(
                replicate / "intermediate/metadata/cell_metadata.tsv", sep="\t"
            )
            self.assertEqual(
                metadata["cell_type"].nunique(),
                (config.fate_map.node_count + 1) // 2,
            )

            rna = RNAData(replicate / "method_inputs/RegArch/expression_data.csv")
            for node_id in rna.node_ids:
                self.assertEqual(
                    len(rna.get_values(grn_module.REGULATOR_GENE, node_id)),
                    config.observations.cells_per_leaf,
                )

    def test_sparse_method_output_is_evaluated_on_full_universe(self) -> None:
        universe = pd.DataFrame(
            {
                "cell_type": ["T1"] * 4,
                "regulator": ["TF1", "TF1", "TF2", "TF2"],
                "target": ["G1", "G2", "G1", "G2"],
                "true_edge": [1, 0, 0, 1],
                "true_sign": [1, 0, 0, -1],
                "true_weight": [1.0, 0.0, 0.0, -1.0],
            }
        )
        scores = pd.DataFrame(
            {
                "method": ["M"],
                "cell_type": ["T1"],
                "regulator": ["TF1"],
                "target": ["G1"],
                "score": [1.0],
            }
        )
        metrics = evaluate_scores(universe, scores)
        row = metrics.loc[metrics["cell_type"] == "T1"].iloc[0]
        self.assertEqual(row["true_edge_count"], 2)
        self.assertAlmostEqual(row["recall_at_true_k"], 0.5)

    def test_signed_metrics_use_only_reported_signs(self) -> None:
        universe = pd.DataFrame(
            {
                "cell_type": ["T1"] * 3,
                "regulator": ["TF1", "TF2", "TF3"],
                "target": ["G1", "G1", "G1"],
                "true_edge": [1, 1, 0],
                "true_sign": [1, -1, 0],
            }
        )
        scores = pd.DataFrame(
            {
                "method": ["M"] * 3,
                "cell_type": ["T1"] * 3,
                "regulator": ["TF1", "TF2", "TF3"],
                "target": ["G1", "G1", "G1"],
                "score": [0.9, 0.8, 0.1],
                "sign_optional": [1, 1, 0],
            }
        )
        row = evaluate_scores(universe, scores).query("cell_type == 'T1'").iloc[0]
        self.assertAlmostEqual(row["signed_precision_at_true_k"], 0.5)
        self.assertAlmostEqual(row["signed_recall_at_true_k"], 0.5)
        self.assertAlmostEqual(
            row["sign_accuracy_on_recovered_true_edges"],
            0.5,
        )

    def test_figr_output_standardization(self) -> None:
        table = pd.DataFrame({"Motif": ["TF1"], "DORC": ["G1"], "Score": [-2.5]})
        spec = {
            "columns": {
                "regulator": ["Motif"],
                "target": ["DORC"],
                "score": ["Score"],
                "sign": ["Score"],
            }
        }
        result = standardize_table(table, "FigR", "T1", spec)
        self.assertEqual(result.loc[0, "score"], 2.5)
        self.assertEqual(result.loc[0, "sign_optional"], -1)

    def test_duplicate_standardization_keeps_selected_score_sign(self) -> None:
        table = pd.DataFrame(
            {
                "TF": ["TF1", "TF1"],
                "target": ["G1", "G1"],
                "score": [0.2, 0.9],
                "correlation": [-0.5, 0.4],
            }
        )
        spec = {
            "columns": {
                "regulator": ["TF"],
                "target": ["target"],
                "score": ["score"],
                "sign": ["correlation"],
            }
        }
        result = standardize_table(table, "Example", "T1", spec)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.loc[0, "score"], 0.9)
        self.assertEqual(result.loc[0, "sign_optional"], 1)

    def test_scenic_uses_cis_weighted_score(self) -> None:
        table = pd.DataFrame(
            {
                "TF": ["TF1"],
                "target": ["G1"],
                "importance": [4.0],
                "score": [1.5],
                "correlation": [-0.2],
            }
        )
        spec = {
            "columns": {
                "regulator": ["TF"],
                "target": ["target"],
                "score": ["score", "importance"],
                "sign": ["correlation"],
            }
        }
        result = standardize_table(table, "SCENIC", "T1", spec)
        self.assertEqual(result.loc[0, "score"], 1.5)
        self.assertEqual(result.loc[0, "sign_optional"], -1)

    def test_plot_method_order_contains_all_requested_methods(self) -> None:
        self.assertEqual(len(METHOD_ORDER), 13)
        self.assertEqual(METHOD_ORDER[0], "RegArch")
        self.assertEqual(set(METHOD_ORDER), set(METHODS))

    def test_generated_inputs_exclude_unrequested_screni(self) -> None:
        config = load_config(CURRENT_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            replicate = generate_dataset(config, tmp)
            self.assertFalse((replicate / "method_inputs/ScReNI").exists())

    def test_random_baseline_covers_full_edge_universe(self) -> None:
        config = load_config(CURRENT_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            replicate = generate_dataset(config, tmp)
            random_scores = run_random_baseline(replicate, seed=123)
            universe = pd.read_csv(
                replicate / "intermediate/evaluation/edge_universe.tsv",
                sep="\t",
            )
            self.assertEqual(len(random_scores), len(universe))
            self.assertEqual(set(random_scores["method"]), {"Random"})
            self.assertTrue(random_scores["score"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
