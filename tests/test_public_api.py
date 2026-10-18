"""Focused regression tests for the cleaned public RegArch API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regarch.downstream_analysis import identify_fate_bias_regulators
from regarch.simulation import generate_expression_data
from regarch.simulation import sergio as sergio_module
from regarch.utils.data_io import read_target_network
from regarch.utils.metrics import compute_auprc, compute_auroc


def test_grn_metrics_preserve_continuous_prediction_ranking() -> None:
    truth = pd.DataFrame([[1.0, 0.0], [1.0, 0.0]])
    inferred = pd.DataFrame([[0.9, 0.2], [0.8, 0.1]])

    assert compute_auroc(inferred, truth, threshold=0.05) == pytest.approx(1.0)
    assert compute_auprc(inferred, truth, threshold=0.05) == pytest.approx(1.0)


def test_grn_metrics_reject_single_class_truth() -> None:
    truth = pd.DataFrame(np.ones((2, 2)))
    inferred = pd.DataFrame(np.arange(4).reshape(2, 2))

    with pytest.raises(ValueError, match="both positive and negative"):
        compute_auroc(inferred, truth, threshold=0.05)


def test_fate_bias_scores_are_computed_independently_for_each_branch() -> None:
    networks = {
        "ancestor": pd.DataFrame(
            {"TF1": [1.0, 0.1]}, index=["branch_1_gene", "branch_2_gene"]
        )
    }

    branch_1, branch_2 = identify_fate_bias_regulators(
        networks,
        "ancestor",
        ["branch_1_gene"],
        ["branch_2_gene"],
        ["TF1"],
    )

    assert branch_1 is not branch_2
    assert branch_1.loc[0, "FBI"] == pytest.approx(1 / 1.1)
    assert branch_2.loc[0, "FBI"] == pytest.approx(0.1 / 1.1)


def test_target_network_reader_validates_and_parses_rows(tmp_path) -> None:
    target_file = tmp_path / "target_gene_G1.csv"
    target_file.write_text(
        "['G1', 'Root', 0.5, 0.2, -0.3]\n" "['G1', 'Leaf', 0.5, 0.4, 0.0]\n",
        encoding="utf-8",
    )

    observed = read_target_network(target_file, ["TF1", "TF2"])

    assert observed["node_id"].tolist() == ["Root", "Leaf"]
    assert observed.loc[0, "TF2"] == pytest.approx(-0.3)


def test_expression_simulation_uses_the_sergio_public_interface(
    tmp_path, monkeypatch
) -> None:
    class StubSimulator:
        graph_arguments = None

        def __init__(self, **_kwargs):
            pass

        def build_graph(self, input_file_targets, input_file_regs, shared_coop_state=0):
            self.__class__.graph_arguments = (
                input_file_targets,
                input_file_regs,
                shared_coop_state,
            )

        def simulate(self):
            pass

        def get_expressions(self):
            return np.ones((1, 3, 2))

    monkeypatch.setattr(sergio_module, "SergioSimulator", StubSimulator)
    config_dir = tmp_path / "sergio"
    output_dir = tmp_path / "expression"
    networks = {"grn0": pd.DataFrame([[0.5], [-0.2]])}

    generate_expression_data(
        networks,
        nodes_number=1,
        target_number=1,
        regulator_number=2,
        cells_number=2,
        sergio_files_path=str(config_dir),
        output_path=str(output_dir),
    )

    assert StubSimulator.graph_arguments == (
        str(config_dir / "File1_0.txt"),
        str(config_dir / "File2.txt"),
        0,
    )
    assert np.loadtxt(output_dir / "Exp_grn0.csv", delimiter=",").shape == (3, 2)
