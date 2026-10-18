import itertools
import json
import os
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.special import expit

from regarch.cell_fate_map import FateMap, FateMapEdge
from regarch.utils.constants import (
    ATAC_PRIOR_WEIGHT,
    BETA_1,
    BETA_2,
    LAMBDA_1,
    LAMBDA_HIGH_BOUND,
    LAMBDA_LOW_BOUND,
    LIKELIHOOD_WEIGHT,
    LINEAGE_EDGE_TEMPERATURE,
    LINEAGE_EDGE_THRESHOLD,
    LINEAGE_GAIN_RATE,
    LINEAGE_LOSS_RATE,
    LINEAGE_POSITIVE_FRACTION,
    LINEAGE_PRIOR_WEIGHT,
    LINEAGE_ROOT_PRIOR_MODE,
    LINEAGE_ROOT_PRIOR_PROBABILITY,
    LINEAGE_SIGN_SWITCH_RATE,
    LINEAGE_SIGN_TEMPERATURE,
    LINEAGE_TRANSITION_MODEL,
    OPTIMIZE_MAX_FUNCTION_EVALUATIONS,
    OPTIMIZE_METHOD,
    REGULATION_STRENGTH_HIGH_BOUND,
    REGULATION_STRENGTH_LOW_BOUND,
    REGULATOR_GENE,
    SIGMA,
    TARGET_GENE,
)
from regarch.utils.data_io import ATACData, ATACSample, RNAData, interaction_key
from regarch.utils.logging_utils import get_logger

CPU_NUM = cpu_count()
logger = get_logger()

_WORKER_SETTING_NAMES = [
    "OPTIMIZE_METHOD",
    "OPTIMIZE_MAX_FUNCTION_EVALUATIONS",
    "LINEAGE_TRANSITION_MODEL",
    "LINEAGE_ROOT_PRIOR_MODE",
    "LINEAGE_ROOT_PRIOR_PROBABILITY",
    "LINEAGE_PRIOR_WEIGHT",
    "ATAC_PRIOR_WEIGHT",
    "LIKELIHOOD_WEIGHT",
    "LINEAGE_GAIN_RATE",
    "LINEAGE_LOSS_RATE",
    "LINEAGE_SIGN_SWITCH_RATE",
    "LINEAGE_POSITIVE_FRACTION",
    "LINEAGE_EDGE_THRESHOLD",
    "LINEAGE_EDGE_TEMPERATURE",
    "LINEAGE_SIGN_TEMPERATURE",
]


def _worker_settings() -> dict:
    return {name: globals()[name] for name in _WORKER_SETTING_NAMES}


def _initialize_inference_worker(settings: dict) -> None:
    globals().update(settings)


def _get_grn_initial_data(
    atac_node_data: "Dict[str, ATACSample]",
    reg_genes: "List[str]",
    target_gene_id: "List[str]",
):
    """
    Initialize gene regulatory network data using ATAC node data.

    Parameters:
    atac_node_data (Dict[str, ATACSample]): A dictionary mapping keys to ATACSample objects.
    reg_genes (List[str]): List of regulatory gene identifiers.
    target_gene_id (List[str]): List of target gene identifiers.

    Returns:
    np.ndarray: An array of initial GRN values for the given genes.
    """
    data_indexer = {idx: gene_id for idx, gene_id in enumerate(reg_genes)}
    data = np.zeros(shape=len(reg_genes), dtype=np.float32)
    for idx, _ in enumerate(data):
        key = interaction_key(data_indexer[idx], target_gene_id)
        data[idx] = atac_node_data.get(key).value if atac_node_data.get(key) else 0.0

    return data


def _self_regulator_index(
    regulator_genes: "List[str]",
    target_gene_id: "str",
):
    """Return the prohibited self-edge index when the target is a regulator."""
    try:
        return regulator_genes.index(target_gene_id)
    except ValueError:
        return None


def _generate_ancestor_state_space(ancestor_nodes: "List[str]"):
    """
    Generate state space array for ancestor nodes with binary combinations.

    Parameters:
    ancestor_nodes (List[str]): List of ancestor node identifiers.

    Returns:
    np.ndarray: Array representing all possible binary state combinations of ancestor nodes.
    """
    results = np.array(
        [
            [value for _, value in zip(ancestor_nodes, combination)]
            for combination in itertools.product([0, 1], repeat=len(ancestor_nodes))
        ]
    )
    return results


def _parent_grn_inference(
    edges: "List[FateMapEdge]",
    grn_values: "Dict[str, List[float]]",
    regulator_gene_nums: "int",
):
    """
    Infer the parent gene regulatory network values based on child edges and GRN values.

    Parameters:
    edges (List[FateMapEdge]): List of edges from parent to children.
    grn_values (Dict[str, List[float]]): Dictionary containing GRN values for child nodes.
    regulator_gene_nums (int): Number of regulator genes.

    Returns:
    List[float]: Inferred GRN values for the parent node.
    """
    child_weight_1 = edges[0].weight
    child_lambda_1 = grn_values[edges[0].end][0]
    child_grn_value_1 = grn_values[edges[0].end][1:]

    child_weight_2 = edges[1].weight
    child_lambda_2 = grn_values[edges[1].end][0]
    child_grn_value_2 = grn_values[edges[1].end][1:]

    parent_grn_value = []

    for j in range(regulator_gene_nums):
        prob1 = float(np.exp(-child_lambda_1 * child_weight_1))
        prob2 = float(np.exp(-child_lambda_2 * child_weight_2))
        value1 = child_grn_value_1[j]
        value2 = child_grn_value_2[j]
        if value1 * value2 <= 0:
            parent_grn_value.append(value1 if prob1 > prob2 else value2)
        else:
            denominator = prob1 + prob2
            parent_grn_value.append(
                prob1 / denominator * value1 + prob2 / denominator * value2
            )

    parent_lambda = (child_lambda_1 + child_lambda_2) / 2
    return [parent_lambda] + parent_grn_value


class GRNInference:
    """Infer leaf and ancestral gene-regulatory networks along a fate map."""

    def __init__(
        self,
        atac_file_path,
        expression_file_path,
        fate_map: "FateMap",
        saved_dir: "str",
    ):
        """
        Initialize GRNInference with file paths, fate map, and directory for saving results.

        Parameters:
        atac_file_path: Path to the ATAC data file.
        expression_file_path: Path to the expression data file.
        fate_map (FateMap): Fate map object containing network structure.
        saved_dir (str): Directory path to save inference results.
        """
        self.fate_map = fate_map
        self.atac_data = ATACData(atac_file_path)
        self.rna_data = RNAData(expression_file_path)
        self.saved_dir = saved_dir

    def _likelihood(self, leaves_grn_data, regulator_data, target_data):
        """
        Calculate the likelihood of GRN parameters given gene expression data.

        Parameters:
        leaves_grn_data: Array of GRN data for leaf nodes.
        regulator_data: Array of regulator gene expression data.
        target_data: Array of target gene expression data.

        Returns:
        float: Calculated likelihood value weighted by a constant.
        """
        weights = np.reshape(
            leaves_grn_data,
            [len(self.fate_map.node_leaves), len(self.rna_data.regulator_genes)],
        )
        squared_error = 0.0
        for leaf_index, (design, response) in enumerate(
            zip(regulator_data, target_data)
        ):
            residual = response[:, 0] - design @ weights[leaf_index]
            squared_error += float(residual @ residual)
        exp_sum = -0.5 * SIGMA * squared_error / len(self.rna_data.cell_ids)
        constants_sum = (
            -0.5
            * len(self.rna_data.regulator_genes)
            * np.log(2 * np.pi)
            * len(self.fate_map.node_leaves)
        )
        result = exp_sum + constants_sum
        return result * LIKELIHOOD_WEIGHT

    def _atac_prior(self, leaves_grn_data: "np.ndarray", mdata: "str"):
        """
        Calculate ATAC prior probability based on GRN and metadata.

        Parameters:
        leaves_grn_data (np.ndarray): Array of GRN data for leaf nodes.
        mdata (str): Processed metadata related to ATAC.

        Returns:
        float: Prior probability weighted by a constant.
        """
        smooth_magnitudes = np.sqrt(np.square(leaves_grn_data) + 1e-8)
        edge_presence = expit(
            (smooth_magnitudes - LINEAGE_EDGE_THRESHOLD) / LINEAGE_EDGE_TEMPERATURE
        )
        edge_states = np.stack((edge_presence, 1 - edge_presence), axis=-1)
        G = np.reshape(
            edge_states,
            [
                len(self.fate_map.node_leaves),
                len(self.rna_data.regulator_genes),
                2,
            ],
        )
        p = np.sum(np.einsum("ijk,ijk -> ij", mdata, G))
        return p * ATAC_PRIOR_WEIGHT

    def _lineage_prior(
        self,
        leaves_grn_data: "np.ndarray",
        lambda_1: "float",
        root_grn_data: "np.ndarray",
        ancestor_nodes_space: "np.ndarray",
        weights: "np.ndarray",
    ):
        """
        Calculate lineage prior probability based on GRN data across the lineage tree.

        Parameters:
        leaves_grn_data (np.ndarray): Array of GRN data for leaf nodes.
        lambda_1 (float): Hyperparameter controlling weight adjustments.
        root_grn_data (np.ndarray): GRN data at the root node.
        ancestor_nodes_space (np.ndarray): Possible state space of ancestor nodes.
        weights (np.ndarray): Weights array for lineage prior computation.

        Returns:
        float: Logarithm of the lineage prior probability weighted by a constant.
        """
        lambda_1 = np.clip(lambda_1, LAMBDA_LOW_BOUND, LAMBDA_HIGH_BOUND)
        weights = np.power(weights, lambda_1)
        m = 0
        leaf_node_ids = self.fate_map.node_leaves
        ancestor_node_ids = [self.fate_map.node_root] + self.fate_map.node_internals

        leaves_grn_data = np.reshape(leaves_grn_data, (len(leaf_node_ids), -1)).T

        root_flags = ancestor_nodes_space[
            :, ancestor_node_ids.index(self.fate_map.node_root)
        ]
        root_flags = np.concatenate(
            [root_flags[..., np.newaxis], 1 - root_flags[..., np.newaxis]], axis=-1
        )

        leaf_node_space = np.repeat(
            leaves_grn_data[:, np.newaxis, ...], len(ancestor_nodes_space), axis=1
        )
        ancestor_nodes_spaces = np.repeat(
            ancestor_nodes_space[np.newaxis, ...],
            len(self.rna_data.regulator_genes),
            axis=0,
        )
        nodes_space = np.array(
            np.abs(np.concatenate((ancestor_nodes_spaces, leaf_node_space), axis=-1))
            > 0.1
        )
        nodes_space = np.int32(
            nodes_space[..., np.newaxis] == nodes_space[..., np.newaxis, :]
        )[..., np.newaxis]
        nodes_space = np.concatenate((nodes_space, 1 - nodes_space), axis=-1)
        values = np.einsum("rnijk,nijk -> rnij", nodes_space, weights)
        values = np.reshape(
            values + (values == 0),
            (len(self.rna_data.regulator_genes), len(ancestor_nodes_space), -1),
        )
        values = np.prod(values, axis=-1)[..., np.newaxis]
        joint_probility = np.einsum("r,rjk -> rjk", abs(root_grn_data), values)

        root_prior1 = np.concatenate([joint_probility, (1 - joint_probility)], axis=-1)
        p = np.einsum("rjk,jk -> rj", root_prior1, root_flags)
        m = np.sum(np.log(np.sum(p, axis=-1) + 1e-10))
        return m * LINEAGE_PRIOR_WEIGHT

    def _root_edge_probabilities(self, root_grn_data: "np.ndarray") -> "np.ndarray":
        if LINEAGE_ROOT_PRIOR_MODE == "uniform":
            probabilities = np.full(
                len(root_grn_data),
                LINEAGE_ROOT_PRIOR_PROBABILITY,
                dtype=float,
            )
        else:
            probabilities = np.abs(root_grn_data)
        return np.clip(probabilities, 1e-6, 1 - 1e-6)

    def _signed_state_probabilities(
        self,
        grn_data: "np.ndarray",
    ) -> "np.ndarray":
        """Map continuous edge weights to smooth probabilities for -1, 0, +1."""
        values = np.asarray(grn_data, dtype=float)
        magnitudes = np.sqrt(np.square(values) + 1e-8)
        present = expit(
            (magnitudes - LINEAGE_EDGE_THRESHOLD) / LINEAGE_EDGE_TEMPERATURE
        )
        positive_given_present = expit(values / LINEAGE_SIGN_TEMPERATURE)
        negative = present * (1.0 - positive_given_present)
        zero = 1.0 - present
        positive = present * positive_given_present
        probabilities = np.stack((negative, zero, positive), axis=-1)
        return probabilities / probabilities.sum(axis=-1, keepdims=True)

    def _signed_root_prior(
        self,
        root_grn_data: "np.ndarray",
    ) -> "np.ndarray":
        edge_probabilities = self._root_edge_probabilities(root_grn_data)
        negative = edge_probabilities * (1.0 - LINEAGE_POSITIVE_FRACTION)
        zero = 1.0 - edge_probabilities
        positive = edge_probabilities * LINEAGE_POSITIVE_FRACTION
        return np.stack((negative, zero, positive), axis=-1)

    def _reshape_lineage_signed_ctmc_input(self) -> "dict":
        """Build the simulator-matched signed CTMC and rooted tree structure."""
        negative_gain = LINEAGE_GAIN_RATE * (1.0 - LINEAGE_POSITIVE_FRACTION)
        positive_gain = LINEAGE_GAIN_RATE * LINEAGE_POSITIVE_FRACTION
        generator = np.array(
            [
                [
                    -(LINEAGE_LOSS_RATE + LINEAGE_SIGN_SWITCH_RATE),
                    LINEAGE_LOSS_RATE,
                    LINEAGE_SIGN_SWITCH_RATE,
                ],
                [
                    negative_gain,
                    -LINEAGE_GAIN_RATE,
                    positive_gain,
                ],
                [
                    LINEAGE_SIGN_SWITCH_RATE,
                    LINEAGE_LOSS_RATE,
                    -(LINEAGE_LOSS_RATE + LINEAGE_SIGN_SWITCH_RATE),
                ],
            ],
            dtype=float,
        )
        children = {node_id: [] for node_id in self.fate_map.nodes}
        parents = {}
        for edge in self.fate_map.edges:
            transition = expm(generator * edge.weight)
            transition[transition < 0] = 0
            transition /= transition.sum(axis=1, keepdims=True)
            children[edge.start].append((edge.end, transition))
            parents[edge.end] = (edge.start, transition)
        return {
            "root": self.fate_map.node_root,
            "children": children,
            "parents": parents,
        }

    def _signed_tree_messages(
        self,
        leaf_state_probabilities: "np.ndarray",
        root_grn_data: "np.ndarray",
        tree: "dict",
        return_posteriors: "bool" = False,
    ):
        """Run exact sum-product inference independently for every TF-target edge."""
        leaf_ids = self.fate_map.node_leaves
        regulator_count = leaf_state_probabilities.shape[0]
        root_priors = self._signed_root_prior(root_grn_data)
        children = tree["children"]
        root = tree["root"]
        postorder = []

        def visit(node_id):
            for child_id, _ in children[node_id]:
                visit(child_id)
            postorder.append(node_id)

        visit(root)
        leaf_index = {node_id: index for index, node_id in enumerate(leaf_ids)}
        all_posteriors = {
            node_id: np.zeros((regulator_count, 3), dtype=float) for node_id in children
        }
        log_likelihood = 0.0

        for regulator_index in range(regulator_count):
            evidence = {}
            for node_id in children:
                if node_id in leaf_index:
                    evidence[node_id] = leaf_state_probabilities[
                        regulator_index,
                        leaf_index[node_id],
                    ]
                else:
                    evidence[node_id] = np.ones(3, dtype=float)

            upward = {}
            child_messages = {}
            for node_id in postorder:
                value = evidence[node_id].copy()
                for child_id, transition in children[node_id]:
                    message = transition @ upward[child_id]
                    child_messages[(node_id, child_id)] = message
                    value *= message
                upward[node_id] = value

            marginal_likelihood = float(
                np.dot(root_priors[regulator_index], upward[root])
            )
            log_likelihood += np.log(marginal_likelihood + 1e-12)
            if not return_posteriors:
                continue

            downward = {root: root_priors[regulator_index]}
            stack = [root]
            while stack:
                node_id = stack.pop()
                marginal = downward[node_id] * evidence[node_id]
                for child_id, _ in children[node_id]:
                    marginal *= child_messages[(node_id, child_id)]
                total = marginal.sum()
                if total > 0:
                    all_posteriors[node_id][regulator_index] = marginal / total

                for child_id, transition in children[node_id]:
                    context = downward[node_id] * evidence[node_id]
                    for sibling_id, _ in children[node_id]:
                        if sibling_id != child_id:
                            context *= child_messages[(node_id, sibling_id)]
                    downward[child_id] = context @ transition
                    stack.append(child_id)

        if return_posteriors:
            return log_likelihood, all_posteriors
        return log_likelihood

    def _lineage_prior_ctmc_signed(
        self,
        leaves_grn_data: "np.ndarray",
        root_grn_data: "np.ndarray",
        tree: "dict",
    ) -> "float":
        leaf_weights = np.reshape(
            leaves_grn_data,
            (len(self.fate_map.node_leaves), -1),
        ).T
        leaf_probabilities = self._signed_state_probabilities(leaf_weights)
        return (
            self._signed_tree_messages(
                leaf_probabilities,
                root_grn_data,
                tree,
            )
            * LINEAGE_PRIOR_WEIGHT
        )

    def _signed_node_posteriors_ctmc(
        self,
        leaves_grn_data: "np.ndarray",
        root_grn_data: "np.ndarray",
        tree: "dict",
    ) -> "Dict[str, np.ndarray]":
        leaf_weights = np.reshape(
            leaves_grn_data,
            (len(self.fate_map.node_leaves), -1),
        ).T
        leaf_probabilities = self._signed_state_probabilities(leaf_weights)
        _, posteriors = self._signed_tree_messages(
            leaf_probabilities,
            root_grn_data,
            tree,
            return_posteriors=True,
        )
        return posteriors

    def _lineage_prior_ctmc(
        self,
        leaves_grn_data: "np.ndarray",
        root_grn_data: "np.ndarray",
        ancestor_nodes_space: "np.ndarray",
        transitions: "List[tuple[int, int, np.ndarray]]",
    ):
        """Evaluate the exact binary edge-presence CTMC on the fate map.

        The simulator's signed three-state CTMC is lumpable with respect to
        edge presence because both signs share the same loss rate and the two
        gain rates sum to ``LINEAGE_GAIN_RATE``. Primary leaf benchmarking is
        unsigned, so this is the matching marginal transition model.
        """
        leaf_node_ids = self.fate_map.node_leaves
        ancestor_node_ids = [self.fate_map.node_root] + self.fate_map.node_internals
        node_ids = ancestor_node_ids + leaf_node_ids
        root_index = node_ids.index(self.fate_map.node_root)

        leaf_weights = np.reshape(
            leaves_grn_data,
            (len(leaf_node_ids), -1),
        ).T
        smooth_magnitudes = np.sqrt(np.square(leaf_weights) + 1e-8)
        leaf_presence = expit(
            (smooth_magnitudes - LINEAGE_EDGE_THRESHOLD) / LINEAGE_EDGE_TEMPERATURE
        )
        root_probabilities = self._root_edge_probabilities(root_grn_data)

        log_marginals = []
        ancestor_count = len(ancestor_node_ids)
        for regulator_index, regulator_leaf_presence in enumerate(leaf_presence):
            probability = np.where(
                ancestor_nodes_space[:, root_index] == 1,
                root_probabilities[regulator_index],
                1 - root_probabilities[regulator_index],
            )
            for parent_index, child_index, transition in transitions:
                parent_states = ancestor_nodes_space[:, parent_index]
                if child_index < ancestor_count:
                    child_states = ancestor_nodes_space[:, child_index]
                    probability *= transition[parent_states, child_states]
                else:
                    child_probability = regulator_leaf_presence[
                        child_index - ancestor_count
                    ]
                    probability *= (
                        transition[parent_states, 0] * (1 - child_probability)
                        + transition[parent_states, 1] * child_probability
                    )
            log_marginals.append(np.log(probability.sum() + 1e-10))
        return float(np.sum(log_marginals)) * LINEAGE_PRIOR_WEIGHT

    def _ancestor_posteriors_ctmc(
        self,
        leaves_grn_data: "np.ndarray",
        root_grn_data: "np.ndarray",
        ancestor_nodes_space: "np.ndarray",
        transitions: "List[tuple[int, int, np.ndarray]]",
    ) -> "np.ndarray":
        """Return posterior edge-presence probabilities for internal nodes."""
        leaf_node_ids = self.fate_map.node_leaves
        ancestor_node_ids = [self.fate_map.node_root] + self.fate_map.node_internals
        node_ids = ancestor_node_ids + leaf_node_ids
        root_index = node_ids.index(self.fate_map.node_root)
        ancestor_count = len(ancestor_node_ids)

        leaf_weights = np.reshape(
            leaves_grn_data,
            (len(leaf_node_ids), -1),
        ).T
        smooth_magnitudes = np.sqrt(np.square(leaf_weights) + 1e-8)
        leaf_presence = expit(
            (smooth_magnitudes - LINEAGE_EDGE_THRESHOLD) / LINEAGE_EDGE_TEMPERATURE
        )
        root_probabilities = self._root_edge_probabilities(root_grn_data)
        posteriors = np.zeros(
            (len(self.rna_data.regulator_genes), ancestor_count),
            dtype=float,
        )

        for regulator_index, regulator_leaf_presence in enumerate(leaf_presence):
            probability = np.where(
                ancestor_nodes_space[:, root_index] == 1,
                root_probabilities[regulator_index],
                1 - root_probabilities[regulator_index],
            )
            for parent_index, child_index, transition in transitions:
                parent_states = ancestor_nodes_space[:, parent_index]
                if child_index < ancestor_count:
                    child_states = ancestor_nodes_space[:, child_index]
                    probability *= transition[parent_states, child_states]
                else:
                    child_probability = regulator_leaf_presence[
                        child_index - ancestor_count
                    ]
                    probability *= (
                        transition[parent_states, 0] * (1 - child_probability)
                        + transition[parent_states, 1] * child_probability
                    )
            normalizer = probability.sum()
            if normalizer <= 0:
                continue
            normalized = probability / normalizer
            posteriors[regulator_index] = normalized @ ancestor_nodes_space
        return posteriors

    def _get_strength(self, target_gene_id: "int" = 0, saved: "bool" = True):
        """
        Estimate gene regulatory network values for a given target gene.

        Parameters:
        target_gene_id (int): Identifier of the target gene to estimate GRN for.
        saved (bool): Flag indicating whether to save the results.

        Returns:
        Dict[str, List[float]]: Inferred GRN values keyed by node identifiers.
        """
        grn_values = {}
        x = list(self.infer_leaf_grn(target_gene_id))

        leaf_node_ids = self.fate_map.node_leaves
        for idx, node_id in enumerate(leaf_node_ids):
            grn_value = x[
                1
                + idx * len(self.rna_data.regulator_genes) : 1
                + (idx + 1) * len(self.rna_data.regulator_genes)
            ]
            grn_values[node_id] = x[:1] + grn_value

        grn_values = self.infer_ancestor_grn(grn_values, target_gene_id)
        if saved:
            self._save(grn_values, target_gene_id)
            logger.info(f"Saved grn values for target_gene_id:{target_gene_id}")

        return grn_values

    def _reshape_likelihood_input(self, target_gene_id):
        """
        Reshape RNA data for likelihood computation.

        Parameters:
        target_gene_id: Identifier for the target gene.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Regulator and target data arrays reshaped for likelihood.
        """
        regulator_parts = []
        target_parts = []
        for node_id in self.fate_map.node_leaves:
            regulators = self.rna_data.get_values(
                REGULATOR_GENE,
                node_id,
            ).values.astype(float)
            target = self.rna_data.get_values(TARGET_GENE, node_id=node_id,)[
                [target_gene_id]
            ].values.astype(float)
            regulator_scale = regulators.std(axis=0, keepdims=True)
            target_scale = target.std(axis=0, keepdims=True)
            regulators = (
                regulators - regulators.mean(axis=0, keepdims=True)
            ) / np.where(regulator_scale > 1e-8, regulator_scale, 1.0)
            target = (target - target.mean(axis=0, keepdims=True)) / np.where(
                target_scale > 1e-8, target_scale, 1.0
            )
            regulator_parts.append(regulators)
            target_parts.append(target)
        return regulator_parts, target_parts

    def _reshape_atac_prior_input(self, target_gene_id):
        """
        Reshape ATAC data for prior computation.

        Parameters:
        target_gene_id: Identifier for the target gene.

        Returns:
        np.ndarray: Processed metadata array for ATAC prior calculation.
        """
        samples = [
            self.atac_data.get_node_data(node_id=node_id)
            for node_id in self.fate_map.node_leaves
        ]
        mdata = np.abs(
            [
                self.atac_data.reshape(
                    sample, self.rna_data.regulator_genes, self.rna_data.target_genes
                )[[target_gene_id]].values
                for sample in samples
            ]
        )
        component1 = -np.log(1 + np.exp(-BETA_1 - BETA_2 * mdata))
        component2 = (-BETA_1 - BETA_2 * mdata) - np.log(
            1 + np.exp(-BETA_1 - BETA_2 * mdata)
        )
        mdata = np.concatenate((component1, component2), axis=-1)
        return mdata

    def _reshape_lineage_prior_input(self):
        """
        Prepare inputs for lineage prior computation by generating ancestor state space and weight matrices.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Ancestor state space array and corresponding weights.
        """
        ancestor_node_ids = [self.fate_map.node_root] + self.fate_map.node_internals
        ancestor_nodes_space = _generate_ancestor_state_space(ancestor_node_ids)

        node_ids = [node_id for node_id in self.fate_map.nodes.keys()]

        weights = pd.DataFrame(0.0, index=node_ids, columns=node_ids)
        for grn_edge in self.fate_map.edges:
            weights.loc[grn_edge.start, grn_edge.end] = np.exp(-grn_edge.weight)

        weights = weights.values[..., np.newaxis]
        weights = np.concatenate((weights, 1 - weights), axis=-1)
        weights = np.repeat(weights[np.newaxis, ...], len(ancestor_nodes_space), axis=0)

        return ancestor_nodes_space, weights

    def _reshape_lineage_ctmc_input(self):
        """Build per-branch presence transition matrices in node-index order."""
        ancestor_node_ids = [self.fate_map.node_root] + self.fate_map.node_internals
        ancestor_nodes_space = _generate_ancestor_state_space(ancestor_node_ids)
        node_ids = ancestor_node_ids + self.fate_map.node_leaves
        node_index = {node_id: index for index, node_id in enumerate(node_ids)}
        generator = np.array(
            [
                [-LINEAGE_GAIN_RATE, LINEAGE_GAIN_RATE],
                [LINEAGE_LOSS_RATE, -LINEAGE_LOSS_RATE],
            ],
            dtype=float,
        )
        transitions = []
        for edge in self.fate_map.edges:
            transition = expm(generator * edge.weight)
            transition[transition < 0] = 0
            transition /= transition.sum(axis=1, keepdims=True)
            transitions.append(
                (node_index[edge.start], node_index[edge.end], transition)
            )
        return ancestor_nodes_space, transitions

    def infer_leaf_grn(self, target_gene_id: "str" = None):
        """
        Infer GRN values for all leaf nodes for a specific target gene.

        Parameters:
        target_gene_id (str): Identifier of the target gene.

        Returns:
        np.ndarray: Optimized GRN values after inference.
        """
        logger.info(f"Start fitting target_gene_id:{target_gene_id}")
        atac_root_data = self.atac_data.get_node_data(node_id=self.fate_map.node_root)
        reg_genes = self.rna_data.regulator_genes
        root_grn_initial = _get_grn_initial_data(
            atac_root_data, reg_genes, target_gene_id
        )
        self_regulator_index = _self_regulator_index(
            reg_genes,
            target_gene_id,
        )
        if self_regulator_index is not None:
            root_grn_initial[self_regulator_index] = 0.0

        regulator_data, target_data = self._reshape_likelihood_input(target_gene_id)
        leaf_grn_initial = []
        for leaf_index, node_id in enumerate(self.fate_map.node_leaves):
            node_atac_data = self.atac_data.get_node_data(node_id)
            atac_initial = _get_grn_initial_data(
                node_atac_data,
                reg_genes,
                target_gene_id,
            ).astype(float)
            design = regulator_data[leaf_index]
            response = target_data[leaf_index][:, 0]
            ridge = np.eye(design.shape[1], dtype=float)
            coefficients = np.linalg.solve(
                design.T @ design + ridge,
                design.T @ response,
            )
            coefficient_scale = np.max(np.abs(coefficients))
            normalized_coefficients = (
                coefficients / coefficient_scale
                if coefficient_scale > 1e-8
                else coefficients
            )
            magnitudes = np.clip(
                0.7 * np.abs(atac_initial) + 0.3 * np.abs(normalized_coefficients),
                0.0,
                1.0,
            )
            signs = np.where(normalized_coefficients < 0, -1.0, 1.0)
            initial = magnitudes * signs
            if self_regulator_index is not None:
                initial[self_regulator_index] = 0.0
            leaf_grn_initial.extend(list(initial))

        mdata = self._reshape_atac_prior_input(target_gene_id)
        if LINEAGE_TRANSITION_MODEL == "ctmc_signed":
            lineage_data = self._reshape_lineage_signed_ctmc_input()
            ancestor_nodes_space = None
        elif LINEAGE_TRANSITION_MODEL == "ctmc_presence":
            ancestor_nodes_space, lineage_data = self._reshape_lineage_ctmc_input()
        else:
            ancestor_nodes_space, lineage_data = self._reshape_lineage_prior_input()

        def _loss(x):
            if LINEAGE_TRANSITION_MODEL == "ctmc_signed":
                y1 = self._lineage_prior_ctmc_signed(
                    x[1:],
                    root_grn_initial,
                    lineage_data,
                )
            elif LINEAGE_TRANSITION_MODEL == "ctmc_presence":
                y1 = self._lineage_prior_ctmc(
                    x[1:],
                    root_grn_initial,
                    ancestor_nodes_space,
                    lineage_data,
                )
            else:
                y1 = self._lineage_prior(
                    x[1:],
                    x[0],
                    root_grn_initial,
                    ancestor_nodes_space,
                    lineage_data,
                )
            y2 = self._atac_prior(x[1:], mdata)
            y3 = self._likelihood(x[1:], regulator_data, target_data)
            y = -y1 - y2 - y3
            return y

        x0 = np.array([LAMBDA_1] + leaf_grn_initial)
        lambda_bounds = (
            (LAMBDA_1, LAMBDA_1)
            if LINEAGE_TRANSITION_MODEL in {"ctmc_presence", "ctmc_signed"}
            else (LAMBDA_LOW_BOUND, LAMBDA_HIGH_BOUND)
        )
        edge_bounds = [
            (REGULATION_STRENGTH_LOW_BOUND, REGULATION_STRENGTH_HIGH_BOUND)
        ] * len(leaf_grn_initial)
        if self_regulator_index is not None:
            regulator_count = len(reg_genes)
            for leaf_index in range(len(self.fate_map.node_leaves)):
                edge_bounds[leaf_index * regulator_count + self_regulator_index] = (
                    0.0,
                    0.0,
                )
        bounds = [lambda_bounds] + edge_bounds

        optimize_options = {}
        if OPTIMIZE_METHOD == "L-BFGS-B":
            optimize_options["maxfun"] = OPTIMIZE_MAX_FUNCTION_EVALUATIONS
        result = minimize(
            fun=_loss,
            x0=x0,
            method=OPTIMIZE_METHOD,
            bounds=bounds,
            options=optimize_options,
        )
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError(
                f"Non-finite optimization result for target_gene_id:{target_gene_id}"
            )
        diagnostics_dir = os.path.join(
            os.path.dirname(self.saved_dir),
            "optimization_diagnostics",
        )
        os.makedirs(diagnostics_dir, exist_ok=True)
        diagnostic = {
            "target_gene_id": target_gene_id,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "objective": float(result.fun),
            "iterations": int(getattr(result, "nit", 0)),
            "function_evaluations": int(getattr(result, "nfev", 0)),
            "optimizer": OPTIMIZE_METHOD,
            "max_function_evaluations": OPTIMIZE_MAX_FUNCTION_EVALUATIONS,
            "lineage_model": LINEAGE_TRANSITION_MODEL,
        }
        with open(
            os.path.join(diagnostics_dir, f"target_gene_{target_gene_id}.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(diagnostic, handle, indent=2)
        if not result.success:
            logger.warning(
                f"Optimizer did not report convergence for "
                f"target_gene_id:{target_gene_id}: {result.message}"
            )

        logger.info(
            f"Finish inferencing leaves grn value for target_gene_id:{target_gene_id}"
        )
        return result.x

    def infer_ancestor_grn(
        self, grn_values: "Dict[str, List[float]]", target_gene_id: "str"
    ):
        """
        Infer ancestor GRN values by aggregating information from leaf nodes upward.

        Parameters:
        grn_values (Dict[str, List[float]]): Dictionary of GRN values for leaf nodes.
        target_gene_id (str): Identifier of the target gene.

        Returns:
        Dict[str, List[float]]: Updated dictionary including ancestor GRN values.
        """
        if LINEAGE_PRIOR_WEIGHT == 0:
            logger.info(
                f"Skip ancestor propagation for no-lineage variant "
                f"target_gene_id:{target_gene_id}"
            )
            return grn_values

        if LINEAGE_TRANSITION_MODEL == "ctmc_signed":
            leaf_values = np.concatenate(
                [grn_values[node_id][1:] for node_id in self.fate_map.node_leaves]
            )
            root_atac = self.atac_data.get_node_data(node_id=self.fate_map.node_root)
            root_initial = _get_grn_initial_data(
                root_atac,
                self.rna_data.regulator_genes,
                target_gene_id,
            )
            self_regulator_index = _self_regulator_index(
                self.rna_data.regulator_genes,
                target_gene_id,
            )
            if self_regulator_index is not None:
                root_initial[self_regulator_index] = 0.0
            tree = self._reshape_lineage_signed_ctmc_input()
            posteriors = self._signed_node_posteriors_ctmc(
                leaf_values,
                root_initial,
                tree,
            )
            node_ids = (
                [self.fate_map.node_root]
                + self.fate_map.node_internals
                + self.fate_map.node_leaves
            )
            parent_lambda = float(
                np.mean(
                    [grn_values[node_id][0] for node_id in self.fate_map.node_leaves]
                )
            )
            state_output = {}
            for node_id in node_ids:
                node_posteriors = posteriors[node_id]
                presence = node_posteriors[:, 0] + node_posteriors[:, 2]
                signs = np.where(
                    node_posteriors[:, 2] >= node_posteriors[:, 0],
                    1.0,
                    -1.0,
                )
                signed_scores = presence * signs
                if self_regulator_index is not None:
                    signed_scores[self_regulator_index] = 0.0
                    node_posteriors[self_regulator_index] = [0.0, 1.0, 0.0]
                grn_values[node_id] = [parent_lambda] + list(signed_scores)
                state_output[node_id] = {
                    regulator: {
                        "negative": float(probabilities[0]),
                        "zero": float(probabilities[1]),
                        "positive": float(probabilities[2]),
                        "presence": float(probabilities[0] + probabilities[2]),
                    }
                    for regulator, probabilities in zip(
                        self.rna_data.regulator_genes,
                        node_posteriors,
                    )
                }
            posterior_dir = os.path.join(
                os.path.dirname(self.saved_dir),
                "state_posteriors",
            )
            os.makedirs(posterior_dir, exist_ok=True)
            with open(
                os.path.join(
                    posterior_dir,
                    f"target_gene_{target_gene_id}.json",
                ),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(state_output, handle, indent=2)
            logger.info(
                f"Finish signed global all-node inference for "
                f"target_gene_id:{target_gene_id}"
            )
            return grn_values

        if LINEAGE_TRANSITION_MODEL == "ctmc_presence":
            leaf_values = np.concatenate(
                [grn_values[node_id][1:] for node_id in self.fate_map.node_leaves]
            )
            root_atac = self.atac_data.get_node_data(node_id=self.fate_map.node_root)
            root_initial = _get_grn_initial_data(
                root_atac,
                self.rna_data.regulator_genes,
                target_gene_id,
            )
            self_regulator_index = _self_regulator_index(
                self.rna_data.regulator_genes,
                target_gene_id,
            )
            if self_regulator_index is not None:
                root_initial[self_regulator_index] = 0.0
            ancestor_space, transitions = self._reshape_lineage_ctmc_input()
            posteriors = self._ancestor_posteriors_ctmc(
                leaf_values,
                root_initial,
                ancestor_space,
                transitions,
            )
            ancestor_ids = [self.fate_map.node_root] + self.fate_map.node_internals
            parent_lambda = float(
                np.mean(
                    [grn_values[node_id][0] for node_id in self.fate_map.node_leaves]
                )
            )
            for index, node_id in enumerate(ancestor_ids):
                values = posteriors[:, index]
                if self_regulator_index is not None:
                    values[self_regulator_index] = 0.0
                grn_values[node_id] = [parent_lambda] + list(values)
            logger.info(
                f"Finish inferring ancestor grn values for target_gene_id:{target_gene_id}"
            )
            return grn_values

        stacks = deepcopy(self.fate_map.node_leaves)

        def _isin_stacks(node_ids, stacks):
            flag = (
                False
                if sum([0 if node_id in stacks else 1 for node_id in node_ids])
                else True
            )
            return flag

        idx = 0
        while idx < len(stacks):
            node_id = stacks[idx]
            parent_node_id = self.fate_map.nodes[node_id].upstream_node_id
            if not parent_node_id or parent_node_id in stacks:
                idx += 1
                continue

            parent_node = self.fate_map.get_parent_node(node_id)
            edges = parent_node.directed_edges
            children_ids = [edge.end for edge in edges]

            if not _isin_stacks(children_ids, stacks):
                idx += 1
                continue

            parent_grn_value = _parent_grn_inference(
                edges, grn_values, len(self.rna_data.regulator_genes)
            )
            stacks.append(parent_node.node_id)
            grn_values[parent_node.node_id] = parent_grn_value
            idx += 1

        logger.info(
            f"Finish inferring ancestor grn values for target_gene_id:{target_gene_id}"
        )
        return grn_values

    def _save(self, grn_values: "Dict[str, List[float]]", target_gene_id: "str"):
        """
        Save the inferred GRN values to a CSV file.

        Parameters:
        grn_values (Dict[str, List[float]]): Dictionary of GRN values keyed by node.
        target_gene_id (str): Identifier of the target gene.
        """
        os.makedirs(self.saved_dir, exist_ok=True)

        saved_path = os.path.join(self.saved_dir, f"target_gene_{target_gene_id}.csv")
        with open(saved_path, "w+") as f:
            for node_id, grn_value in grn_values.items():
                line = [target_gene_id, node_id] + grn_value
                line = json.dumps(line)
                f.write(line + "\n")

    def infer_grn(self, max_processing=CPU_NUM):
        """
        Estimate GRN values for all target genes using parallel processing if specified.

        Parameters:
        max_processing: Maximum number of parallel processes to use.
        """
        if max_processing > 1:
            with Pool(
                max_processing,
                initializer=_initialize_inference_worker,
                initargs=(_worker_settings(),),
            ) as p:
                p.map(self._get_strength, self.rna_data.target_genes)
        else:
            for target_gene_id in self.rna_data.target_genes:
                self._get_strength(target_gene_id)

    def get_target_networks(self, threshold: "float") -> "Dict[str, Dict[str, Dict]]":
        """
        Retrieve gene regulatory networks for each target gene based on a threshold.

        Parameters:
        threshold (float): Threshold for filtering GRN values.
        Returns:
        Dict[str, Dict[str, Dict]]: Nested dictionary of GRN networks keyed by node and target gene.
        """
        grn_dict = {}
        filenames = [
            f"target_gene_{target_gene_id}.csv"
            for target_gene_id in self.rna_data.target_genes
        ]
        for filename in filenames:
            file_path = os.path.join(self.saved_dir, filename)
            with open(file_path, "r", encoding="utf-8") as target_file:
                for line in target_file:
                    line = json.loads(line.strip())
                    target_gene_id = line[0]
                    node_id = line[1]
                    lambda_1 = line[2]
                    grn_value = [
                        0.0 if abs(value) < threshold else value for value in line[3:]
                    ]
                    if node_id not in grn_dict:
                        grn_dict[node_id] = {
                            target_gene_id: {"lambda": lambda_1, "grn_value": grn_value}
                        }
                    else:
                        grn_dict[node_id][target_gene_id] = {
                            "lambda": lambda_1,
                            "grn_value": grn_value,
                        }

        return grn_dict
