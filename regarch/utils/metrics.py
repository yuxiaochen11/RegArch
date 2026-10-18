"""Evaluation metrics used by the example and perturbation workflows."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def _edge_labels_and_scores(
    inferred_grn: pd.DataFrame,
    true_grn: pd.DataFrame,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return binary truth labels and continuous absolute prediction scores."""
    if inferred_grn.shape != true_grn.shape:
        raise ValueError("inferred_grn and true_grn must have the same shape")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    scores = np.abs(inferred_grn.to_numpy(dtype=float)).ravel()
    truth = np.abs(true_grn.to_numpy(dtype=float)).ravel()
    finite = np.isfinite(scores) & np.isfinite(truth)
    if not finite.any():
        raise ValueError("GRN inputs contain no finite paired values")

    labels = (truth[finite] > threshold).astype(int)
    if np.unique(labels).size < 2:
        raise ValueError("true_grn must contain both positive and negative edges")
    return labels, scores[finite]


def compute_auroc(
    inferred_grn: pd.DataFrame,
    true_grn: pd.DataFrame,
    threshold: float,
) -> float:
    """Compute AUROC from continuous inferred scores and thresholded truth."""
    labels, scores = _edge_labels_and_scores(inferred_grn, true_grn, threshold)
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    return float(auc(false_positive_rate, true_positive_rate))


def compute_auprc(
    inferred_grn: pd.DataFrame,
    true_grn: pd.DataFrame,
    threshold: float,
) -> float:
    """Compute AUPRC from continuous inferred scores and thresholded truth."""
    labels, scores = _edge_labels_and_scores(inferred_grn, true_grn, threshold)
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def normalize_to_zero_mean(frame: pd.DataFrame) -> pd.DataFrame:
    """Center all values in a data frame around the global mean."""
    return frame - frame.to_numpy().mean()


def count_greater_than_threshold(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return the fraction of values above ``threshold`` in each row."""
    fractions = (frame > threshold).sum(axis=1) / frame.shape[1]
    return fractions.to_frame(name="count")


def metric_perturbation_impact(
    grn_before: Mapping[str, pd.DataFrame],
    grn_after: Mapping[str, pd.DataFrame],
    threshold: float = 0.05,
    top_k: int = 30,
) -> tuple[float, float]:
    """Measure mean edge Jaccard similarity and top-regulator overlap."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    jaccard_scores = []
    overlap_scores = []
    for node_id in sorted(grn_before.keys() & grn_after.keys()):
        before = grn_before[node_id]
        after = grn_after[node_id]
        if set(before.index) != set(after.index) or set(before.columns) != set(
            after.columns
        ):
            raise ValueError(f"GRN labels differ for node {node_id!r}")
        before = before.loc[after.index, after.columns]

        before_edges = before.abs() > threshold
        after_edges = after.abs() > threshold
        intersection = (before_edges & after_edges).to_numpy().sum()
        union = (before_edges | after_edges).to_numpy().sum()
        jaccard_scores.append(float(intersection / union) if union else 1.0)

        retained = min(top_k, after.shape[1])
        before_top = set(before.abs().sum(axis=0).nlargest(retained).index)
        after_top = set(after.abs().sum(axis=0).nlargest(retained).index)
        overlap_scores.append(len(before_top & after_top) / retained)

    if not jaccard_scores:
        return 0.0, 0.0
    return float(np.mean(jaccard_scores)), float(np.mean(overlap_scores))


def _binarize_paired_arrays(
    inferred: np.ndarray, observed: np.ndarray, threshold: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-finite pairs and convert both arrays to binary edge states."""
    inferred = np.asarray(inferred, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if inferred.shape != observed.shape:
        raise ValueError("paired arrays must have identical shapes")
    finite = np.isfinite(inferred) & np.isfinite(observed)
    return (
        (inferred[finite] > threshold).astype(int),
        (observed[finite] > threshold).astype(int),
    )


def match_expression_data(
    inferred_data: Sequence[np.ndarray],
    observed_data: Sequence[np.ndarray],
    rank: int = 1,
    sample_size: int = 10_000,
    seed: int = 0,
) -> tuple[np.ndarray, float, float, tuple[int, ...], list[tuple[int, ...]]]:
    """Rank mappings between inferred and observed expression-state samples.

    All permutations are evaluated for at most seven samples. Larger inputs use
    a deterministic Monte Carlo subset that always includes the identity map.
    ``rank`` is one-based.
    """
    sample_count = len(inferred_data)
    if sample_count != len(observed_data) or sample_count == 0:
        raise ValueError("inferred_data and observed_data must be equally non-empty")
    if rank < 1:
        raise ValueError("rank must be at least 1")
    if sample_size < 1:
        raise ValueError("sample_size must be positive")

    distances = np.zeros((sample_count, sample_count), dtype=float)
    for inferred_index in range(sample_count):
        for observed_index in range(sample_count):
            inferred, observed = _binarize_paired_arrays(
                inferred_data[inferred_index], observed_data[observed_index]
            )
            distances[inferred_index, observed_index] = np.sum(inferred != observed)

    if sample_count <= 7:
        permutations = list(itertools.permutations(range(sample_count)))
    else:
        requested = min(sample_size, math.factorial(sample_count))
        permutation_set = {tuple(range(sample_count))}
        random = np.random.default_rng(seed)
        while len(permutation_set) < requested:
            permutation_set.add(tuple(random.permutation(sample_count)))
        permutations = sorted(permutation_set)

    losses = np.asarray(
        [
            sum(distances[index, permutation[index]] for index in range(sample_count))
            for permutation in permutations
        ]
    )
    order = np.argsort(losses, kind="stable")
    sorted_losses = losses[order]
    sorted_permutations = [permutations[index] for index in order]
    if rank > len(sorted_permutations):
        raise ValueError("rank exceeds the number of evaluated permutations")

    minimum, maximum = sorted_losses[0], sorted_losses[-1]
    normalized_losses = (
        (sorted_losses - minimum) / (maximum - minimum)
        if maximum > minimum
        else np.zeros_like(sorted_losses)
    )
    selected = rank - 1
    return (
        normalized_losses,
        float(sorted_losses[selected]),
        float(normalized_losses[selected]),
        sorted_permutations[selected],
        sorted_permutations,
    )
