from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REQUIRED_OUTPUT_COLUMNS = {
    "method",
    "cell_type",
    "regulator",
    "target",
    "score",
}


def build_edge_universe(true_leaf_grn: pd.DataFrame) -> pd.DataFrame:
    universe = true_leaf_grn.rename(columns={"node": "cell_type"}).copy()
    return universe.loc[universe["regulator"] != universe["target"]].reset_index(
        drop=True
    )


def validate_standardized_output(scores: pd.DataFrame) -> None:
    missing = REQUIRED_OUTPUT_COLUMNS - set(scores.columns)
    if missing:
        raise ValueError(
            f"Standardized method output is missing columns: {sorted(missing)}"
        )
    if scores.duplicated(["method", "cell_type", "regulator", "target"]).any():
        raise ValueError("Standardized method output contains duplicate edge scores.")


def _top_k_metrics(
    labels: np.ndarray, scores: np.ndarray, k: int
) -> tuple[float, float]:
    order = np.argsort(-scores, kind="stable")[:k]
    true_positive = int(labels[order].sum())
    precision = true_positive / k if k else np.nan
    recall = true_positive / labels.sum() if labels.sum() else np.nan
    return precision, recall


def evaluate_scores(
    edge_universe: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    validate_standardized_output(scores)
    merged_parts = []
    for method, method_scores in scores.groupby("method"):
        complete = edge_universe.merge(
            method_scores.drop(columns="method"),
            on=["cell_type", "regulator", "target"],
            how="left",
        )
        complete.insert(0, "method", method)
        complete["score"] = complete["score"].fillna(0.0).abs()
        if "sign_optional" not in complete:
            complete["sign_optional"] = np.nan
        merged_parts.append(complete)
    merged = pd.concat(merged_parts, ignore_index=True)
    rows = []
    for (method, cell_type), group in merged.groupby(["method", "cell_type"]):
        labels = group["true_edge"].to_numpy(dtype=int)
        values = group["score"].to_numpy(dtype=float)
        if labels.min() == labels.max():
            auroc = np.nan
            auprc = np.nan
        else:
            auroc = roc_auc_score(labels, values)
            auprc = average_precision_score(labels, values)
        k = int(labels.sum())
        precision_at_k, recall_at_k = _top_k_metrics(labels, values, k)
        order = np.argsort(-values, kind="stable")[:k]
        true_positive_mask = labels[order] == 1
        predicted_signs = group["sign_optional"].to_numpy(dtype=float)[order]
        true_signs = group["true_sign"].to_numpy(dtype=float)[order]
        signed_available = np.isfinite(predicted_signs) & (predicted_signs != 0)
        recovered_with_sign = true_positive_mask & signed_available
        correctly_signed = recovered_with_sign & (
            np.sign(predicted_signs) == np.sign(true_signs)
        )
        signed_true_positive_count = int(correctly_signed.sum())
        signed_precision_at_k = signed_true_positive_count / k if k else np.nan
        signed_recall_at_k = (
            signed_true_positive_count / labels.sum() if labels.sum() else np.nan
        )
        sign_accuracy_on_recovered_true_edges = (
            float(correctly_signed.sum() / recovered_with_sign.sum())
            if recovered_with_sign.any()
            else np.nan
        )
        sign_coverage_on_recovered_true_edges = (
            float(recovered_with_sign.sum() / true_positive_mask.sum())
            if true_positive_mask.any()
            else np.nan
        )

        true_regulator_scores = (
            group.loc[group["true_edge"].eq(1)].groupby("regulator")["true_edge"].sum()
        )
        regulator_k = min(
            10,
            group["regulator"].nunique(),
            len(true_regulator_scores),
        )
        true_regulators = set(true_regulator_scores.nlargest(regulator_k).index)
        inferred_regulators = set(
            group.groupby("regulator")["score"].sum().nlargest(regulator_k).index
        )
        regulator_overlap = (
            len(true_regulators & inferred_regulators) / len(true_regulators)
            if true_regulators
            else np.nan
        )
        rows.append(
            {
                "method": method,
                "cell_type": cell_type,
                "true_edge_count": k,
                "auprc": auprc,
                "auroc": auroc,
                "precision_at_true_k": precision_at_k,
                "recall_at_true_k": recall_at_k,
                "signed_precision_at_true_k": signed_precision_at_k,
                "signed_recall_at_true_k": signed_recall_at_k,
                "sign_accuracy_on_recovered_true_edges": (
                    sign_accuracy_on_recovered_true_edges
                ),
                "sign_coverage_on_recovered_true_edges": (
                    sign_coverage_on_recovered_true_edges
                ),
                "top10_regulator_overlap": regulator_overlap,
            }
        )
    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        aggregate = (
            metrics.groupby("method", as_index=False)[
                [
                    "auprc",
                    "auroc",
                    "precision_at_true_k",
                    "recall_at_true_k",
                    "signed_precision_at_true_k",
                    "signed_recall_at_true_k",
                    "sign_accuracy_on_recovered_true_edges",
                    "sign_coverage_on_recovered_true_edges",
                    "top10_regulator_overlap",
                ]
            ]
            .mean()
            .assign(cell_type="__mean__", true_edge_count=np.nan)
        )
        metrics = pd.concat([metrics, aggregate[metrics.columns]], ignore_index=True)
    return metrics


def load_standardized_outputs(paths: list[str | Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        path = Path(path)
        separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        frames.append(pd.read_csv(path, sep=separator))
    return pd.concat(frames, ignore_index=True)
