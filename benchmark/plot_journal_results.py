from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch

from benchmark.plot_style import (
    DOUBLE_COLUMN,
    export_figure,
    get_palette,
    set_paper_style,
)

METHOD_ORDER = [
    "RegArch",
    "FigR",
    "SCENICplus",
    "LINGER",
    "CellOracle",
    "GRaNIE",
    "Pando",
    "scMTNI",
    "Dictys",
    "SCENIC",
    "GENIE3",
    "GRNBoost",
]

METHOD_LABELS = {"SCENICplus": "SCENIC+"}
UNSIGNED_METHODS = {"GENIE3", "GRNBoost", "LINGER"}

METRICS = [
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
    ("precision_at_true_k", "Precision at true K"),
    ("signed_precision_at_true_k", "Signed precision"),
]

FATE_CONDITIONS = ["full", "wrong_fate_map", "no_fate_map"]
FATE_LABELS = {
    "full": "Full",
    "wrong_fate_map": "Wrong fate map",
    "no_fate_map": "No fate map",
}

HYPERPARAM_LABELS = {
    "baseline_cells30_genes20_lbfgsb_max60000": "Baseline",
    "cells15_genes20_lbfgsb_max60000": "15 cells",
    "cells60_genes20_lbfgsb_max60000": "60 cells",
    "cells30_genes40_lbfgsb_max60000": "40 genes",
    "baseline_lbfgsb_max20000": "20k",
    "baseline_lbfgsb_max120000": "120k",
    "baseline_slsqp_max20000": "SLSQP",
    "baseline_cobyla_max20000": "COBYLA",
}


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def _clean_output_dir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for path in outdir.iterdir():
        if path.suffix.lower() in {".pdf", ".png", ".tsv"}:
            path.unlink()
        elif path.is_dir() and path.name == "__pycache__":
            shutil.rmtree(path)


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _save(fig: plt.Figure, name: str, outdir: Path) -> list[Path]:
    fig.tight_layout()
    saved = export_figure(fig, name, outdir=outdir, formats=("pdf", "png"), dpi=600)
    plt.close(fig)
    return saved


def _style_boxplot(boxplot: dict, color: str) -> None:
    for box in boxplot["boxes"]:
        box.set(facecolor=color, edgecolor="black", linewidth=0.6, alpha=0.9)
    for median in boxplot["medians"]:
        median.set(color="black", linewidth=0.9)
    for whisker in boxplot["whiskers"]:
        whisker.set(color="black", linewidth=0.6)
    for cap in boxplot["caps"]:
        cap.set(color="black", linewidth=0.6)
    for flier in boxplot["fliers"]:
        flier.set(marker="o", markersize=1.8, markerfacecolor=color, markeredgewidth=0)


def _boxplot_by_category(
    ax: plt.Axes,
    frame: pd.DataFrame,
    category_col: str,
    categories: list[str],
    value_col: str,
    labels: list[str],
    colors: list[str],
    ylim: tuple[float, float] | None = None,
) -> None:
    data = []
    positions = []
    box_colors = []
    for index, category in enumerate(categories, start=1):
        values = (
            frame.loc[frame[category_col].eq(category), value_col].dropna().to_numpy()
        )
        if values.size == 0:
            continue
        data.append(values)
        positions.append(index)
        box_colors.append(colors[index - 1])
    bp = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        widths=0.58,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 0.9},
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set(facecolor=color, edgecolor="black", linewidth=0.6, alpha=0.9)
    for key in ("whiskers", "caps"):
        for item in bp[key]:
            item.set(color="black", linewidth=0.6)
    for flier, color in zip(bp["fliers"], box_colors):
        flier.set(
            marker="o",
            markersize=1.8,
            markerfacecolor=color,
            markeredgecolor=color,
            alpha=0.55,
        )
    ax.set_xticks(np.arange(1, len(categories) + 1))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    if ylim is None:
        ax.set_ylim(*_adaptive_ylim(frame[value_col].dropna()))
    else:
        ax.set_ylim(*ylim)


def _adaptive_ylim(
    values: pd.Series | np.ndarray, lower_bound: float = 0.0, upper_bound: float = 1.0
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return lower_bound, upper_bound
    low = float(np.nanmin(array))
    high = float(np.nanmax(array))
    if np.isclose(low, high):
        pad = 0.05 if high == 0 else max(0.03, abs(high) * 0.08)
    else:
        pad = max(0.025, (high - low) * 0.18)
    return max(lower_bound, low - pad), min(upper_bound, high + pad)


def _set_adaptive_ylim(ax: plt.Axes, values: pd.Series | np.ndarray) -> None:
    ax.set_ylim(*_adaptive_ylim(values))


def plot_benchmark_main(metrics_path: Path, outdir: Path, tag: str) -> list[Path]:
    metrics = _read_tsv(metrics_path)
    metrics = metrics.loc[metrics["cell_type"].ne("__mean__")].copy()
    methods = [method for method in METHOD_ORDER if method in set(metrics["method"])]
    palette = get_palette("default")
    colors = [
        palette["colors"][4] if method == "RegArch" else palette["colors"][0]
        for method in methods
    ]

    fig, axes = plt.subplots(2, 2, figsize=DOUBLE_COLUMN)
    for ax, (metric, title) in zip(axes.ravel(), METRICS):
        plot_metrics = metrics.copy()
        if metric == "signed_precision_at_true_k":
            plot_metrics.loc[
                plot_metrics["method"].isin(UNSIGNED_METHODS),
                metric,
            ] = np.nan
        _boxplot_by_category(
            ax,
            plot_metrics,
            "method",
            methods,
            metric,
            [_method_label(m) for m in methods],
            colors,
        )
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_xlabel("")
        if metric == "signed_precision_at_true_k":
            ax.text(
                0.99,
                0.03,
                "LINGER omitted: unsigned score output",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.8,
                color=palette["gray"],
            )
    fig.suptitle(f"{tag}: leaf-node benchmark across 12 methods", y=1.02)
    return _save(fig, f"{tag}_benchmark_4metric_boxplots", outdir)


def _filter_atac(summary: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "coverage":
        frame = summary.loc[summary["requested_correctness"].eq(1.0)].copy()
        return frame.sort_values("requested_coverage")
    if mode == "correctness":
        frame = summary.loc[summary["requested_coverage"].eq(1.0)].copy()
        return frame.sort_values("requested_correctness")
    raise ValueError(mode)


def _hyper_value(summary: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    frame = summary.loc[summary["condition"].isin(conditions)].copy()
    order = {condition: index for index, condition in enumerate(conditions)}
    frame["order"] = frame["condition"].map(order)
    return frame.sort_values("order")


def plot_sensitivity_main(
    atac_summary_path: Path,
    hyper_summary_path: Path,
    runtime_summary_path: Path,
    outdir: Path,
    tag: str,
) -> list[Path]:
    atac = _read_tsv(atac_summary_path)
    hyper = _read_tsv(hyper_summary_path)
    runtime = _read_tsv(runtime_summary_path) if runtime_summary_path.exists() else None
    palette = get_palette("default")
    colors = palette["colors"]

    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN[0], DOUBLE_COLUMN[1] * 1.15))

    coverage = _filter_atac(atac, "coverage")
    axes[0, 0].plot(
        coverage["requested_coverage"],
        coverage["auprc_mean"],
        color=colors[0],
        marker="o",
    )
    axes[0, 0].set_title("ATAC coverage")
    axes[0, 0].set_xlabel("Coverage")
    axes[0, 0].set_ylabel("AUPRC")
    _set_adaptive_ylim(axes[0, 0], coverage["auprc_mean"])

    correctness = _filter_atac(atac, "correctness")
    axes[0, 1].plot(
        correctness["requested_correctness"],
        correctness["auprc_mean"],
        color=colors[2],
        marker="o",
    )
    axes[0, 1].set_title("ATAC correctness")
    axes[0, 1].set_xlabel("Correctness")
    axes[0, 1].set_ylabel("AUPRC")
    _set_adaptive_ylim(axes[0, 1], correctness["auprc_mean"])

    cells = _hyper_value(
        hyper,
        [
            "cells15_genes20_lbfgsb_max60000",
            "baseline_cells30_genes20_lbfgsb_max60000",
            "cells60_genes20_lbfgsb_max60000",
        ],
    )
    axes[0, 2].bar(
        np.arange(len(cells)),
        cells["auprc_mean"],
        color=colors[1],
        edgecolor="none",
    )
    axes[0, 2].set_xticks(np.arange(len(cells)))
    axes[0, 2].set_xticklabels(
        [HYPERPARAM_LABELS[c] for c in cells["condition"]], rotation=25, ha="right"
    )
    axes[0, 2].set_title("Cell number")
    axes[0, 2].set_ylabel("AUPRC")
    _set_adaptive_ylim(axes[0, 2], cells["auprc_mean"])

    genes = _hyper_value(
        hyper,
        [
            "baseline_cells30_genes20_lbfgsb_max60000",
            "cells30_genes40_lbfgsb_max60000",
        ],
    )
    axes[1, 0].bar(
        np.arange(len(genes)),
        genes["auprc_mean"],
        color=colors[3],
        edgecolor="none",
    )
    axes[1, 0].set_xticks(np.arange(len(genes)))
    axes[1, 0].set_xticklabels(
        [HYPERPARAM_LABELS[c] for c in genes["condition"]], rotation=25, ha="right"
    )
    axes[1, 0].set_title("Gene number")
    axes[1, 0].set_ylabel("AUPRC")
    _set_adaptive_ylim(axes[1, 0], genes["auprc_mean"])

    budget = _hyper_value(
        hyper,
        [
            "baseline_lbfgsb_max20000",
            "baseline_cells30_genes20_lbfgsb_max60000",
            "baseline_lbfgsb_max120000",
        ],
    )
    x_budget = [20, 60, 120]
    axes[1, 1].plot(
        x_budget[: len(budget)],
        budget["auprc_mean"],
        color=colors[4],
        marker="o",
    )
    axes[1, 1].set_title("L-BFGS-B budget")
    axes[1, 1].set_xlabel("Max evaluations (thousand)")
    axes[1, 1].set_ylabel("AUPRC")
    _set_adaptive_ylim(axes[1, 1], budget["auprc_mean"])

    optimizers = _hyper_value(
        hyper,
        [
            "baseline_cells30_genes20_lbfgsb_max60000",
            "baseline_slsqp_max20000",
            "baseline_cobyla_max20000",
        ],
    )
    axes[1, 2].bar(
        np.arange(len(optimizers)),
        optimizers["auprc_mean"],
        color=colors[0],
        edgecolor="none",
    )
    axes[1, 2].set_xticks(np.arange(len(optimizers)))
    axes[1, 2].set_xticklabels(
        [HYPERPARAM_LABELS[c] for c in optimizers["condition"]], rotation=25, ha="right"
    )
    axes[1, 2].set_title("Optimizer")
    axes[1, 2].set_ylabel("AUPRC")
    _set_adaptive_ylim(axes[1, 2], optimizers["auprc_mean"])

    if runtime is not None and "mean" in runtime:
        runtime_minutes = runtime.set_index("condition")["mean"] / 60.0
        for ax, conditions in [
            (axes[0, 2], list(cells["condition"])),
            (axes[1, 0], list(genes["condition"])),
            (axes[1, 2], list(optimizers["condition"])),
        ]:
            for index, condition in enumerate(conditions):
                if condition in runtime_minutes.index:
                    ax.text(
                        index,
                        ax.get_ylim()[0] + 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                        f"{runtime_minutes.loc[condition]:.1f} min",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                        color=palette["gray"],
                    )

    fig.suptitle(f"{tag}: sensitivity analysis", y=1.02)
    return _save(fig, f"{tag}_sensitivity_parameter_panels", outdir)


def plot_fate_ablation_main(metrics_path: Path, outdir: Path, tag: str) -> list[Path]:
    metrics = _read_tsv(metrics_path)
    metrics = metrics.loc[
        metrics["condition"].isin(FATE_CONDITIONS) & metrics["node_type"].eq("leaf")
    ].copy()
    palette = get_palette("default")
    colors = [palette["colors"][4], palette["colors"][1], palette["colors"][0]]

    fig, axes = plt.subplots(2, 2, figsize=DOUBLE_COLUMN)
    for ax, (metric, title) in zip(axes.ravel(), METRICS):
        summary = (
            metrics.groupby("condition", as_index=False)[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary = summary.set_index("condition").reindex(FATE_CONDITIONS)
        x = np.arange(len(FATE_CONDITIONS))
        ax.bar(
            x,
            summary["mean"],
            yerr=summary["std"],
            color=colors,
            edgecolor="none",
            linewidth=0,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [FATE_LABELS[c] for c in FATE_CONDITIONS], rotation=25, ha="right"
        )
        _set_adaptive_ylim(
            ax,
            pd.concat(
                [summary["mean"] - summary["std"], summary["mean"] + summary["std"]]
            ),
        )
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_xlabel("")
    fig.suptitle(f"{tag}: fate-map ablation", y=1.02)
    return _save(fig, f"{tag}_fate_map_ablation_4metric_bars", outdir)


def plot_method_runtime(runtime_path: Path, outdir: Path, tag: str) -> list[Path]:
    runtime = _read_tsv(runtime_path)
    runtime = runtime.loc[runtime["method"].isin(METHOD_ORDER)].copy()
    runtime["method"] = pd.Categorical(
        runtime["method"], categories=METHOD_ORDER, ordered=True
    )
    runtime = runtime.sort_values("method")
    runtime["runtime_minutes"] = runtime["measured_seconds_mean"] / 60.0
    palette = get_palette("default")
    colors = [
        palette["colors"][4] if str(method) == "RegArch" else palette["colors"][0]
        for method in runtime["method"]
    ]

    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN)
    x = np.arange(len(runtime))
    ax.bar(x, runtime["runtime_minutes"], color=colors, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_method_label(str(method)) for method in runtime["method"]],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("Runtime (min, measured mean)")
    ax.set_xlabel("")
    ax.set_title(f"{tag}: method runtime")
    ax.set_yscale("log")
    ax.text(
        0.99,
        0.96,
        "Log scale; precomputed replicates are excluded from measured mean",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=palette["gray"],
    )
    return _save(fig, f"{tag}_method_runtime_bars", outdir)


def _draw_lineage_schematic(ax: plt.Axes) -> None:
    palette = get_palette("default")
    node_style = {
        "root": (palette["colors"][4], "Root\nunseen"),
        "internal": (palette["colors"][1], "Ancestor\nunseen"),
        "leaf": (palette["colors"][0], "Leaf\nobserved"),
    }
    coords = {
        "root": (0.50, 0.88),
        "a1": (0.28, 0.58),
        "a2": (0.72, 0.58),
        "l1": (0.15, 0.22),
        "l2": (0.38, 0.22),
        "l3": (0.62, 0.22),
        "l4": (0.85, 0.22),
    }
    edges = [
        ("root", "a1"),
        ("root", "a2"),
        ("a1", "l1"),
        ("a1", "l2"),
        ("a2", "l3"),
        ("a2", "l4"),
    ]
    for start, end in edges:
        ax.add_patch(
            FancyArrowPatch(
                coords[start],
                coords[end],
                arrowstyle="-",
                mutation_scale=1,
                linewidth=0.8,
                color=palette["gray"],
            )
        )
    for key, kind in [
        ("root", "root"),
        ("a1", "internal"),
        ("a2", "internal"),
        ("l1", "leaf"),
        ("l2", "leaf"),
        ("l3", "leaf"),
        ("l4", "leaf"),
    ]:
        color, _ = node_style[kind]
        ax.add_patch(
            Circle(
                coords[key], 0.065, facecolor=color, edgecolor="white", linewidth=0.8
            )
        )
    ax.text(
        0.50,
        0.98,
        "Fate map links observed leaves\nto hidden ancestral states",
        ha="center",
        va="top",
    )
    ax.text(
        0.50,
        0.06,
        "RegArch performs global signed inference\nfor all nodes on the tree",
        ha="center",
        va="bottom",
    )
    for y, kind in [(0.82, "root"), (0.52, "internal"), (0.16, "leaf")]:
        color, label = node_style[kind]
        ax.add_patch(
            Circle((0.05, y), 0.028, facecolor=color, edgecolor="white", linewidth=0.5)
        )
        ax.text(0.10, y, label.replace("\n", " "), va="center", ha="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _load_ancestor_metrics(results_root: Path, leaf_count: int) -> pd.DataFrame:
    if leaf_count == 10:
        path = (
            results_root
            / "lineage_signal_10leaf_ancestor_recovery"
            / "node_metrics.tsv"
        )
        metrics = _read_tsv(path)
        return metrics.loc[metrics["method"].eq("RegArch")].copy()
    path = (
        results_root
        / "lineage_signal_15leaf_sensitivity"
        / "regarch_ancestor_metrics.tsv"
    )
    ancestor = _read_tsv(path)
    leaf = _read_tsv(
        results_root / "lineage_signal_15leaf_benchmark" / "leaf_metrics.tsv"
    )
    leaf = leaf.loc[leaf["method"].eq("RegArch")].copy()
    leaf["node_type"] = "leaf"
    leaf["developmental_time"] = 1.0
    ancestor["node_type"] = np.where(
        ancestor["cell_type"].eq("Root"), "root", "internal"
    )
    ancestor["method"] = "RegArch"
    if "developmental_time" not in ancestor.columns:
        ancestor["developmental_time"] = np.nan
    columns = [
        "replicate",
        "method",
        "cell_type",
        "true_edge_count",
        "auprc",
        "auroc",
        "precision_at_true_k",
        "signed_precision_at_true_k",
        "node_type",
        "developmental_time",
    ]
    return pd.concat([ancestor[columns], leaf[columns]], ignore_index=True)


def _node_sort_key(node: str) -> tuple[int, int]:
    if node == "Root":
        return (0, 0)
    prefix = node[0]
    try:
        number = int(node[1:])
    except ValueError:
        number = 0
    if prefix == "P":
        return (1, number)
    if prefix == "T":
        return (2, number)
    return (3, number)


def _node_type_color(node_type: str) -> str:
    palette = get_palette("default")
    return {
        "root": palette["colors"][4],
        "internal": palette["colors"][1],
        "leaf": palette["colors"][0],
    }.get(node_type, palette["light_gray"])


def _load_all_node_regarch_metrics(results_root: Path, leaf_count: int) -> pd.DataFrame:
    metrics = _load_ancestor_metrics(results_root, leaf_count)
    metrics = metrics.loc[metrics["method"].eq("RegArch")].copy()
    if "node_type" not in metrics.columns:
        metrics["node_type"] = np.select(
            [
                metrics["cell_type"].eq("Root"),
                metrics["cell_type"].str.startswith("P"),
                metrics["cell_type"].str.startswith("T"),
            ],
            ["root", "internal", "leaf"],
            default="unknown",
        )
    metrics["cell_type"] = metrics["cell_type"].astype(str)
    return metrics


def plot_all_node_boxplots(
    results_root: Path, outdir: Path, leaf_count: int
) -> list[Path]:
    tag = f"{leaf_count}leaf"
    metrics = _load_all_node_regarch_metrics(results_root, leaf_count)
    node_order = sorted(metrics["cell_type"].unique(), key=_node_sort_key)
    node_types = (
        metrics.drop_duplicates("cell_type")
        .set_index("cell_type")
        .reindex(node_order)["node_type"]
        .fillna("unknown")
    )
    colors = [_node_type_color(node_type) for node_type in node_types]

    fig_width = max(DOUBLE_COLUMN[0], 0.22 * len(node_order) + 2.6)
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, DOUBLE_COLUMN[1] * 1.18))
    for ax, (metric, title) in zip(axes.ravel(), METRICS):
        _boxplot_by_category(
            ax,
            metrics,
            "cell_type",
            node_order,
            metric,
            node_order,
            colors,
        )
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=5.8)

        for boundary in [
            index + 0.5
            for index in range(1, len(node_order))
            if node_types.iloc[index] != node_types.iloc[index - 1]
        ]:
            ax.axvline(
                boundary,
                color=get_palette()["light_gray"],
                linewidth=0.55,
                linestyle="--",
            )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            color=_node_type_color("root"),
            label="Root",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            color=_node_type_color("internal"),
            label="Internal",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            color=_node_type_color("leaf"),
            label="Leaf",
        ),
    ]
    axes[0, 1].legend(handles=legend_handles, loc="lower right", ncol=3)
    fig.suptitle(f"{tag}: RegArch performance for every node", y=1.02)
    return _save(fig, f"{tag}_regarch_all_node_4metric_boxplots", outdir)


def plot_all_node_metric_heatmap(
    results_root: Path, outdir: Path, leaf_count: int
) -> list[Path]:
    tag = f"{leaf_count}leaf"
    metrics = _load_all_node_regarch_metrics(results_root, leaf_count)
    node_order = sorted(metrics["cell_type"].unique(), key=_node_sort_key)
    metric_columns = [metric for metric, _ in METRICS]
    heatmap = metrics.groupby("cell_type")[metric_columns].mean().reindex(node_order)
    display = heatmap.rename(columns={metric: label for metric, label in METRICS})

    fig_height = max(DOUBLE_COLUMN[1] * 0.72, 0.18 * len(node_order) + 1.2)
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN[0] * 0.72, fig_height))
    image = ax.imshow(
        display.to_numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0
    )
    ax.set_yticks(np.arange(len(display.index)))
    ax.set_yticklabels(display.index)
    ax.set_xticks(np.arange(len(display.columns)))
    ax.set_xticklabels(display.columns, rotation=25, ha="right")
    ax.set_title(f"{tag}: node-wise mean recovery")

    node_types = (
        metrics.drop_duplicates("cell_type")
        .set_index("cell_type")
        .reindex(node_order)["node_type"]
        .fillna("unknown")
    )
    for index, node_type in enumerate(node_types):
        ax.add_patch(
            plt.Rectangle(
                (-0.72, index - 0.5),
                0.12,
                1.0,
                facecolor=_node_type_color(node_type),
                edgecolor="none",
                clip_on=False,
            )
        )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Mean metric value")
    return _save(fig, f"{tag}_regarch_all_node_metric_heatmap", outdir)


def plot_ancestor_recovery_main(
    results_root: Path, outdir: Path, leaf_count: int
) -> list[Path]:
    tag = f"{leaf_count}leaf"
    metrics = _load_ancestor_metrics(results_root, leaf_count)
    metrics["node_group"] = metrics["node_type"].replace(
        {"root": "Root", "internal": "Ancestor", "leaf": "Leaf"}
    )
    group_order = ["Root", "Ancestor", "Leaf"]
    palette = get_palette("default")

    fig = plt.figure(figsize=DOUBLE_COLUMN)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.25, 0.95], height_ratios=[1, 1])
    ax_scheme = fig.add_subplot(gs[:, 0])
    ax_auprc = fig.add_subplot(gs[0, 1])
    ax_signed = fig.add_subplot(gs[1, 1])
    ax_capability = fig.add_subplot(gs[:, 2])

    _draw_lineage_schematic(ax_scheme)

    _boxplot_by_category(
        ax_auprc,
        metrics,
        "node_group",
        group_order,
        "auprc",
        group_order,
        [palette["colors"][4], palette["colors"][1], palette["colors"][0]],
    )
    ax_auprc.set_title("Ancestral GRN recovery")
    ax_auprc.set_ylabel("AUPRC")
    ax_auprc.set_xlabel("")

    _boxplot_by_category(
        ax_signed,
        metrics,
        "node_group",
        group_order,
        "signed_precision_at_true_k",
        group_order,
        [palette["colors"][4], palette["colors"][1], palette["colors"][0]],
    )
    ax_signed.set_title("Signed edge recovery")
    ax_signed.set_ylabel("Signed precision at true K")
    ax_signed.set_xlabel("")

    row_labels = ["RegArch", "Other 11 methods"]
    col_labels = ["Observed\nleaf GRNs", "Unseen\nancestor GRNs"]
    matrix = np.array([[1, 1], [1, 0]], dtype=float)
    ax_capability.imshow(matrix, cmap="Greens", vmin=0, vmax=1)
    ax_capability.set_xticks(np.arange(2))
    ax_capability.set_xticklabels(col_labels)
    ax_capability.set_yticks(np.arange(2))
    ax_capability.set_yticklabels(row_labels)
    ax_capability.set_title("Inference scope")
    for i in range(2):
        for j in range(2):
            label = "available" if matrix[i, j] else "not inferable"
            ax_capability.text(j, i, label, ha="center", va="center", color="black")
    for spine in ax_capability.spines.values():
        spine.set_visible(False)
    ax_capability.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax_capability.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax_capability.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax_capability.tick_params(which="minor", bottom=False, left=False)
    ax_capability.text(
        0.5,
        1.28,
        "Competing methods are evaluated only on\nobserved leaf cell types because they require\ncell-type-specific RNA/ATAC observations.",
        transform=ax_capability.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=palette["gray"],
    )

    fig.suptitle(
        f"{tag}: RegArch recovers GRNs for unseen ancestral cell types", y=1.02
    )
    return _save(fig, f"{tag}_ancestor_recovery_unseen_grn", outdir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot journal-style benchmark figures."
    )
    parser.add_argument("--results-root", type=Path, default=Path("benchmark/results"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark/results/journal_figures")
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete existing PDF/PNG/TSV files in the output directory.",
    )
    args = parser.parse_args()

    set_paper_style(base_fontsize=7)
    if not args.keep_existing:
        _clean_output_dir(args.output_dir)

    saved: list[Path] = []
    for leaf_count in (10, 15):
        tag = f"{leaf_count}leaf"
        benchmark_dir = args.results_root / f"lineage_signal_{leaf_count}leaf_benchmark"
        ablation_dir = args.results_root / f"lineage_signal_{leaf_count}leaf_ablations"
        atac_dir = (
            args.results_root / f"lineage_signal_{leaf_count}leaf_atac_sensitivity"
        )
        hyper_dir = args.results_root / f"hyperparameter_{leaf_count}leaf_sensitivity"

        saved.extend(
            plot_benchmark_main(
                benchmark_dir / "leaf_metrics.tsv",
                args.output_dir,
                tag,
            )
        )
        saved.extend(
            plot_sensitivity_main(
                atac_dir / "atac_sensitivity_leaf_summary.tsv",
                hyper_dir / "hyperparameter_leaf_delta_vs_baseline.tsv",
                hyper_dir / "regarch_runtime_summary.tsv",
                args.output_dir,
                tag,
            )
        )
        saved.extend(
            plot_fate_ablation_main(
                ablation_dir / "ablation_metrics.tsv",
                args.output_dir,
                tag,
            )
        )
        runtime_path = benchmark_dir / "method_runtime_summary.tsv"
        if runtime_path.exists():
            saved.extend(plot_method_runtime(runtime_path, args.output_dir, tag))
        saved.extend(
            plot_ancestor_recovery_main(args.results_root, args.output_dir, leaf_count)
        )
        saved.extend(
            plot_all_node_boxplots(args.results_root, args.output_dir, leaf_count)
        )
        saved.extend(
            plot_all_node_metric_heatmap(args.results_root, args.output_dir, leaf_count)
        )

    index = pd.DataFrame({"figure_path": [str(path) for path in saved]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.output_dir / "figure_index.tsv", sep="\t", index=False)
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
