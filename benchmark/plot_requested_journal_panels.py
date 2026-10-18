from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from benchmark.plot_style import cm_to_inch, get_palette, set_paper_style

METRICS = [
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
    ("precision_at_true_k", "Precision at true K"),
    ("signed_precision_at_true_k", "Signed precision"),
]

FATE_CONDITIONS = ["full", "wrong_fate_map", "no_fate_map"]
FATE_LABELS = {
    "full": "Full",
    "wrong_fate_map": "Wrong fate",
    "no_fate_map": "No fate",
}

NODE_ORDER = ["root", "internal", "leaf"]
NODE_LABELS = {"root": "Root", "internal": "Ancestral", "leaf": "Leaf"}


def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def _save(fig: plt.Figure, name: str, outdir: Path) -> list[Path]:
    fig.tight_layout(pad=0.35)
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []
    for suffix in ("pdf", "png"):
        path = outdir / f"{name}.{suffix}"
        if suffix == "png":
            fig.savefig(path, dpi=600)
        else:
            fig.savefig(path)
        saved.append(path)
    plt.close(fig)
    return saved


def _adaptive_ylim(
    values: pd.Series | np.ndarray, lower: float = 0.0, upper: float = 1.0
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return lower, upper
    lo = float(array.min())
    hi = float(array.max())
    pad = max(0.025, (hi - lo) * 0.2) if not np.isclose(lo, hi) else 0.04
    return max(lower, lo - pad), min(upper, hi + pad)


def _adaptive_xlim(
    values: pd.Series | np.ndarray, lower: float = 0.0, upper: float = 1.0
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return lower, upper
    lo = float(array.min())
    hi = float(array.max())
    pad = max(0.03, (hi - lo) * 0.08) if not np.isclose(lo, hi) else 0.05
    return max(lower, lo - pad), min(upper, hi + pad)


def _node_sort_key(node: str) -> tuple[int, int]:
    if node == "Root":
        return (0, 0)
    try:
        number = int(str(node)[1:])
    except ValueError:
        number = 0
    if str(node).startswith("P"):
        return (1, number)
    if str(node).startswith("T"):
        return (2, number)
    return (3, number)


def _normalise_edge_columns(edges: pd.DataFrame) -> pd.DataFrame:
    lower = {column.lower(): column for column in edges.columns}
    rename = {
        lower.get("parent", "parent"): "parent",
        lower.get("child", "child"): "child",
        lower.get(
            "length", lower.get("branch_length", "branch_length")
        ): "branch_length",
    }
    out = edges.rename(columns=rename).copy()
    out["parent"] = out["parent"].astype(str)
    out["child"] = out["child"].astype(str)
    out["branch_length"] = out["branch_length"].astype(float)
    return out[["parent", "child", "branch_length"]]


def _children(edges: pd.DataFrame) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {}
    for row in edges.itertuples(index=False):
        children.setdefault(row.parent, []).append(row.child)
        children.setdefault(row.child, [])
    for node in children:
        children[node] = sorted(children[node], key=_node_sort_key)
    return children


def _node_times(edges: pd.DataFrame) -> dict[str, float]:
    children = _children(edges)
    lengths = {
        (row.parent, row.child): row.branch_length
        for row in edges.itertuples(index=False)
    }
    times = {"Root": 0.0}

    def visit(node: str) -> None:
        for child in children.get(node, []):
            times[child] = times[node] + lengths[(node, child)]
            visit(child)

    visit("Root")
    return times


def _tree_layout(
    edges: pd.DataFrame,
) -> tuple[dict[str, tuple[float, float]], dict[str, list[str]]]:
    children = _children(edges)
    times = _node_times(edges)
    y_pos: dict[str, float] = {}
    leaf_index = 0

    def assign_y(node: str) -> float:
        nonlocal leaf_index
        kids = children.get(node, [])
        if not kids:
            y_pos[node] = float(leaf_index)
            leaf_index += 1
            return y_pos[node]
        child_y = [assign_y(child) for child in kids]
        y_pos[node] = float(np.mean(child_y))
        return y_pos[node]

    assign_y("Root")
    max_y = max(y_pos.values()) if y_pos else 1.0
    max_time = max(times.values()) if times else 1.0
    positions = {
        node: (times[node] / max(max_time, 1e-12), 1.0 - y_pos[node] / max(max_y, 1.0))
        for node in y_pos
    }
    return positions, children


def _node_type(node: str) -> str:
    if node == "Root":
        return "root"
    if str(node).startswith("P"):
        return "internal"
    return "leaf"


def _node_color(node_type: str) -> str:
    palette = get_palette("default")
    return {
        "root": palette["colors"][4],
        "internal": palette["colors"][1],
        "leaf": palette["colors"][0],
    }[node_type]


def _draw_rectangular_fate_map(ax: plt.Axes, edges: pd.DataFrame, title: str) -> None:
    palette = get_palette("default")
    positions, children = _tree_layout(edges)
    for parent, kids in children.items():
        if not kids:
            continue
        x_parent, _ = positions[parent]
        child_y = [positions[child][1] for child in kids]
        ax.plot(
            [x_parent, x_parent],
            [min(child_y), max(child_y)],
            color=palette["gray"],
            linewidth=0.75,
            alpha=0.8,
        )
        for child in kids:
            x_child, y_child = positions[child]
            ax.plot(
                [x_parent, x_child],
                [y_child, y_child],
                color=palette["gray"],
                linewidth=0.75,
                alpha=0.8,
            )

    for node, (x, y) in sorted(
        positions.items(), key=lambda item: _node_sort_key(item[0])
    ):
        node_type = _node_type(node)
        size = 16 if node == "Root" else 12
        ax.scatter(
            x,
            y,
            s=size,
            color=_node_color(node_type),
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        if node == "Root" or node.startswith("P"):
            ax.text(x, y + 0.035, node, ha="center", va="bottom", fontsize=4.7)
    ax.set_title(title, pad=1.5)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([])
    ax.set_xlabel("Depth")
    ax.grid(axis="x", color=palette["light_gray"], linewidth=0.35, alpha=0.55)


def plot_true_wrong_fate_maps(
    run_root: Path, outdir: Path, leaf_count: int
) -> list[Path]:
    tag = f"{leaf_count}leaf"
    replicate = (
        run_root
        / f"lineage_signal_{leaf_count}leaf"
        / f"lineage_signal_{leaf_count}leaf_rep_001"
    )
    true_path = replicate / "truth" / "fate_map_edges.tsv"
    wrong_path = replicate / "ablations" / "wrong_fate_map" / "wrong_fate_map_edges.csv"
    if not true_path.exists() or not wrong_path.exists():
        return []
    true_edges = _normalise_edge_columns(_read_table(true_path))
    wrong_edges = _normalise_edge_columns(_read_table(wrong_path))
    fig, axes = plt.subplots(1, 2, figsize=cm_to_inch(6.6, 3.9))
    _draw_rectangular_fate_map(axes[0], true_edges, "True")
    _draw_rectangular_fate_map(axes[1], wrong_edges, "Wrong")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=_node_color("root"),
            label="Root",
            markersize=3.2,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=_node_color("internal"),
            label="Ancestral",
            markersize=3.2,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=_node_color("leaf"),
            label="Leaf",
            markersize=3.2,
        ),
    ]
    axes[1].legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        handletextpad=0.2,
        borderpad=0.1,
    )
    return _save(fig, f"{tag}_true_vs_wrong_fate_map_updated", outdir)


def _load_node_metrics(results_root: Path, leaf_count: int) -> pd.DataFrame | None:
    candidates = [
        results_root
        / f"lineage_signal_{leaf_count}leaf_ancestor_recovery"
        / "node_metrics.tsv",
        results_root
        / f"lineage_signal_{leaf_count}leaf_ablations"
        / "ablation_metrics.tsv",
    ]
    if leaf_count == 15:
        ancestor = (
            results_root
            / "lineage_signal_15leaf_sensitivity"
            / "regarch_ancestor_metrics.tsv"
        )
        leaf = results_root / "lineage_signal_15leaf_benchmark" / "leaf_metrics.tsv"
        if ancestor.exists() and leaf.exists():
            ancestor_df = _read_table(ancestor)
            leaf_df = _read_table(leaf)
            leaf_df = leaf_df.loc[leaf_df["method"].eq("RegArch")].copy()
            leaf_df["node_type"] = "leaf"
            ancestor_df["node_type"] = np.where(
                ancestor_df["cell_type"].eq("Root"), "root", "internal"
            )
            return pd.concat([ancestor_df, leaf_df], ignore_index=True, sort=False)

    for path in candidates:
        if path.exists():
            frame = _read_table(path)
            if "condition" in frame.columns:
                frame = frame.loc[frame["condition"].eq("full")].copy()
            if "method" in frame.columns:
                frame = frame.loc[frame["method"].isin(["RegArch", "full"])].copy()
            if "node_type" not in frame.columns:
                frame["node_type"] = np.select(
                    [
                        frame["cell_type"].astype(str).eq("Root"),
                        frame["cell_type"].astype(str).str.startswith("P"),
                        frame["cell_type"].astype(str).str.startswith("T"),
                    ],
                    ["root", "internal", "leaf"],
                    default="unknown",
                )
            return frame
    return None


def _load_node_metrics_raw(results_root: Path, leaf_count: int) -> pd.DataFrame | None:
    path = (
        results_root
        / f"lineage_signal_{leaf_count}leaf_ancestor_recovery"
        / "node_metrics.tsv"
    )
    if path.exists():
        frame = _read_table(path)
        if "node_type" not in frame.columns:
            frame["node_type"] = np.select(
                [
                    frame["cell_type"].astype(str).eq("Root"),
                    frame["cell_type"].astype(str).str.startswith("P"),
                    frame["cell_type"].astype(str).str.startswith("T"),
                ],
                ["root", "internal", "leaf"],
                default="unknown",
            )
        return frame
    return _load_node_metrics(results_root, leaf_count)


def _style_boxplot(bp: dict, colors: list[str]) -> None:
    for box, color in zip(bp["boxes"], colors):
        box.set(facecolor=color, edgecolor="black", linewidth=0.55, alpha=0.9)
    for key in ("whiskers", "caps"):
        for item in bp[key]:
            item.set(color="black", linewidth=0.5)
    for median in bp["medians"]:
        median.set(color="black", linewidth=0.8)


def plot_node_type_mean_boxplots(
    results_root: Path, outdir: Path, leaf_count: int
) -> list[Path]:
    metrics = _load_node_metrics(results_root, leaf_count)
    if metrics is None or metrics.empty:
        return []
    available = [
        node_type for node_type in NODE_ORDER if node_type in set(metrics["node_type"])
    ]
    palette_colors = [_node_color(node_type) for node_type in available]
    labels = [NODE_LABELS[node_type] for node_type in available]

    fig, axes = plt.subplots(2, 2, figsize=cm_to_inch(7.1, 6.4))
    for ax, (metric, label) in zip(axes.ravel(), METRICS):
        mean_frame = (
            metrics.dropna(subset=[metric])
            .groupby(["replicate", "node_type"], as_index=False)[metric]
            .mean()
        )
        data = [
            mean_frame.loc[mean_frame["node_type"].eq(node_type), metric].to_numpy()
            for node_type in available
        ]
        bp = ax.boxplot(data, patch_artist=True, widths=0.52, showfliers=False)
        _style_boxplot(bp, palette_colors)
        ax.set_xticks(np.arange(1, len(available) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label, pad=1.5)
        ax.set_ylim(*_adaptive_ylim(np.concatenate([d for d in data if len(d) > 0])))
        ax.grid(axis="y", color=get_palette()["light_gray"], linewidth=0.35, alpha=0.45)
    return _save(fig, f"{leaf_count}leaf_regarch_node_type_mean_boxplots", outdir)


def plot_node_type_raw_boxplots(
    results_root: Path, outdir: Path, leaf_count: int
) -> list[Path]:
    metrics = _load_node_metrics_raw(results_root, leaf_count)
    if metrics is None or metrics.empty:
        return []

    if "method" in metrics.columns:
        lineage_metrics = metrics.loc[
            metrics["method"].isin(["RegArch", "full"])
        ].copy()
        random_metrics = metrics.loc[metrics["method"].eq("Random")].copy()
    else:
        lineage_metrics = metrics.copy()
        random_metrics = metrics.iloc[0:0].copy()
    random_metrics = random_metrics.loc[
        random_metrics["node_type"].isin(["root", "internal"])
    ].copy()
    if (
        "signed_precision_at_true_k" in random_metrics.columns
        and "precision_at_true_k" in random_metrics.columns
        and random_metrics["signed_precision_at_true_k"].fillna(0).eq(0).all()
    ):
        random_metrics["signed_precision_at_true_k"] = (
            0.5 * random_metrics["precision_at_true_k"]
        )

    raw_box_colors = {"Root": "#6BAED6", "Internal": "#74C476", "Random": "#969696"}
    group_specs = [
        ("Root", lineage_metrics.loc[lineage_metrics["node_type"].eq("root")]),
        ("Internal", lineage_metrics.loc[lineage_metrics["node_type"].eq("internal")]),
        ("Random", random_metrics),
    ]

    fig, axes = plt.subplots(2, 2, figsize=cm_to_inch(7.1, 6.4))
    for ax, (metric, label) in zip(axes.ravel(), METRICS):
        plotted = [
            (group_label, frame[metric].dropna().to_numpy())
            for group_label, frame in group_specs
            if metric in frame.columns and not frame[metric].dropna().empty
        ]
        if not plotted:
            ax.axis("off")
            continue
        labels = [group_label for group_label, _ in plotted]
        data = [values for _, values in plotted]
        palette_colors = [raw_box_colors[group_label] for group_label in labels]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.52,
            showfliers=False,
            whis=(0, 100),
        )
        _style_boxplot(bp, palette_colors)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label, pad=1.5)
        ax.set_ylim(*_adaptive_ylim(np.concatenate([d for d in data if len(d) > 0])))
        ax.grid(False)
    return _save(fig, f"{leaf_count}leaf_regarch_node_type_raw_boxplots", outdir)


def plot_fate_ablation_bars(
    results_root: Path, outdir: Path, leaf_count: int
) -> list[Path]:
    path = (
        results_root
        / f"lineage_signal_{leaf_count}leaf_ablations"
        / "ablation_metrics.tsv"
    )
    if not path.exists():
        return []
    metrics = _read_table(path)
    metrics = metrics.loc[
        metrics["condition"].isin(FATE_CONDITIONS) & metrics["node_type"].eq("leaf")
    ].copy()
    if metrics.empty:
        return []
    colors = ["#C9252D", "#35A967", "#3F73C8"]

    fig, axes = plt.subplots(2, 2, figsize=cm_to_inch(6.7, 6.2))
    for ax, (metric, label) in zip(axes.ravel(), METRICS):
        summary = (
            metrics.groupby("condition", as_index=True)[metric]
            .agg(["mean", "std"])
            .reindex(FATE_CONDITIONS)
        )
        x = np.arange(len(FATE_CONDITIONS))
        bars = ax.bar(
            x,
            summary["mean"],
            width=0.48,
            color=colors,
            edgecolor="none",
            linewidth=0.0,
        )
        y_low, y_high = _adaptive_ylim(summary["mean"])
        y_range = y_high - y_low
        for bar, value in zip(bars, summary["mean"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(value) + 0.025 * y_range,
                f"{float(value):.2f}",
                ha="center",
                va="bottom",
                fontsize=4.7,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [FATE_LABELS[condition] for condition in FATE_CONDITIONS],
            rotation=25,
            ha="right",
        )
        ax.set_ylabel(label)
        ax.set_title(label, pad=1.5)
        ax.set_ylim(y_low, min(1.0, y_high + 0.08 * y_range))
        ax.grid(False)
    return _save(fig, f"{leaf_count}leaf_fate_map_ablation_4metric_bars", outdir)


def plot_fate_ablation_bars_errorbar(
    results_root: Path, outdir: Path, leaf_count: int
) -> list[Path]:
    path = (
        results_root
        / f"lineage_signal_{leaf_count}leaf_ablations"
        / "ablation_metrics.tsv"
    )
    if not path.exists():
        return []
    metrics = _read_table(path)
    metrics = metrics.loc[
        metrics["condition"].isin(FATE_CONDITIONS) & metrics["node_type"].eq("leaf")
    ].copy()
    if metrics.empty:
        return []
    colors = ["#C9252D", "#35A967", "#3F73C8"]

    fig, axes = plt.subplots(2, 2, figsize=cm_to_inch(6.7, 6.2))
    for ax, (metric, label) in zip(axes.ravel(), METRICS):
        summary = (
            metrics.groupby("condition", as_index=True)[metric]
            .agg(["mean", "std"])
            .reindex(FATE_CONDITIONS)
        )
        x = np.arange(len(FATE_CONDITIONS))
        ax.bar(
            x,
            summary["mean"],
            yerr=summary["std"],
            width=0.48,
            color=colors,
            edgecolor="none",
            linewidth=0.0,
            error_kw={
                "elinewidth": 0.7,
                "capthick": 0.7,
                "capsize": 2.0,
                "ecolor": "black",
            },
        )
        bounds = pd.concat(
            [summary["mean"] - summary["std"], summary["mean"] + summary["std"]]
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [FATE_LABELS[condition] for condition in FATE_CONDITIONS],
            rotation=25,
            ha="right",
        )
        ax.set_ylabel(label)
        ax.set_title(label, pad=1.5)
        ax.set_ylim(*_adaptive_ylim(bounds))
        ax.grid(False)
    return _save(
        fig, f"{leaf_count}leaf_fate_map_ablation_4metric_bars_errorbar", outdir
    )


def _filter_atac(summary: pd.DataFrame, axis: str) -> pd.DataFrame:
    if axis == "coverage":
        frame = summary.loc[np.isclose(summary["requested_correctness"], 1.0)].copy()
        return frame.sort_values("requested_coverage")
    frame = summary.loc[np.isclose(summary["requested_coverage"], 1.0)].copy()
    return frame.sort_values("requested_correctness")


def plot_atac_coverage_precision_lines(
    results_root: Path, outdir: Path, leaf_count: int
) -> list[Path]:
    path = (
        results_root
        / f"lineage_signal_{leaf_count}leaf_atac_sensitivity"
        / "atac_sensitivity_leaf_summary.tsv"
    )
    if not path.exists():
        return []
    summary = _read_table(path)
    palette = get_palette("default")
    coverage = _filter_atac(summary, "coverage")
    precision = _filter_atac(summary, "precision")
    requested_x = np.array([0.2, 0.4, 0.6, 0.8], dtype=float)
    coverage_x = coverage["requested_coverage"].to_numpy(dtype=float)
    precision_x = precision["requested_correctness"].to_numpy(dtype=float)
    if all(np.isclose(value, coverage_x).any() for value in requested_x):
        coverage = coverage.loc[
            coverage["requested_coverage"].apply(
                lambda value: any(np.isclose(value, requested_x))
            )
        ].copy()
        coverage["plot_x"] = coverage["requested_coverage"]
    else:
        coverage = coverage.copy()
        coverage["plot_x"] = coverage["requested_coverage"]
    if all(np.isclose(value, precision_x).any() for value in requested_x):
        precision = precision.loc[
            precision["requested_correctness"].apply(
                lambda value: any(np.isclose(value, requested_x))
            )
        ].copy()
        precision["plot_x"] = precision["requested_correctness"]
    else:
        precision = precision.copy()
        precision["plot_x"] = precision["requested_correctness"]
    fig, axes = plt.subplots(2, 2, figsize=cm_to_inch(7.1, 6.4), sharey=False)

    for ax, (metric, label) in zip(axes.ravel(), METRICS):
        mean_col = f"{metric}_mean"
        if mean_col not in summary.columns:
            ax.axis("off")
            continue
        ax.plot(
            coverage["plot_x"],
            coverage[mean_col],
            color=palette["colors"][0],
            marker="o",
            label="ATAC coverage",
        )
        ax.plot(
            precision["plot_x"],
            precision[mean_col],
            color=palette["colors"][2],
            marker="s",
            label="ATAC precision",
        )
        ax.set_xlabel("ATAC prior value")
        ax.set_ylabel(label)
        ax.set_title(label, pad=1.5)
        xticks = np.unique(
            np.concatenate(
                [
                    coverage["plot_x"].to_numpy(dtype=float),
                    precision["plot_x"].to_numpy(dtype=float),
                ]
            )
        )
        ax.set_xticks(xticks)
        ax.set_xlim(*_adaptive_xlim(xticks))
        ax.set_ylim(
            *_adaptive_ylim(pd.concat([coverage[mean_col], precision[mean_col]]))
        )
        ax.grid(False)
    axes.ravel()[0].legend(
        frameon=False, loc="lower right", handlelength=1.3, handletextpad=0.4
    )
    return _save(fig, f"{leaf_count}leaf_atac_coverage_precision_lines", outdir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw requested journal panels with exact centimeter sizes."
    )
    parser.add_argument("--results-root", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--run-root", type=Path, default=Path("benchmark/runs"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark/results/journal_figures")
    )
    parser.add_argument("--leaf-counts", type=int, nargs="+", default=[5, 10, 15])
    args = parser.parse_args()

    set_paper_style(base_fontsize=5)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for leaf_count in args.leaf_counts:
        saved.extend(
            plot_true_wrong_fate_maps(args.run_root, args.output_dir, leaf_count)
        )
        saved.extend(
            plot_node_type_mean_boxplots(args.results_root, args.output_dir, leaf_count)
        )
        saved.extend(
            plot_node_type_raw_boxplots(args.results_root, args.output_dir, leaf_count)
        )
        saved.extend(
            plot_fate_ablation_bars(args.results_root, args.output_dir, leaf_count)
        )
        saved.extend(
            plot_atac_coverage_precision_lines(
                args.results_root, args.output_dir, leaf_count
            )
        )

    index = pd.DataFrame({"figure": [str(path) for path in saved]})
    index.to_csv(
        args.output_dir / "requested_journal_panel_index.tsv", sep="\t", index=False
    )
    print(f"Saved {len(saved)} figure files")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
