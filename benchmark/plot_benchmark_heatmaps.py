from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.plot_style import cm_to_inch, export_figure, get_palette, set_paper_style

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


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _load_leaf_summary(results_root: Path, leaf_count: int) -> pd.DataFrame:
    path = (
        results_root / f"lineage_signal_{leaf_count}leaf_benchmark" / "leaf_metrics.tsv"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    metrics = pd.read_csv(path, sep="\t")
    metrics = metrics.loc[
        metrics["cell_type"].ne("__mean__") & metrics["method"].isin(METHOD_ORDER)
    ].copy()
    metrics.loc[
        metrics["method"].isin(UNSIGNED_METHODS),
        "signed_precision_at_true_k",
    ] = np.nan

    summary = metrics.groupby("method", as_index=True)[
        [metric for metric, _ in METRICS]
    ].mean()
    method_order = [method for method in METHOD_ORDER if method in summary.index]
    return summary.reindex(method_order)


def plot_benchmark_heatmap(
    results_root: Path, outdir: Path, leaf_count: int
) -> list[Path]:
    tag = f"{leaf_count}leaf"
    summary = _load_leaf_summary(results_root, leaf_count)
    display = summary.rename(columns={metric: label for metric, label in METRICS})

    palette = get_palette("default")
    fig, ax = plt.subplots(figsize=cm_to_inch(6.2, 4.8))

    masked = np.ma.masked_invalid(display.to_numpy(dtype=float))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="#F2F2F2")
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_yticks(np.arange(len(display.index)))
    ax.set_yticklabels([_method_label(method) for method in display.index])
    ax.set_xticks(np.arange(len(display.columns)))
    ax.set_xticklabels(display.columns, rotation=28, ha="right")

    values = display.to_numpy(dtype=float)
    for column_index, column in enumerate(display.columns):
        column_values = values[:, column_index]
        if np.all(np.isnan(column_values)):
            continue
        best = np.nanmax(column_values)
        winners = np.where(np.isclose(column_values, best, rtol=1e-9, atol=1e-12))[0]
        for row_index in winners:
            ax.text(
                column_index,
                row_index,
                "*",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white" if best < 0.72 else palette["accent"],
            )
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if np.isnan(value):
                continue
            ax.text(
                column_index,
                row_index + 0.18,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=5.0,
                color="white" if value > 0.55 else palette["accent"],
            )

    for row_index, method in enumerate(display.index):
        if method == "RegArch":
            ax.add_patch(
                plt.Rectangle(
                    (-0.52, row_index - 0.5),
                    0.08,
                    1.0,
                    facecolor=palette["colors"][4],
                    edgecolor="none",
                    clip_on=False,
                )
            )

    ax.set_xticks(np.arange(-0.5, len(display.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(display.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label("Mean metric value")
    ax.text(
        0.99,
        -0.28,
        "Best per metric marked by *; gray = unsigned output.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.0,
        color=palette["gray"],
    )
    fig.tight_layout()
    return export_figure(fig, f"{tag}_benchmark_metric_heatmap", outdir=outdir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark metric heatmaps.")
    parser.add_argument("--results-root", type=Path, default=Path("benchmark/results"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark/results/journal_figures")
    )
    parser.add_argument("--leaf-counts", nargs="+", type=int, default=[5, 10, 15])
    args = parser.parse_args()

    set_paper_style(base_fontsize=5)
    saved: list[Path] = []
    for leaf_count in args.leaf_counts:
        saved.extend(
            plot_benchmark_heatmap(args.results_root, args.output_dir, leaf_count)
        )
        plt.close("all")

    index = pd.DataFrame({"figure_path": [str(path) for path in saved]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(
        args.output_dir / "benchmark_heatmap_figure_index.tsv", sep="\t", index=False
    )
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
