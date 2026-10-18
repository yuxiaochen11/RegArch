from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

METHOD_ORDER = [
    "RegArch",
    "Random",
    "CellOracle",
    "SCENIC",
    "SCENICplus",
    "GENIE3",
    "GRNBoost",
    "FigR",
    "Pando",
    "GRaNIE",
    "LINGER",
    "Dictys",
    "scMTNI",
]


def plot_benchmark(
    metrics_path: str | Path,
    status_path: str | Path,
    output_path: str | Path,
) -> None:
    metrics = pd.read_csv(metrics_path, sep="\t")
    metrics = metrics.loc[metrics["cell_type"] != "__mean__"].copy()
    status = json.loads(Path(status_path).read_text(encoding="utf-8"))

    sns.set_theme(style="whitegrid", context="talk")
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 11),
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    figure.subplots_adjust(hspace=0.75, bottom=0.08, top=0.94)

    completed = [method for method in METHOD_ORDER if method in set(metrics["method"])]
    palette = {
        method: (
            "#d55e00"
            if method == "RegArch"
            else "#9e9e9e"
            if method == "Random"
            else "#4c78a8"
        )
        for method in completed
    }
    if completed:
        sns.barplot(
            data=metrics,
            x="method",
            y="auprc",
            order=completed,
            hue="method",
            palette=palette,
            legend=False,
            errorbar="sd",
            ax=axes[0],
        )
        sns.stripplot(
            data=metrics,
            x="method",
            y="auprc",
            order=completed,
            color="black",
            alpha=0.55,
            size=5,
            ax=axes[0],
        )
    axes[0].set_title("Leaf-node GRN reconstruction on shared simulated observations")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("AUPRC")
    axes[0].tick_params(axis="x", rotation=25)

    status_rows = []
    status_labels = {
        "completed": "run",
        "precomputed": "run",
        "not_installed": "not installed",
        "installed_not_run": "not run",
        "incompatible_synthetic_motif_resource": "motif mismatch",
    }
    for method in METHOD_ORDER:
        value = status.get(method, "not_installed")
        completed_flag = value in {"completed", "precomputed"}
        status_rows.append(
            {
                "method": method,
                "available": int(completed_flag),
                "status": value,
            }
        )
    status_frame = pd.DataFrame(status_rows)
    colors = ["#009e73" if value else "#bdbdbd" for value in status_frame["available"]]
    axes[1].bar(status_frame["method"], 1, color=colors)
    for index, row in status_frame.iterrows():
        label = status_labels.get(row["status"], row["status"])
        axes[1].text(
            index, 0.5, label, rotation=90, ha="center", va="center", fontsize=9
        )
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks([])
    axes[1].set_ylabel("Execution")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", labelbottom=False)
    axes[1].set_title("Method execution status in the current environment")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot leaf benchmark results.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_benchmark(args.metrics, args.status, args.output)


if __name__ == "__main__":
    main()
