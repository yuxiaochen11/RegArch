from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

from benchmark.plot_style import (
    add_panel_label,
    cm_to_inch,
    export_figure,
    set_paper_style,
)

FACTOR_ORDER = [
    "gain_rate",
    "loss_rate",
    "sign_switch_rate",
    "positive_gain_odds",
    "root_prior_odds",
    "branch_length",
    "terminal_temperature",
    "topology",
]
FACTOR_LABELS = {
    "gain_rate": r"Gain rate ($\lambda_{\mathrm{gain}}$)",
    "loss_rate": r"Loss rate ($\lambda_{\mathrm{loss}}$)",
    "sign_switch_rate": r"Sign-switch rate ($\lambda_{\mathrm{switch}}$)",
    "positive_gain_odds": "Positive fraction ($\\pi_{+}$)\n(odds-scaled)",
    "root_prior_odds": "Root-edge prior odds",
    "branch_length": "Branch-length scale",
    "terminal_temperature": "Terminal-evidence\ntemperature scale",
    "topology": "Tree topology",
}
NODE_ORDER = ["root", "internal", "leaf"]
NODE_LABELS = {"root": "Root", "internal": "Internal ancestor", "leaf": "Leaf"}
NODE_COLORS = {"root": "#D55E00", "internal": "#E69F00", "leaf": "#0072B2"}
NODE_MARKERS = {"root": "o", "internal": "s", "leaf": "^"}
METRIC_LABELS = {
    "auprc": "AUPRC",
    "signed_precision_at_true_k": r"Signed precision at $K_{\mathrm{true}}$",
}


def _portable_index_path(path: Path) -> str:
    """Prefer a repository-relative POSIX path in published figure indexes."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _read_tables(
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    replicate_path = results_dir / "model_misspecification_replicate_metrics.tsv"
    if not replicate_path.exists():
        raise FileNotFoundError(f"Missing sensitivity results: {replicate_path}")
    metrics = pd.read_csv(replicate_path, sep="\t")
    required = {
        "replicate",
        "condition",
        "factor",
        "multiplier",
        "node_type",
        "auprc",
        "signed_precision_at_true_k",
    }
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"Sensitivity metrics are missing columns: {sorted(missing)}")
    if "matched" not in set(metrics["condition"]):
        raise ValueError("Sensitivity metrics must include the matched condition")

    convergence_path = results_dir / "model_misspecification_convergence.tsv"
    convergence = (
        pd.read_csv(convergence_path, sep="\t") if convergence_path.exists() else None
    )
    parameter_path = results_dir / "model_misspecification_inference_parameters.tsv"
    parameters = (
        pd.read_csv(parameter_path, sep="\t") if parameter_path.exists() else None
    )
    return metrics, convergence, parameters


def _quality_problems(
    metrics: pd.DataFrame,
    convergence: pd.DataFrame | None,
) -> list[str]:
    problems = []
    replicate_count = metrics["replicate"].nunique()
    if replicate_count < 3:
        problems.append(
            f"only {replicate_count} independent replicate(s); at least 3 required"
        )
    if convergence is not None:
        required = {"condition", "successful_fits", "target_fits"}
        missing = required - set(convergence)
        if missing:
            problems.append(f"convergence table is missing {sorted(missing)}")
        else:
            pooled = convergence.groupby("condition", as_index=False).agg(
                successful_fits=("successful_fits", "sum"),
                target_fits=("target_fits", "sum"),
            )
            pooled["success_rate"] = pooled["successful_fits"] / pooled[
                "target_fits"
            ].replace(0, np.nan)
            low = pooled.loc[
                pooled["success_rate"].isna() | pooled["success_rate"].lt(0.9),
                "condition",
            ].tolist()
            if low:
                problems.append(
                    "optimizer success below 90% for: " + ", ".join(map(str, low))
                )
    return problems


def _paired_deltas(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    baseline = metrics.loc[
        metrics["condition"].eq("matched"),
        ["replicate", "node_type", metric],
    ].rename(columns={metric: "matched_value"})
    if baseline.duplicated(["replicate", "node_type"]).any():
        raise ValueError(
            "matched metrics must have one row per replicate and node type"
        )
    paired = metrics.merge(
        baseline,
        on=["replicate", "node_type"],
        how="left",
        validate="many_to_one",
    )
    if paired["matched_value"].isna().any():
        raise ValueError("one or more conditions have no replicate-matched baseline")
    paired["delta"] = paired[metric] - paired["matched_value"]
    return paired


def _factor_data(paired: pd.DataFrame, factor: str) -> pd.DataFrame:
    matched = paired.loc[paired["condition"].eq("matched")].copy()
    matched["factor"] = factor
    matched["multiplier"] = 1.0
    return pd.concat(
        [matched, paired.loc[paired["factor"].eq(factor)]],
        ignore_index=True,
    )


def _bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    draws: int = 10000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def _panel_summary(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for (condition, node_type), group in data.groupby(
        ["condition", "node_type"], sort=False
    ):
        values = group["delta"].to_numpy(dtype=float)
        lower, upper = _bootstrap_interval(values, rng)
        rows.append(
            {
                "condition": condition,
                "node_type": node_type,
                "multiplier": float(group["multiplier"].iloc[0]),
                "mean": float(values.mean()),
                "ci_low": lower,
                "ci_high": upper,
                "n": len(values),
            }
        )
    return pd.DataFrame(rows)


def _condition_order(data: pd.DataFrame, factor: str) -> list[str]:
    if factor == "topology":
        return [
            name
            for name in ("matched", "topology_perturbed")
            if name in set(data["condition"])
        ]
    return (
        data[["condition", "multiplier"]]
        .drop_duplicates()
        .sort_values("multiplier")["condition"]
        .tolist()
    )


def _x_axis(
    data: pd.DataFrame,
    factor: str,
    parameters: pd.DataFrame | None,
) -> tuple[dict[str, float], list[float], list[str]]:
    order = _condition_order(data, factor)
    if factor == "topology":
        mapping = {condition: float(index) for index, condition in enumerate(order)}
        return mapping, list(mapping.values()), ["Correct", "Perturbed"][: len(order)]

    multipliers = (
        data[["condition", "multiplier"]]
        .drop_duplicates()
        .set_index("condition")["multiplier"]
    )
    mapping = {
        condition: float(np.log2(multipliers.loc[condition])) for condition in order
    }
    if factor == "positive_gain_odds" and parameters is not None:
        actual = parameters.groupby("condition")["lineage_positive_fraction"].median()
        labels = [f"{actual.loc[condition]:.2f}" for condition in order]
    else:
        labels = [f"{multipliers.loc[condition]:g}×" for condition in order]
    return mapping, [mapping[condition] for condition in order], labels


def _add_preview_watermark(fig: plt.Figure) -> None:
    fig.text(
        0.5,
        0.5,
        "PREVIEW — NOT FOR INFERENCE",
        ha="center",
        va="center",
        rotation=28,
        fontsize=20,
        color="#C96A5A",
        alpha=0.14,
        zorder=100,
    )


def plot_sensitivity_panels(
    metrics: pd.DataFrame,
    parameters: pd.DataFrame | None,
    metric: str,
    output_dir: Path,
    prefix: str,
    preview: bool = False,
    factors: list[str] | None = None,
    filename_suffix: str = "",
) -> tuple[list[Path], pd.DataFrame]:
    set_paper_style(7)
    if factors is None:
        factors = FACTOR_ORDER
    paired = _paired_deltas(metrics, metric)
    panel_data = {factor: _factor_data(paired, factor) for factor in factors}
    panel_summaries = {
        factor: _panel_summary(data, 20260820 + index)
        for index, (factor, data) in enumerate(panel_data.items())
    }
    observed = np.concatenate(
        [
            summary[["ci_low", "ci_high"]].to_numpy().ravel()
            for summary in panel_summaries.values()
            if not summary.empty
        ]
    )
    limit = max(0.05, float(np.nanmax(np.abs(observed))) * 1.16)
    limit = min(1.0, np.ceil(limit * 20.0) / 20.0)

    fig, axes = plt.subplots(2, 4, figsize=cm_to_inch(17.4, 11.5), sharey=True)
    jitter_rng = np.random.default_rng(20260820)
    plot_rows = []
    for panel_index, (ax, factor) in enumerate(zip(axes.flat, factors)):
        data = panel_data[factor].copy()
        summary = panel_summaries[factor].copy()
        x_mapping, ticks, ticklabels = _x_axis(data, factor, parameters)
        data["plot_x"] = data["condition"].map(x_mapping)
        summary["plot_x"] = summary["condition"].map(x_mapping)
        condition_order = _condition_order(data, factor)
        summary["display_label"] = summary["condition"].map(
            dict(zip(condition_order, ticklabels))
        )
        summary.insert(0, "factor", factor)
        summary.insert(1, "metric", metric)
        plot_rows.append(summary.copy())

        for node_type in NODE_ORDER:
            points = data.loc[data["node_type"].eq(node_type)].copy()
            estimates = summary.loc[summary["node_type"].eq(node_type)].sort_values(
                "plot_x"
            )
            if estimates.empty:
                continue
            color = NODE_COLORS[node_type]
            jitter = jitter_rng.uniform(-0.045, 0.045, len(points))
            ax.scatter(
                points["plot_x"] + jitter,
                points["delta"],
                s=5.5,
                color=color,
                alpha=0.24,
                linewidths=0,
                zorder=2,
            )
            yerr = np.vstack(
                [
                    estimates["mean"] - estimates["ci_low"],
                    estimates["ci_high"] - estimates["mean"],
                ]
            )
            ax.errorbar(
                estimates["plot_x"],
                estimates["mean"],
                yerr=yerr,
                color=color,
                marker=NODE_MARKERS[node_type],
                label=NODE_LABELS[node_type],
                linewidth=1.05,
                markersize=3.2,
                capsize=1.8,
                zorder=3,
            )

        ax.axhline(0, color="#666666", linestyle="--", linewidth=0.65, zorder=0)
        matched_x = x_mapping.get("matched", 0.0)
        ax.axvline(matched_x, color="#D9D9D9", linewidth=0.55, zorder=0)
        ax.set_title(FACTOR_LABELS[factor], pad=3.5)
        ax.set_xticks(ticks, ticklabels)
        ax.set_ylim(-limit, limit)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.4)
        add_panel_label(ax, index=panel_index, x=-0.18, y=1.04, fontsize=8)

    unused_axes = list(axes.flat[len(factors) :])
    for ax in unused_axes:
        ax.set_axis_off()
    if unused_axes:
        unused_axes[0].text(
            0.5,
            0.52,
            "Local-scale detail\nTopology is shown in the primary figure\n"
            "A–G use one common y-scale",
            transform=unused_axes[0].transAxes,
            ha="center",
            va="center",
            color="#555555",
            fontsize=6.2,
            linespacing=1.45,
        )

    metric_label = METRIC_LABELS[metric]
    fig.supylabel(f"Δ{metric_label} vs reference", x=0.012, fontsize=7)
    fig.supxlabel("Inference setting", y=0.018, fontsize=7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
        title=(
            f"Independent replicates: n={metrics['replicate'].nunique()}; "
            "error bars: paired-bootstrap 95% CI"
        ),
        title_fontsize=5.5,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        bottom=0.105,
        top=0.88,
        wspace=0.32,
        hspace=0.48,
    )
    if preview:
        _add_preview_watermark(fig)
    filename = f"{prefix}{filename_suffix}_delta_{metric}"
    saved = export_figure(fig, filename, output_dir, formats=("pdf", "png"))
    plt.close(fig)
    plot_data = pd.concat(plot_rows, ignore_index=True)
    plot_data.to_csv(output_dir / f"{filename}_plot_data.tsv", sep="\t", index=False)
    return saved, plot_data


def _pretty_condition(name: str) -> str:
    replacements = {
        "matched": "matched",
        "gain_rate": r"λgain",
        "loss_rate": r"λloss",
        "sign_switch_rate": r"λswitch",
        "positive_gain_odds": "π+ odds",
        "root_prior_odds": "root-prior odds",
        "branch_length": "branch length",
        "terminal_temperature": "terminal temperatures",
        "topology_perturbed": "topology perturbed",
    }
    if name in replacements:
        return replacements[name]
    for prefix, label in replacements.items():
        token = prefix + "_x"
        if name.startswith(token):
            return f"{label} {name.removeprefix(token)}×"
    return name.replace("_", " ")


def plot_convergence(
    convergence: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    preview: bool = False,
) -> tuple[list[Path], pd.DataFrame]:
    set_paper_style(7)
    required = {"condition", "successful_fits", "target_fits"}
    missing = required - set(convergence)
    if missing:
        raise ValueError(f"Convergence table is missing columns: {sorted(missing)}")
    summary = convergence.groupby("condition", as_index=False).agg(
        successful_fits=("successful_fits", "sum"),
        target_fits=("target_fits", "sum"),
    )
    summary["success_rate"] = summary["successful_fits"] / summary[
        "target_fits"
    ].replace(0, np.nan)
    preferred = ["matched"]
    for factor in FACTOR_ORDER:
        subset = summary.loc[
            summary["condition"].str.startswith(factor + "_"), "condition"
        ].tolist()
        preferred.extend(
            sorted(
                subset,
                key=lambda name: float(name.rsplit("_x", 1)[1])
                if "_x" in name
                else 1.0,
            )
        )
    preferred.extend(
        condition for condition in summary["condition"] if condition not in preferred
    )
    rank = {condition: index for index, condition in enumerate(preferred)}
    summary["order"] = summary["condition"].map(rank)
    summary = summary.sort_values("order", ascending=False).reset_index(drop=True)

    fig_height = max(7.2, 0.43 * len(summary) + 2.2)
    fig, ax = plt.subplots(figsize=cm_to_inch(8.5, fig_height))
    y = np.arange(len(summary))
    colors = np.where(summary["success_rate"].ge(0.9), "#0072B2", "#D55E00")
    ax.barh(y, summary["success_rate"].fillna(0), color=colors, height=0.64)
    ax.scatter(
        summary["success_rate"].fillna(0),
        y,
        s=7,
        color=colors,
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )
    for row_index, row in summary.iterrows():
        value = float(row["success_rate"]) if np.isfinite(row["success_rate"]) else 0.0
        text_x = min(value + 0.015, 0.925)
        ax.text(
            text_x,
            row_index,
            f"{int(row['successful_fits'])}/{int(row['target_fits'])}",
            va="center",
            ha="left",
            fontsize=5.5,
        )
    ax.axvline(0.9, color="#666666", linestyle="--", linewidth=0.75)
    ax.set_yticks(y, [_pretty_condition(name) for name in summary["condition"]])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Optimizer success rate")
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.4)
    fig.subplots_adjust(left=0.39, right=0.98, bottom=0.10, top=0.98)
    if preview:
        _add_preview_watermark(fig)
    filename = f"{prefix}_optimizer_qc"
    saved = export_figure(fig, filename, output_dir, formats=("pdf", "png"))
    plt.close(fig)
    summary.drop(columns="order").to_csv(
        output_dir / f"{filename}_plot_data.tsv", sep="\t", index=False
    )
    return saved, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot paired RegArch model-misspecification sensitivity."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/results/journal_figures"),
    )
    parser.add_argument("--prefix", default="model_misspecification")
    parser.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Allow incomplete smoke-test data and add a visible NOT FOR "
            "INFERENCE watermark."
        ),
    )
    args = parser.parse_args()
    metrics, convergence, parameters = _read_tables(args.results_dir)
    problems = _quality_problems(metrics, convergence)
    if problems and not args.preview:
        raise RuntimeError(
            "Refusing to create publication figures: " + "; ".join(problems)
        )
    if problems:
        warnings.warn("Preview quality warnings: " + "; ".join(problems), stacklevel=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    index_rows = []
    for metric, role in (
        ("auprc", "primary"),
        ("signed_precision_at_true_k", "supplementary"),
    ):
        paths, _ = plot_sensitivity_panels(
            metrics,
            parameters,
            metric,
            args.output_dir,
            args.prefix,
            preview=args.preview,
        )
        saved.extend(paths)
        index_rows.extend(
            {
                "role": role,
                "metric": metric,
                "figure": _portable_index_path(path),
            }
            for path in paths
        )
        if metric == "auprc":
            local_paths, _ = plot_sensitivity_panels(
                metrics,
                parameters,
                metric,
                args.output_dir,
                args.prefix,
                preview=args.preview,
                factors=FACTOR_ORDER[:-1],
                filename_suffix="_local",
            )
            saved.extend(local_paths)
            index_rows.extend(
                {
                    "role": "detail",
                    "metric": metric,
                    "figure": _portable_index_path(path),
                }
                for path in local_paths
            )
    if convergence is not None:
        paths, _ = plot_convergence(
            convergence,
            args.output_dir,
            args.prefix,
            preview=args.preview,
        )
        saved.extend(paths)
        index_rows.extend(
            {
                "role": "quality_control",
                "metric": "optimizer_success",
                "figure": _portable_index_path(path),
            }
            for path in paths
        )
    pd.DataFrame(index_rows).to_csv(
        args.output_dir / f"{args.prefix}_figure_index.tsv",
        sep="\t",
        index=False,
    )
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
