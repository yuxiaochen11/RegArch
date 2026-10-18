from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from benchmark.plot_style import (
    DOUBLE_COLUMN,
    export_figure,
    get_palette,
    set_paper_style,
)


def _read_edges(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    lower = {column.lower(): column for column in frame.columns}
    return frame.rename(
        columns={
            lower.get("parent", "parent"): "parent",
            lower.get("child", "child"): "child",
            lower.get(
                "length", lower.get("branch_length", "branch_length")
            ): "branch_length",
        }
    )


def _children(edges: pd.DataFrame) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {}
    for row in edges.itertuples(index=False):
        children.setdefault(row.parent, []).append(row.child)
        children.setdefault(row.child, [])
    for node in children:
        children[node] = sorted(children[node], key=_node_sort_key)
    return children


def _node_sort_key(node: str) -> tuple[int, int]:
    if node == "Root":
        return (0, 0)
    try:
        return (1 if node.startswith("P") else 2, int(node[1:]))
    except ValueError:
        return (3, 0)


def _edge_lengths(edges: pd.DataFrame) -> dict[tuple[str, str], float]:
    return {
        (str(row.parent), str(row.child)): float(row.branch_length)
        for row in edges.itertuples(index=False)
    }


def _cumulative_times(
    edges: pd.DataFrame, children: dict[str, list[str]]
) -> dict[str, float]:
    if "child_time" in edges.columns:
        times = {"Root": 0.0}
        for row in edges.itertuples(index=False):
            times[str(row.child)] = float(row.child_time)
        return times

    lengths = _edge_lengths(edges)
    times = {"Root": 0.0}

    def visit(node: str) -> None:
        for child in children.get(node, []):
            times[child] = times[node] + lengths[(node, child)]
            visit(child)

    visit("Root")
    return times


def _tree_layout(edges: pd.DataFrame) -> dict[str, tuple[float, float]]:
    children = _children(edges)
    times = _cumulative_times(edges, children)
    x_positions: dict[str, float] = {}
    leaf_index = 0

    def assign(node: str) -> float:
        nonlocal leaf_index
        kids = children.get(node, [])
        if not kids:
            x_positions[node] = float(leaf_index)
            leaf_index += 1
            return x_positions[node]
        child_x = [assign(child) for child in kids]
        x_positions[node] = float(np.mean(child_x))
        return x_positions[node]

    assign("Root")
    max_x = max(x_positions.values()) if x_positions else 1
    leaf_times = [
        times[node] for node, kids in children.items() if not kids and node in times
    ]
    max_time = max(leaf_times) if leaf_times else max(times.values())
    return {
        node: (
            x_positions[node] / max(max_x, 1),
            1.0 - times[node] / max(max_time, 1e-12),
        )
        for node in x_positions
    }


def _leaf_times(edges: pd.DataFrame) -> dict[str, float]:
    children = _children(edges)
    times = _cumulative_times(edges, children)
    return {
        node: times[node]
        for node, kids in children.items()
        if not kids and node in times
    }


def _node_type(node: str) -> str:
    if node == "Root":
        return "root"
    if node.startswith("P"):
        return "internal"
    return "leaf"


def _node_color(node: str) -> str:
    palette = get_palette("default")
    return {
        "root": palette["colors"][4],
        "internal": palette["colors"][1],
        "leaf": palette["colors"][0],
    }[_node_type(node)]


def _plot_fate_map(ax: plt.Axes, edges: pd.DataFrame, title: str) -> None:
    pos = _tree_layout(edges)
    palette = get_palette("default")
    for row in edges.itertuples(index=False):
        x0, y0 = pos[row.parent]
        x1, y1 = pos[row.child]
        ax.plot(
            [x0, x1], [y0, y1], color="#8c8c8c", linewidth=1.0, alpha=0.75, zorder=1
        )
    for node, (x, y) in pos.items():
        ax.scatter(
            x,
            y,
            s=42 if node == "Root" else 30,
            color=_node_color(node),
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        offset = 0.028 if _node_type(node) != "leaf" else -0.035
        va = "bottom" if offset > 0 else "top"
        ax.text(x, y + offset, node, ha="center", va=va, fontsize=5.7)
    ax.set_title(title)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xticks([])
    ax.set_yticks([1.0, 0.5, 0.0])
    ax.set_yticklabels(["0", "0.5", "1.0"])
    ax.set_ylabel("Developmental time")
    ax.grid(axis="y", color=palette["light_gray"], linewidth=0.4, alpha=0.45)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)


def plot_fate_maps(replicate_dir: Path, outdir: Path, tag: str) -> list[Path]:
    true_edges = _read_edges(replicate_dir / "truth" / "fate_map_edges.tsv")
    wrong_edges = _read_edges(
        replicate_dir / "ablations" / "wrong_fate_map" / "wrong_fate_map_edges.csv"
    )
    for label, edges in [("true", true_edges), ("wrong", wrong_edges)]:
        leaf_times = _leaf_times(edges)
        unique_times = sorted({round(value, 6) for value in leaf_times.values()})
        if unique_times != [1.0]:
            raise ValueError(
                f"{tag} {label} fate map leaves are not all at time 1: {unique_times}"
            )
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN)
    _plot_fate_map(axes[0], true_edges, "Ground-truth fate map")
    _plot_fate_map(axes[1], wrong_edges, "Wrong fate map")
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", color=_node_color("Root"), label="Root"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=_node_color("P1"),
            label="Internal",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="", color=_node_color("T1"), label="Leaf"
        ),
    ]
    axes[1].legend(handles=handles, loc="lower right", ncol=3)
    fig.suptitle(f"{tag}: true and wrong fate maps", y=1.02)
    return _save(fig, f"{tag}_true_vs_wrong_fate_map", outdir)


def _save(fig: plt.Figure, name: str, outdir: Path) -> list[Path]:
    fig.tight_layout()
    try:
        saved = export_figure(fig, name, outdir=outdir, formats=("pdf", "png"), dpi=600)
    except PermissionError:
        saved = export_figure(
            fig, f"{name}_updated", outdir=outdir, formats=("pdf", "png"), dpi=600
        )
    plt.close(fig)
    return saved


def _parse_regarch_outputs(inferred_dir: Path, regulator_list: Path) -> pd.DataFrame:
    regulators = [
        line.strip() for line in regulator_list.read_text().splitlines() if line.strip()
    ]
    rows = []
    for path in sorted(inferred_dir.glob("target_gene_*.csv")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            values = ast.literal_eval(line)
            target, node = values[0], values[1]
            for regulator, weight in zip(regulators, values[3:]):
                rows.append(
                    {
                        "node": node,
                        "target": target,
                        "regulator": regulator,
                        "inferred_weight": float(weight),
                        "inferred_score": abs(float(weight)),
                        "inferred_sign": int(float(weight) > 0)
                        - int(float(weight) < 0),
                    }
                )
    return pd.DataFrame(rows)


def _load_pair_data(replicate_dir: Path) -> pd.DataFrame:
    truth = pd.read_csv(replicate_dir / "truth" / "true_grn_all_nodes.tsv", sep="\t")
    inferred = _parse_regarch_outputs(
        replicate_dir / "method_outputs" / "raw" / "RegArch_full" / "inferred_grns",
        replicate_dir / "method_inputs" / "RegArch" / "regulator_list.txt",
    )
    merged = truth.merge(inferred, on=["node", "target", "regulator"], how="left")
    merged["inferred_score"] = merged["inferred_score"].fillna(0.0)
    merged["inferred_sign"] = merged["inferred_sign"].fillna(0).astype(int)
    return merged


def _node_order(nodes: list[str]) -> list[str]:
    return sorted(nodes, key=_node_sort_key)


def _edge_sets(
    node_data: pd.DataFrame,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    true_edges = set(
        node_data.loc[node_data["true_edge"].eq(1), ["regulator", "target"]].itertuples(
            index=False, name=None
        )
    )
    k = len(true_edges)
    inferred_top = node_data.sort_values("inferred_score", ascending=False).head(k)[
        ["regulator", "target"]
    ]
    inferred_edges = set(inferred_top.itertuples(index=False, name=None))
    return true_edges, inferred_edges


def _network_positions(
    node_data: pd.DataFrame,
    edge_union: set[tuple[str, str]],
) -> dict[str, tuple[float, float]]:
    genes = sorted(
        set(node_data["regulator"]).union(set(node_data["target"])),
        key=_node_sort_key,
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(genes)
    graph.add_edges_from(edge_union)
    raw = nx.spring_layout(
        graph,
        seed=17,
        k=0.72 / math.sqrt(max(len(genes), 1)),
        iterations=120,
        weight=None,
    )
    xs = np.array([coord[0] for coord in raw.values()])
    ys = np.array([coord[1] for coord in raw.values()])
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    return {
        gene: (
            0.08 + 0.84 * (coord[0] - x_min) / max(x_max - x_min, 1e-12),
            0.08 + 0.84 * (coord[1] - y_min) / max(y_max - y_min, 1e-12),
        )
        for gene, coord in raw.items()
    }


def _gene_roles(node_data: pd.DataFrame) -> dict[str, str]:
    regulators = set(node_data["regulator"])
    targets = set(node_data["target"])
    roles = {}
    for gene in regulators | targets:
        if gene in regulators and gene in targets:
            roles[gene] = "both"
        elif gene in regulators:
            roles[gene] = "regulator"
        else:
            roles[gene] = "target"
    return roles


def _draw_directed_edge(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    alpha: float,
    width: float,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=4.5,
        linewidth=width,
        color=color,
        alpha=alpha,
        shrinkA=4.0,
        shrinkB=4.0,
        connectionstyle="arc3,rad=0.04",
        zorder=1,
    )
    ax.add_patch(patch)


def _draw_grn_node_panel(
    ax: plt.Axes, node_data: pd.DataFrame, title: str
) -> tuple[int, int, int]:
    true_edges, inferred_edges = _edge_sets(node_data)
    tp = true_edges & inferred_edges
    fp = inferred_edges - true_edges
    fn = true_edges - inferred_edges
    edge_union = tp | fp | fn
    pos = _network_positions(node_data, edge_union)
    palette = get_palette("default")
    edge_specs = [
        (fn, "#D55E00", 0.22, 0.75),
        (fp, "#0072B2", 0.22, 0.75),
        (tp, "#009E73", 0.48, 1.0),
    ]
    for edges, color, alpha, width in edge_specs:
        for regulator, target in edges:
            if regulator not in pos or target not in pos:
                continue
            _draw_directed_edge(
                ax,
                pos[regulator],
                pos[target],
                color,
                alpha,
                width,
            )
    roles = _gene_roles(node_data)
    role_colors = {
        "regulator": palette["colors"][3],
        "target": palette["colors"][2],
        "both": "#4B4B8F",
    }
    for role in ["target", "regulator", "both"]:
        genes = [gene for gene, gene_role in roles.items() if gene_role == role]
        if not genes:
            continue
        ax.scatter(
            [pos[item][0] for item in genes],
            [pos[item][1] for item in genes],
            s=12 if role != "both" else 14,
            color=role_colors[role],
            edgecolor="white",
            linewidth=0.2,
            zorder=2,
        )
    ax.text(
        0.02, 0.96, title, transform=ax.transAxes, ha="left", va="top", fontsize=6.2
    )
    ax.text(
        0.02,
        0.04,
        f"TP {len(tp)} | FP {len(fp)} | FN {len(fn)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.2,
        color="#555555",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    return len(tp), len(fp), len(fn)


def plot_grn_pairing_panel(replicate_dir: Path, outdir: Path, tag: str) -> list[Path]:
    pair_data = _load_pair_data(replicate_dir)
    nodes = _node_order(pair_data["node"].unique().tolist())
    ncols = 5 if len(nodes) <= 20 else 6
    nrows = math.ceil(len(nodes) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(DOUBLE_COLUMN[0] * 1.18, 1.25 * nrows)
    )
    axes = np.asarray(axes).ravel()
    for ax, node in zip(axes, nodes):
        _draw_grn_node_panel(ax, pair_data.loc[pair_data["node"].eq(node)], node)
    for ax in axes[len(nodes) :]:
        ax.set_axis_off()
    handles = [
        Line2D([0], [0], color="#009E73", linewidth=1.6, label="Recovered true edge"),
        Line2D([0], [0], color="#0072B2", linewidth=1.6, label="Inferred only"),
        Line2D([0], [0], color="#D55E00", linewidth=1.6, label="Truth missed"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=get_palette()["colors"][3],
            label="Regulator gene",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=get_palette()["colors"][2],
            label="Target gene",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="#4B4B8F",
            label="Regulator and target",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"{tag}: paired ground-truth and RegArch-inferred GRNs by node", y=1.01
    )
    return _save(fig, f"{tag}_paired_truth_inferred_grn_networks", outdir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot fate maps and paired GRN networks."
    )
    parser.add_argument("--run-root", type=Path, default=Path("benchmark/runs"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark/results/journal_figures")
    )
    parser.add_argument("--replicate", type=int, default=1)
    args = parser.parse_args()
    set_paper_style(base_fontsize=7)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for leaf_count in (10, 15):
        tag = f"{leaf_count}leaf"
        replicate_dir = (
            args.run_root
            / f"lineage_signal_{leaf_count}leaf"
            / f"lineage_signal_{leaf_count}leaf_rep_{args.replicate:03d}"
        )
        saved.extend(plot_fate_maps(replicate_dir, args.output_dir, tag))
        saved.extend(plot_grn_pairing_panel(replicate_dir, args.output_dir, tag))
    index_path = args.output_dir / "network_figure_index.tsv"
    pd.DataFrame({"figure_path": [str(path) for path in saved]}).to_csv(
        index_path, sep="\t", index=False
    )
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
