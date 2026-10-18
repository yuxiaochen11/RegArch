import ast
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize

from regarch.cell_fate_map import FateMap, parse_edge_dict
from regarch.downstream_analysis.network_statistics import (
    get_regulators_for_target_gene,
    get_targets_for_regulator_gene,
)


# Display of network topology difference analysis results.
def plot_regulatory_interactions_along_fate_map(
    grns_dict, path, output_path, figsize=(1.5, 1)
):
    """
    Plot the number of dynamic edges for nodes along a specified path in the gene regulatory network.

    Parameters:
    grns_dict (dict): A dictionary containing gene regulatory networks for each node.
    path (list[str]): A list of node IDs representing the path to trace.
    output_path (str): The path to save the output plot.

    Returns:
    None: This function saves the plot to the specified output path and displays it.
    """
    node_id_list = []
    edge_number_list = []

    for node_id in path:
        edge_num = (grns_dict[node_id] < 0).sum().sum()
        node_id_list.append(node_id)
        edge_number_list.append(edge_num)

    edges_number_df = pd.DataFrame(
        {"node_id": node_id_list, "edge_number": edge_number_list}
    )

    plt.figure(figsize=figsize)
    plt.plot(
        "node_id",
        "edge_number",
        data=edges_number_df,
        linestyle="-",
        marker="o",
        linewidth=0.5,
        markersize=3,
        color="#2EA7E0",
        markerfacecolor="none",
    )

    plt.xlabel("Cell type", fontsize=6, fontname="Arial")
    plt.ylabel("# Regulatory interation", fontsize=6, fontname="Arial")

    plt.xticks(rotation=-90, fontsize=5, fontname="Arial")
    plt.yticks(fontsize=5, fontname="Arial")

    plt.savefig(
        output_path + "dynamic_edge_number.eps", format="eps", bbox_inches="tight"
    )
    plt.show()


def plot_regulatory_genes_along_fate_map(
    input_path,
    regulator_names,
    regulator_id,
    path,
    threshold,
    output_path,
    figsize=(1.5, 1),
):
    """
    Plot the number of target genes regulated by a specific regulator along a specified path.

    Parameters:
    input_path (str): The path to the input data files.
    regulator_names (list[str]): A list of regulator names.
    regulator_id (str): The ID of the regulator to analyze.
    path (list[str]): A list of node IDs representing the path to trace.
    threshold (float): A threshold for determining the significance of target gene interactions.
    output_path (str): The directory where the plot will be saved.

    Returns:
    None: This function saves the plot to the specified output path and displays it.
    """

    node_id_list = []
    target_number_list = []

    for node_id in path:
        target_dict = get_targets_for_regulator_gene(
            regulator_id, node_id, input_path, regulator_names
        )
        target_num = sum(1 for v in target_dict.values() if abs(v) > threshold)
        node_id_list.append(node_id)
        target_number_list.append(target_num)

    target_number_df = pd.DataFrame(
        {"node_id": node_id_list, "target_number": target_number_list}
    )

    plt.figure(figsize=figsize)
    plt.plot(
        "node_id",
        "target_number",
        data=target_number_df,
        linestyle="-",
        marker="o",
        linewidth=0.5,
        markersize=3,
        color="#2EA7E0",
        markerfacecolor="none",
    )

    plt.xlabel("Cell type", fontsize=6, fontname="Arial")
    plt.ylabel("# Target gene", fontsize=6, fontname="Arial")
    plt.xticks(rotation=-90, fontsize=5, fontname="Arial")
    plt.yticks(fontsize=5, fontname="Arial")
    plt.savefig(
        output_path + f"dynamic_target_gene_number_for_{regulator_id}.eps",
        format="eps",
        bbox_inches="tight",
    )
    plt.show()


def plot_dynamic_regulator_count(
    input_path,
    regulator_names,
    target_gene_id,
    path,
    threshold,
    output_path,
    figsize=(1.5, 1),
):
    """
    Plot the number of regulators for a specified target gene across different nodes.

    Parameters:
    input_path (str): The folder path where input data is located.
    regulator_names (list): A list of regulator names to analyze.
    target_gene_id (str): The ID of the target gene for which to count regulators.
    path (list): A list of node IDs to analyze.
    threshold (float): The threshold value to determine significant regulatory interactions.

    Returns:
    None: This function saves the plot to the specified output path and displays it.
    """
    node_id_list = []
    regulator_number_list = []

    for node_id in path:
        regulator_dict = get_regulators_for_target_gene(
            target_gene_id, node_id, input_path, regulator_names
        )
        regulator_num = sum(1 for v in regulator_dict.values() if abs(v) > threshold)
        node_id_list.append(node_id)
        regulator_number_list.append(regulator_num)
    regulator_number_df = pd.DataFrame(
        {"node_id": node_id_list, "regulator_number": regulator_number_list}
    )

    plt.figure(figsize=figsize)
    plt.plot(
        "node_id",
        "regulator_number",
        data=regulator_number_df,
        linestyle="-",
        marker="o",
        linewidth=0.5,
        markersize=3,
        color="#2EA7E0",
        markerfacecolor="none",
    )
    plt.xlabel("Cell type", fontsize=6, fontname="Arial")
    plt.ylabel("# Regulatory gene", fontsize=6, fontname="Arial")

    plt.xticks(rotation=-90, fontsize=5, fontname="Arial")
    plt.yticks(fontsize=5, fontname="Arial")

    plt.savefig(
        output_path + "dynamic_regulator_number_for_" + target_gene_id + ".eps",
        format="eps",
        bbox_inches="tight",
    )
    plt.show()


def rank_regulators_by_target_count(
    target_gene_names,
    regulator_names,
    regulatory_module,
    input_path,
    nodes,
    threshold,
):
    """
    Sort target genes in descending order by their total number of regulatory interactions across all nodes.

    :param target_gene_names: List of target gene IDs.
    :type target_gene_names: list[str]
    :param regulator_names: List of regulator factor names.
    :type regulator_names: list[str]
    :param regulatory_module: Regulation mode, one of 'positive', 'negative', or 'total'.
    :type regulatory_module: str
    :param input_path: Path to the directory containing regulation data files.
    :type input_path: str
    :param nodes: List of node IDs.
    :type nodes: list[str]
    :param threshold: Threshold above (or below) which a regulation is considered active.
    :type threshold: float
    :return: Target genes sorted by descending total regulation count.
    :rtype: list[str]
    """
    # Build the raw count matrix
    mat = pd.DataFrame(index=nodes, columns=target_gene_names, dtype=int)
    for gene in target_gene_names:
        for node in nodes:
            regs = get_regulators_for_target_gene(
                gene, node, input_path, regulator_names
            )
            if regulatory_module == "positive":
                cnt = sum(1 for v in regs.values() if v > threshold)
            elif regulatory_module == "negative":
                cnt = sum(1 for v in regs.values() if v < -threshold)
            else:
                cnt = sum(1 for v in regs.values() if abs(v) > threshold)
            mat.at[node, gene] = cnt

    # Sum across nodes and sort genes
    sorted_genes = mat.sum(axis=0).sort_values(ascending=False).index.tolist()
    return sorted_genes


def plot_target_genes_along_fate_map(
    ordered_genes,
    regulator_names,
    regulatory_module,
    input_path,
    nodes,
    threshold,
    output_path,
    figsize=(2.2, 1.5),
):

    """
    Plot a heatmap of regulatory interaction counts for target genes along a fate map.

    Rows correspond to target genes (in descending order of total regulation),
    columns correspond to nodes (cell types).

    :param ordered_genes: List of target gene IDs sorted by regulation count.
    :type ordered_genes: list[str]
    :param regulator_names: List of regulator factor names.
    :type regulator_names: list[str]
    :param regulatory_module: Regulation mode, one of 'positive', 'negative', or 'total'.
    :type regulatory_module: str
    :param input_path: Path to the directory containing regulation data files.
    :type input_path: str
    :param nodes: List of node IDs.
    :type nodes: list[str]
    :param threshold: Threshold above (or below) which a regulation is considered active.
    :type threshold: float
    :param output_path: Directory in which to save the output heatmap.
    :type output_path: str
    :param figsize: Figure size as (width, height) in inches.
    :type figsize: tuple[float, float]
    :return: None. Saves the heatmap to a file and displays it.
    :rtype: None
    """
    plt.rcParams["font.size"] = 6

    # Select colormap based on regulation mode
    if regulatory_module == "positive":
        cmap_name = "Reds"
    elif regulatory_module == "negative":
        cmap_name = "Blues"
    else:
        cmap_name = "YlGnBu"

    # Build the data matrix in the specified gene order
    df = pd.DataFrame(index=nodes, columns=ordered_genes, dtype=int)
    for gene in ordered_genes:
        counts = []
        for node in nodes:
            regs = get_regulators_for_target_gene(
                gene, node, input_path, regulator_names
            )
            if regulatory_module == "positive":
                cnt = sum(1 for v in regs.values() if v > threshold)
            elif regulatory_module == "negative":
                cnt = sum(1 for v in regs.values() if v < -threshold)
            else:
                cnt = sum(1 for v in regs.values() if abs(v) > threshold)
            counts.append(cnt)
        df[gene] = counts

    # Plot the heatmap (transpose to have genes on the rows)
    plt.figure(figsize=figsize)
    g = sns.heatmap(df.T, cmap=plt.get_cmap(cmap_name))

    plt.setp(g.get_yticklabels(), fontsize=5, fontname="Arial")
    plt.setp(g.get_xticklabels(), fontsize=5, fontname="Arial", rotation=-90)
    g.tick_params(axis="both", which="major", labelsize=5, length=1)
    if g.collections:
        g.collections[0].colorbar.ax.tick_params(labelsize=5, length=1)

    plt.xlabel("Cell type", fontsize=6, fontname="Arial")
    plt.ylabel("Target genes", fontsize=6, fontname="Arial")

    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, "dynamic_regulator_number_heatmap.eps")
    plt.savefig(out_file, format="eps", bbox_inches="tight")
    plt.show()


def plot_regulatory_network_along_fate_map(
    regulator_id,
    grns_dict,
    nodes,
    output_path,
    threshold=0.1,
    figsize=(6.8, 1.8),
):
    """
    Plot the dynamic regulatory network for a specific regulator across multiple nodes.
    """

    def custom_layout(G, edge_lengths, radius=1.0):
        """
        Create a custom layout for the network graph.
        """
        pos = {}
        nodes_list = list(G.nodes())
        num_regulators = len(nodes_list) - 1
        if num_regulators == 0:
            return pos
        angle_step = 2 * np.pi / num_regulators
        pos[nodes_list[0]] = np.array([0, 0])
        for i, rid in enumerate(nodes_list[1:]):
            angle = i * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            pos[rid] = np.array([x, y])

        for (u, v), length in zip(G.edges(), edge_lengths):
            if u == nodes_list[0] or v == nodes_list[0]:
                if u == nodes_list[0]:
                    rid = v
                else:
                    rid = u
                norm_length = length / max(edge_lengths)
                pos[rid] = np.array(
                    [
                        radius
                        * norm_length
                        * np.cos(angle_step * (nodes_list[1:].index(rid))),
                        radius
                        * norm_length
                        * np.sin(angle_step * (nodes_list[1:].index(rid))),
                    ]
                )
        return pos

    def color_map(value):
        """
        Map a value to a color based on its sign.
        """
        if value > 0:
            return plt.cm.Reds(value * 1)
        else:
            return plt.cm.Blues(-value * 2)

    def merge_and_extract_regulator(nodes_list, grns_dictionary, rid):
        """
        Merge regulatory data across nodes and extract the specified regulator's values.
        """
        df_list = [grns_dictionary[node] for node in nodes_list]
        merged_df = pd.concat(df_list, axis=0)
        return merged_df.loc[:, rid]

    def edge_colormap_plot(
        ax,
        regulator_id,
        grns_dictionary,
        node,
        threshold,
        global_min,
        global_max,
        ticks,
    ):
        """
        Create a colored edge plot for the regulatory network of a specific node.
        """
        value_df = pd.DataFrame(grns_dictionary[node].loc[:, regulator_id])
        value_all = merge_and_extract_regulator(nodes, grns_dictionary, regulator_id)

        targets_all = list(set(value_all[abs(value_all) > threshold].index))
        targets = list(value_df[abs(value_df[regulator_id]) > threshold].index)
        values = list(
            value_df.loc[abs(value_df[regulator_id]) > threshold, regulator_id]
        )

        data_all = pd.DataFrame(
            {"regulator": [regulator_id] * len(targets_all), "targets": targets_all}
        )
        data = {
            "regulator": [regulator_id] * len(targets),
            "targets": targets,
            "value": values,
        }
        data_df = pd.DataFrame(data)

        graph_df = pd.merge(data_all, data_df, on=["regulator", "targets"], how="left")
        graph_df["value"] = graph_df["value"].fillna(0)
        graph_df = graph_df.loc[graph_df["targets"] != regulator_id]

        G = nx.from_pandas_edgelist(graph_df, "regulator", "targets")
        edge_lengths = [0.1] * len(graph_df["value"])
        pos = custom_layout(G, edge_lengths, radius=10)

        edge_colors = [color_map(c * 1) for c in graph_df["value"]]
        node_colors = [
            "#E4F0CF" if target == regulator_id else "#DCDCDD" for target in G.nodes()
        ]
        node_sizes = [200 if target == regulator_id else 100 for target in G.nodes()]

        # Draw
        nx.draw(
            G,
            pos,
            node_color=node_colors,
            edge_color=edge_colors,
            width=1,
            with_labels=True,
            node_size=node_sizes,
            font_size=4,
            ax=ax,
        )

        norm = Normalize(vmin=global_min, vmax=global_max)

        color_list = [color_map(v) for v in np.linspace(global_min, global_max, 200)]
        cmap = ListedColormap(color_list)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Regulatory strength", fontsize=6, fontname="Arial")
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{tick:.2f}" for tick in ticks], fontname="Arial")
        cbar.ax.tick_params(labelsize=5)

        ax.set_title(f"{node}", fontsize=7, fontname="Arial")

    fig, axs = plt.subplots(1, len(nodes), figsize=figsize)
    axs = axs.flatten()
    global_min = float("inf")
    global_max = float("-inf")

    for node in nodes:
        values = list(grns_dict[node].loc[:, regulator_id])
        current_min = min(values)
        current_max = max(values)
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)

    ticks = np.linspace(global_min, global_max, num=6)

    for i, node in enumerate(nodes):
        if i < len(axs):
            edge_colormap_plot(
                axs[i],
                regulator_id,
                grns_dict,
                node,
                threshold,
                global_min,
                global_max,
                ticks,
            )

    plt.tight_layout()
    fig.savefig(
        output_path + "dynamic_regulatory_network.eps",
        format="eps",
        bbox_inches="tight",
    )
    plt.show()


def plot_regulator_activity_across_lineages(
    edges_dict,
    input_folder_path,
    regulator_id,
    regulator_names,
    threshold,
    output_path,
    figsize=(3, 3),
):
    """
    Plot the activity of a specified regulator across different lineages in a polar plot.

    Parameters:
    edges_dict (dict): A dictionary containing edges for different lineages.
    input_folder_path (str): The folder path where input data is located.
    regulator_id (str): The ID of the regulator to analyze.
    output_path (str): The path to save the output plot.

    Returns:
    None: This function saves the plot to the specified output path and displays it.
    """

    fate_map_dict = {}
    for lineage in edges_dict.keys():
        edges = parse_edge_dict(edges_dict[lineage])
        fate_map = FateMap(edges)
        fate_map_dict.update({lineage: fate_map})

    lineage_df = pd.DataFrame()

    for lineage in edges_dict.keys():
        input_path = os.path.join(input_folder_path, lineage)
        nodes = fate_map_dict[lineage].nodes.keys()
        for node_id in nodes:
            target_number_list = [lineage, node_id]
            for reg_id in regulator_names:
                target_dict = get_targets_for_regulator_gene(
                    reg_id, node_id, input_path, regulator_names
                )
                target_num = sum(1 for v in target_dict.values() if abs(v) > threshold)
                target_number_list.append(target_num)
            data = pd.DataFrame({node_id: target_number_list})
            lineage_df = pd.concat([lineage_df, data], axis=1)

    lineage_df.index = ["lineage", "node_id"] + regulator_names
    lineage_df = lineage_df.T

    def add_labels(angles, values, labels, offset, ax):
        """
        Add labels to the polar plot at specified angles and values.

        Parameters:
        angles (np.ndarray): Angles at which to place the labels.
        values (np.ndarray): Values at which to place the labels.
        labels (np.ndarray): Labels to display.
        offset (float): Angle offset for positioning labels.
        ax (matplotlib.axes.Axes): The axes to add labels to.

        Returns:
        None: This function adds labels directly to the axes.
        """
        padding = 1
        for angle, value, label in zip(angles, values, labels):
            rotation = np.rad2deg(angle + offset)
            if angle <= np.pi:
                alignment = "right"
                rotation += 180
            else:
                alignment = "left"

            ax.text(
                x=angle,
                y=value + padding,
                s=label,
                ha=alignment,
                va="center",
                rotation=rotation,
                rotation_mode="anchor",
                fontsize=5,
                fontname="Arial",
            )

    VALUES = lineage_df[regulator_id].values
    LABELS = lineage_df["node_id"].values
    GROUP = lineage_df["lineage"].values
    OFFSET = np.pi / 2
    PAD = 2
    ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
    ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
    WIDTH = (2 * np.pi) / len(ANGLES)
    GROUPS_SIZE = [len(i[1]) for i in lineage_df.groupby("lineage")]

    offset = 0
    IDXS = []

    for size in GROUPS_SIZE:
        IDXS += list(range(offset + PAD, offset + size + PAD))
        offset += size + PAD

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    ax.set_theta_offset(OFFSET)
    ax.set_ylim(-20, 40)
    ax.set_frame_on(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    COLORS_GROUP = ["#48ABE1", "#B28247", "#CA87B8", "#EC6655", "#78BF5B", "#F6B956"]
    COLORS = [
        COLORS_GROUP[i] for i, size in enumerate(GROUPS_SIZE) for _ in range(size)
    ]

    ax.bar(
        ANGLES[IDXS],
        VALUES,
        width=WIDTH,
        color=COLORS,
        edgecolor="white",
        linewidth=0.5,
    )

    add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)

    offset = 0
    for lineage, size in zip(edges_dict.keys(), GROUPS_SIZE):
        x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
        ax.plot(x1, [-5] * 50, color="#E1E5E5")

        ax.text(
            np.mean(x1),
            -10,
            lineage,
            color="#333333",
            fontsize=5,
            ha="center",
            va="center",
            fontname="Arial",
        )

        x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=50)
        for y_val in [5, 10, 15, 20, 25, 30]:
            ax.plot(x2, [y_val] * 50, color="#bebebe", lw=0.1)

        offset += size + PAD

    fig.show()
    fig.savefig(
        output_path + "dynamic_regulator_activity.eps",
        format="eps",
        bbox_inches="tight",
    )


def plot_regulatory_strength_along_fate_map(
    input_path,
    path,
    regulator_names,
    target_gene_id,
    output_path,
    figsize=(3, 1.5),
):
    """
    Plot the regulatory strength of different regulators for a specified target gene across various nodes.

    Parameters:
    input_path (str): The folder path containing input files with regulatory data.
    path (list): A list of node IDs to analyze.
    regulator_names (list): A list of regulator names to be used as column headers.
    target_gene_id (str): The ID of the target gene for which to plot regulatory strengths.
    output_path (str): The path where the output plot will be saved.

    Returns:
    None: This function saves the plot to the specified output path and displays it.
    """
    file_path = input_path + "/target_gene_" + target_gene_id + ".csv"
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            lines = file.readlines()

        data = [ast.literal_eval(line.strip()) for line in lines]
        target_gene_data = pd.DataFrame(data).iloc[:, 3:]
        target_gene_data.columns = regulator_names
        target_gene_data.index = list(pd.DataFrame(data).iloc[:, 1])

    scatter_df = []

    for i in range(target_gene_data.shape[1]):  # Loop over regulators
        for j in range(target_gene_data.shape[0]):  # Loop over nodes
            row = [
                target_gene_data.columns[i],
                abs(target_gene_data.iloc[j, i]),
                target_gene_data.index[j],
            ]
            scatter_df.append(row)

    scatter_df = pd.DataFrame(scatter_df, columns=["regulator", "value", "node_id"])
    scatter_df = scatter_df[
        scatter_df["node_id"].isin(path)
    ]  # Filter for specified nodes

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.axhline(0.3, color="#231815", ls=(0, (5, 5)), alpha=1, zorder=0, linewidth=0.5)
    nodes = sorted(scatter_df["node_id"].unique())
    for node, color, edgecolors, marker in zip(
        nodes,
        ["#FAD5D2", "#C7E8FA", "#FCFAD3", "#D2E8CA", "#2ca02c", "#D0F0C0", "#FCFAD3"],
        ["#E60012", "#2171A9", "#EF7C20", "#339939", "#1f77b4", "#98FB98", "#FCFAD3"],
        ["D", "o", "s", "*", "h", "x"],
    ):
        data = scatter_df[scatter_df["node_id"] == node]
        ax.scatter(
            "regulator",
            "value",
            s=14,
            color=color,
            edgecolors=edgecolors,
            linewidths=0.6,
            marker=marker,
            alpha=1,
            data=data,
        )

    ax.tick_params(axis="x", rotation=-90)  # Rotate x-axis tick labels

    fig.suptitle(
        target_gene_id, x=0.5, y=0.975, ha="center", fontsize=8, color="#231815"
    )

    legend = ax.legend(
        loc=(0.05, 1.05), labelspacing=0.2, markerscale=0.8, frameon=False
    )
    for text, node in zip(legend.get_texts(), nodes):
        text.set_text(node)
        text.set_fontsize(6)
        text.set_fontname("Arial")

    legend.set_title("Cell type")
    legend_title = legend.get_title()
    legend_title.set_fontsize(6.5)
    legend_title.set_fontname("Arial")
    legend_title.set_ha("left")

    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["left"].set_color("#231815")
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_color("#231815")
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(length=0.2)

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], size=5, fontname="Arial")
    ax.set_ylabel("Regulatory strength", size=6, fontname="Arial")

    xticks = list(scatter_df["regulator"].unique())
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, size=5, fontname="Arial")
    ax.set_xlabel("Regulatory gene", size=6, fontname="Arial")

    plt.savefig(
        output_path + "dynamic_regulatory_strength.eps",
        format="eps",
        bbox_inches="tight",
    )
    plt.show()


def plot_key_genes_differentiation(
    data, nodes, regulator_names, output_path, figsize=(5, 1)
):
    """
    Plots key regulators based on their regulatory strength and node information.

    Parameters:
    data (DataFrame): DataFrame containing regulatory information including probabilities and activation scales.
    nodes (list): List of node IDs to filter the data.
    regulator_names (list): List of regulator names for labeling the plot.
    output_path (str): Path where the output plot will be saved.

    Returns:
    ax: The matplotlib axis object with the plot.
    """

    data = data[data["node_id"].isin(nodes)]

    fig, ax = plt.subplots(figsize=figsize)

    COLORS = ["#FFFDDF", "#7FCDBB", "#225EA8", "#132860"]
    cmap = mcolors.LinearSegmentedColormap.from_list("colormap", COLORS, N=100)
    norm = plt.Normalize(vmin=0.2, vmax=data["DRS"].max())

    size_values = [1, 0.75, 0.5]
    size_labels = ["1", "0.75", "0.5"]
    size_colors = ["black", "black", "black"]
    size_edgecolors = ["black", "black", "black"]

    for i, size in enumerate(size_values):
        if size <= 0.5:
            size = size * 0.25
        if size <= 0.75 and size > 0.5:
            size = size * 0.5
        ax.scatter(
            [i],
            [-2],
            s=size * 30,
            color=size_colors[i],
            edgecolor=size_edgecolors[i],
            label=size_labels[i],
        )

    for i, regulator in enumerate(regulator_names):
        d = data[data["regulator_id"] == regulator]
        y = d["node_id"]
        x = [i] * len(y)
        color = cmap(norm(d["DRS"]))
        sizes_data = d["DRC"] * 30
        sizes_data = [x * 0.5 if x <= 0.75 else x for x in sizes_data]
        sizes_data = [x * 0.5 if x < 0.5 else x for x in sizes_data]

        ax.scatter(x, y, color=color, s=sizes_data, edgecolor="black", linewidths=0.25)

    ax.set_frame_on(False)
    ax.grid(linewidth=0.5, alpha=0.7, color="#E5E5E5")
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(len(regulator_names)))
    ax.set_xticklabels(
        regulator_names,
        rotation=90,
        color="black",
        fontsize=5,
        fontname="Arial",
        ha="center",
    )
    ax.tick_params(axis="x", which="both", length=0)

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        color="black",
        fontsize=5,
        fontname="Arial",
        va="center",
    )
    ax.tick_params(axis="y", which="both", length=0)

    ax.set_xlabel("Regulatory gene", color="black", fontsize=6, fontname="Arial")
    ax.set_ylabel("Cell type", color="black", fontsize=6, fontname="Arial")

    y_shrunk = 0.01
    y_lower, y_upper = ax.get_ylim()
    ax.set_ylim(y_lower + y_shrunk, y_upper - y_shrunk)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([1, 0.45, 0.02, 0.4])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("DRS", fontsize=6, fontname="Arial")
    cbar.ax.tick_params(labelsize=5, length=0.5)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.7, 1),
        fontsize=5,
        frameon=False,
        title="DRC",
        title_fontsize=6,
    )

    fig.savefig(output_path + "find_diff_genes.eps", format="eps", bbox_inches="tight")
    plt.show()

    return ax


def plot_key_genes_fate_bias(df, child_nodes, output_path, figsize=(1.5, 12)):
    """Plot fate-bias probability and intensity for two child lineages."""

    fbp_0 = "FBP_" + child_nodes[0]
    fbp_1 = "FBP_" + child_nodes[1]
    fbi_0 = "FBI_" + child_nodes[0]
    fbi_1 = "FBI_" + child_nodes[1]
    df = df[~((df[fbp_0] < 1) & (df[fbp_1] < 1))].copy()
    df.reset_index(drop=True, inplace=True)
    genes = df["Regulatory gene"].values
    y_vals = np.arange(len(genes))
    x_endo, x_epi = 0.0, 0.3
    positions = [0.0, 0.4999, 0.5000, 0.5001, 1.0]
    colors = [
        "#FFFFFF",  # pos=0.0
        "#7FCDBB",  # pos=0.4999
        "#7FCDBB",  # pos=0.5000
        "#225EA8",  # pos=0.5001
        "#132860",  # pos=1.0
    ]
    my_cmap = LinearSegmentedColormap.from_list("my_cmap", list(zip(positions, colors)))

    fbi_min = df[[fbi_0, fbi_1]].min().min()
    fbi_max = df[[fbi_0, fbi_1]].max().max()

    fbp_global_min = df[[fbp_0, fbp_1]].min().min()
    fbp_global_max = df[[fbp_0, fbp_1]].max().max()

    def _fate_bias_probability_to_size(probability):
        size_min_small = 1
        size_max_small = 10
        size_max_large = 40
        if probability < 1:
            return size_min_small + (size_max_small - size_min_small) * (
                probability
            ) / (1 - 0)
        if fbp_global_max == 1:
            return size_max_small
        return size_max_small + (size_max_large - size_max_small) * (
            probability - 1
        ) / (fbp_global_max - 1)

    sizes_endo = df[fbp_0].apply(_fate_bias_probability_to_size)
    sizes_epi = df[fbp_1].apply(_fate_bias_probability_to_size)

    fig, ax = plt.subplots(figsize=figsize)
    sc_endo = ax.scatter(
        np.full_like(y_vals, x_endo, dtype=float),
        y_vals,
        c=df[fbi_0],
        s=sizes_endo,
        cmap=my_cmap,
        vmin=fbi_min,
        vmax=fbi_max,
        edgecolors="k",
        alpha=1,
        linewidths=0.25,
    )
    ax.scatter(
        np.full_like(y_vals, x_epi, dtype=float),
        y_vals,
        c=df[fbi_1],
        s=sizes_epi,
        cmap=my_cmap,
        vmin=fbi_min,
        vmax=fbi_max,
        edgecolors="k",
        alpha=1,
        linewidths=0.25,
    )

    ax.set_xticks([x_endo, x_epi])
    ax.set_xticklabels(child_nodes, fontsize=5, fontname="Arial", rotation=90)
    ax.set_yticks(y_vals)
    ax.set_yticklabels(genes, fontsize=5, fontname="Arial")
    ax.set_ylim(-0.7, len(genes) - 0.7)
    ax.set_xlim(-0.2, x_epi + 0.2)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    ax.tick_params(width=0.25, length=1)

    cbar = plt.colorbar(sc_endo, ax=ax)
    cbar.ax.tick_params(labelsize=5, width=0.2)
    cbar.set_label("FBI", fontsize=5, fontname="Arial")
    cbar_ticks = [fbi_min, 0.5, fbi_max]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{fbi_min:.2g}", "0.5", f"{fbi_max:.2g}"])

    fbp_legend_values = [0.5, 1.0, 2.0, fbp_global_max]
    legend_handles = []
    legend_labels = []
    for val in fbp_legend_values:
        val_clamped = max(fbp_global_min, min(val, fbp_global_max))
        size_ = _fate_bias_probability_to_size(val_clamped)
        h = ax.scatter([], [], s=size_, c="gray", edgecolors="k", linewidths=0.25)
        legend_handles.append(h)
        legend_labels.append(f"{val_clamped:.2g}")
    fbp_legend = ax.legend(
        handles=legend_handles,
        labels=legend_labels[0 : (len(legend_labels) - 1)],
        title="FBP",
        bbox_to_anchor=(1.2, 1.0),
        title_fontsize=6,
        prop={"family": "Arial", "size": 5},
    )

    ax.add_artist(fbp_legend)
    # ax.set_xlabel("Child nodes", fontsize=6, fontname='Arial')
    ax.set_ylabel("Regulatory gene", fontsize=6, fontname="Arial")
    # ax.set_title("Dot Heatmap (x=child_nodes, y=Regulatory gene)")

    plt.tight_layout()
    plt.savefig(
        output_path + "find_fate_bias_genes.eps", format="eps", bbox_inches="tight"
    )
    plt.show()


def plot_regulatory_interaction_clusters(weight_matrix, output_path, width=3, height=7):
    """
    Plots a clustered heatmap of the weights in a given edge_cluster.

    Parameters:
    weight_matrix (DataFrame): A matrix containing the weights of edges in the edge_cluster.
    output_path (str): The path where the output plot will be saved.

    Returns:
    None
    """

    COLORS = ["#FFFDDF", "#7FCDBB", "#225EA8"]

    g = sns.clustermap(
        weight_matrix,
        cmap=mcolors.LinearSegmentedColormap.from_list("colormap", COLORS, N=100),
    )

    g.fig.set_size_inches(width, height)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=5, fontname="Arial")
    plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=5, fontname="Arial", rotation=50)

    g.ax_heatmap.tick_params(axis="both", which="major", labelsize=5, length=0)
    g.cax.tick_params(labelsize=5, length=0.5)

    plt.savefig(
        output_path + "edge_cluster_weight.eps", format="eps", bbox_inches="tight"
    )


def plot_regulatory_interactions_by_cell_type(
    locate_edge_cluster_to_nodes_df, output_path, figsize=(1.8, 2.2)
):
    """
    Plots a heatmap showing the association between edge_clusters and nodes.

    Parameters:
    locate_edge_cluster_to_nodes_df (DataFrame): A DataFrame containing the relationship between edge_clusters and nodes.
    output_path (str): The path where the output plot will be saved.

    Returns:
    None
    """

    COLORS = ["#FFFDDF", "#7FCDBB", "#225EA8"]

    plt.figure(figsize=figsize)

    g = sns.heatmap(
        locate_edge_cluster_to_nodes_df,
        fmt=".3f",
        annot=True,
        annot_kws={"fontsize": 5, "fontname": "Arial"},
        cmap=mcolors.LinearSegmentedColormap.from_list("colormap", COLORS, N=100),
    )

    plt.setp(g.get_yticklabels(), fontsize=5, fontname="Arial")

    plt.setp(g.get_xticklabels(), fontsize=5, fontname="Arial", rotation=50)

    g.tick_params(axis="both", which="major", labelsize=5, length=0)

    if g.collections:
        g.collections[0].colorbar.ax.tick_params(labelsize=5, length=0.5)

    plt.xlabel("Edge clusters", fontsize=6, fontname="Arial")
    plt.ylabel("Cell type", fontsize=6, fontname="Arial")

    plt.savefig(
        output_path + "edge_cluster_to_nodes", format="eps", bbox_inches="tight"
    )
