"""Prioritize regulators associated with differentiation and fate bias."""

import pandas as pd

from regarch.utils.data_io import get_dynamic_networks


def get_high_expression_genes(gene_expression_matrix):
    """Return genes expressed above their per-gene median in at least one cell."""

    threshold_values = gene_expression_matrix.median(axis=1)
    binary_matrix = (gene_expression_matrix.T > threshold_values).T.astype(int)
    binary_vector = (binary_matrix.sum(axis=1) >= 1).astype(int)
    high_expression_genes = binary_vector.index[binary_vector == 1].tolist()

    return high_expression_genes


def identify_differentiation_regulators(
    input_path,
    fate_map,
    threshold,
    regulator_names,
    target_gene_names,
    ancestor_node_id,
):
    """
    Identify key regulators involved in cell fate decision events.

    Parameters:
    input_path (str): The path to the input files containing regulatory data.
    fate_map (object): An object that contains information about the nodes and edges.
    threshold (float): A threshold value to filter regulatory effects.
    regulator_names (list[str]): A list of names of regulators.
    target_gene_names (list[str]): A list of names of target genes.
    ancestor_node_id (str): The ID of the ancestor node from which to analyze child nodes.

    Returns:
    DataFrame: A DataFrame containing key regulators and their associated metrics for the given ancestor node.
    """
    # Get child nodes from the ancestor node
    child_nodes = [
        fate_map.nodes[ancestor_node_id].directed_edges[i].end for i in range(2)
    ]

    # Combine high expression target genes from the child nodes
    high_expression_target_genes_in_children = (
        fate_map.nodes[child_nodes[0]].high_expression_genes_in_leaf
        + fate_map.nodes[child_nodes[1]].high_expression_genes_in_leaf
    )

    # Identify high expression target genes present in both target genes and child nodes
    high_expression_target_genes_in_child_nodes = list(
        set(target_gene_names) & set(high_expression_target_genes_in_children)
    )

    # Retrieve the GRN for the ancestor node
    grn_dict = get_dynamic_networks(
        input_path, fate_map, threshold, regulator_names, target_gene_names
    )

    # Create a DataFrame for regulators
    regulators_df = pd.DataFrame(regulator_names, columns=["regulator_id"])
    regulators_df.index = regulators_df["regulator_id"]

    # Calculate the sum of regulatory values for high expression target genes
    column_sums = (
        grn_dict[ancestor_node_id]
        .loc[high_expression_target_genes_in_child_nodes, :]
        .sum(axis=0)
    )
    grn_df = grn_dict[ancestor_node_id].loc[
        high_expression_target_genes_in_child_nodes, :
    ][column_sums[column_sums > 0].index]

    # Count the number of positive and negative regulatory effects
    positive_regulation_number = pd.DataFrame(grn_df.apply(lambda x: (x > 0).sum()))
    negative_regulation_number = pd.DataFrame(grn_df.apply(lambda x: (x < 0).sum()))

    # Sum regulatory values across all target genes
    regulatory_value_sum = pd.DataFrame(grn_df.sum(axis=0))

    # Merge dataframes to create a summary of key regulators
    key_regulators_df = pd.merge(
        regulatory_value_sum,
        positive_regulation_number,
        left_index=True,
        right_index=True,
        how="outer",
    )
    key_regulators_df = pd.merge(
        key_regulators_df,
        negative_regulation_number,
        left_index=True,
        right_index=True,
        how="outer",
    )

    # Rename columns for clarity
    key_regulators_df.columns = [
        "positive_regulatory_strength",
        "positive_regulation_number",
        "negative_regulation_number",
    ]

    # Calculate probability and activation scale
    key_regulators_df["DRS"] = key_regulators_df["positive_regulatory_strength"].div(
        key_regulators_df["positive_regulation_number"]
    )
    key_regulators_df["DRC"] = key_regulators_df["positive_regulation_number"].div(
        key_regulators_df["positive_regulation_number"]
        + key_regulators_df["negative_regulation_number"]
    )
    key_regulators_df = key_regulators_df.drop_duplicates()  # Remove duplicates

    # Join with regulators DataFrame and sort by regulatory metrics
    key_regulators_df = regulators_df.join(key_regulators_df, how="left")
    key_regulators_df = key_regulators_df.sort_values(
        by=["negative_regulation_number", "positive_regulation_number", "DRC"],
        ascending=[True, False, False],
    )

    key_regulators_df["node_id"] = [ancestor_node_id] * key_regulators_df.shape[
        0
    ]  # Add ancestor node ID to the DataFrame

    return key_regulators_df


def merge_regulator_scores(key_regulators_df, child_nodes):
    """Merge the fate-bias scores for both children of an ancestor node."""

    df1 = key_regulators_df[0][["Regulatory gene", "FBI", "FBP"]].reset_index(drop=True)
    df2 = key_regulators_df[1][["Regulatory gene", "FBI", "FBP"]]
    merged_df = df1.merge(
        df2,
        on="Regulatory gene",
        how="left",
        suffixes=(f"_{child_nodes[0]}", f"_{child_nodes[1]}"),
    )

    return merged_df


def filter_regulators(df, regulator_names):
    """Restrict a fate-bias result table to the requested regulators."""

    filtered_df = df[df["Regulatory gene"].isin(regulator_names)].copy()
    return filtered_df


def identify_fate_bias_regulators(
    grn_dict, node_id, target_genes_1, target_genes_2, regulator_names
):
    """Score regulators that favor either branch downstream of a fate decision."""
    regulation_data = []
    grn_df = grn_dict[node_id]

    for regulator in regulator_names:
        regulator_values = grn_df[regulator]

        tar_1_values = regulator_values[target_genes_1]
        tar_2_values = regulator_values[target_genes_2]

        positive_tar_1_count = sum(tar_1_values > 0)
        negative_tar_2_count = sum(tar_2_values < 0)
        positive_tar_1_strength = (
            tar_1_values[tar_1_values > 0].mean() if positive_tar_1_count > 0 else 0
        )
        positive_tar_2_count = sum(tar_2_values > 0)
        negative_tar_1_count = sum(tar_1_values < 0)
        positive_tar_2_strength = (
            tar_2_values[tar_2_values > 0].mean() if positive_tar_2_count > 0 else 0
        )
        regulation_data.append(
            {
                "Regulatory gene": regulator,
                "Tar_1_positive_count": positive_tar_1_count,
                "Tar_2_negative_count": negative_tar_2_count,
                "Tar_1_strength": positive_tar_1_strength,
                "Tar_2_positive_count": positive_tar_2_count,
                "Tar_1_negative_count": negative_tar_1_count,
                "Tar_2_strength": positive_tar_2_strength,
            }
        )
    key_regulators_df = pd.DataFrame(regulation_data)

    key_regulators_of_T1 = key_regulators_df.copy()
    key_regulators_of_T2 = key_regulators_df.copy()

    key_regulators_of_T1["FAP"] = key_regulators_of_T1["Tar_1_positive_count"] / (
        key_regulators_of_T1["Tar_1_positive_count"]
        + key_regulators_of_T1["Tar_2_positive_count"]
    )
    key_regulators_of_T1["FRP"] = key_regulators_of_T1["Tar_2_negative_count"] / (
        key_regulators_of_T1["Tar_2_negative_count"]
        + key_regulators_of_T1["Tar_1_negative_count"]
    )
    key_regulators_of_T1["FBI"] = key_regulators_of_T1["Tar_1_strength"] / (
        key_regulators_of_T1["Tar_1_strength"] + key_regulators_of_T1["Tar_2_strength"]
    )
    key_regulators_of_T1["FBP"] = (
        key_regulators_of_T1["FAP"] + key_regulators_of_T1["FRP"]
    )
    key_regulators_of_T1 = key_regulators_of_T1.fillna(0)

    key_regulators_of_T2["FAP"] = key_regulators_of_T2["Tar_2_positive_count"] / (
        key_regulators_of_T2["Tar_2_positive_count"]
        + key_regulators_of_T2["Tar_1_positive_count"]
    )
    key_regulators_of_T2["FRP"] = key_regulators_of_T2["Tar_1_negative_count"] / (
        key_regulators_of_T2["Tar_1_negative_count"]
        + key_regulators_of_T2["Tar_2_negative_count"]
    )
    key_regulators_of_T2["FRP"] = key_regulators_of_T2["FRP"].fillna(0)
    key_regulators_of_T2["FBI"] = key_regulators_of_T2["Tar_2_strength"] / (
        key_regulators_of_T2["Tar_2_strength"] + key_regulators_of_T2["Tar_1_strength"]
    )
    key_regulators_of_T2["FBP"] = (
        key_regulators_of_T2["FAP"] + key_regulators_of_T2["FRP"]
    )
    key_regulators_of_T2 = key_regulators_of_T2.fillna(0)

    subsetA_T1 = key_regulators_of_T1[key_regulators_of_T1["FBI"] <= 0.5].copy()
    subsetB_T1 = key_regulators_of_T1[key_regulators_of_T1["FBI"] > 0.5].copy()

    subsetA_T1 = subsetA_T1.sort_values(by="FBP", ascending=True)
    subsetB_T1 = subsetB_T1.sort_values(
        by=["FBP", "FBI", "FRP", "FAP"], ascending=[True, False, True, True]
    )
    key_regulators_of_T1 = pd.concat([subsetA_T1, subsetB_T1], ignore_index=True)

    subsetA_T2 = key_regulators_of_T2[key_regulators_of_T2["FBI"] > 0.5].copy()
    subsetB_T2 = key_regulators_of_T2[key_regulators_of_T2["FBI"] <= 0.5].copy()

    key_regulators_of_T2 = pd.concat([subsetA_T2, subsetB_T2], ignore_index=True)

    return key_regulators_of_T1, key_regulators_of_T2
