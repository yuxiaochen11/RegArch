"""Cluster regulatory interactions by their distribution across fate-map nodes."""

import numpy as np
import pandas as pd

from regarch.utils.data_io import get_dynamic_networks


def _euclidean_distances(values, centers):
    """
    Calculate the Euclidean distance between each point in X and each center.

    Parameters:
    X (ndarray): An array of shape (N, D) where N is the number of points and D is the dimensionality.
    centers (ndarray): An array of shape (C, D) where C is the number of centers.

    Returns:
    ndarray: A 2D array of shape (N, C) containing distances from each point to each center.
    """
    sample_count, _ = np.shape(values)
    cluster_count, _ = np.shape(centers)
    tile_x = np.tile(np.expand_dims(values, 1), [1, cluster_count, 1])
    tile_centers = np.tile(np.expand_dims(centers, axis=0), [sample_count, 1, 1])
    dist = np.sum((tile_x - tile_centers) ** 2, axis=-1)
    return np.sqrt(dist)


def _update_centers(weight, values, fuzziness):
    """
    Calculate new centers for each cluster based on current weights.

    Parameters:
    weight (ndarray): An array of shape (N, C) where N is the number of points and C is the number of clusters.
    X (ndarray): An array of shape (N, D) where D is the dimensionality of data points.
    m (float): Fuzziness parameter, typically greater than 1.

    Returns:
    ndarray: An array of shape (C, D) containing new center coordinates.
    """
    _, feature_count = np.shape(values)
    _, cluster_count = np.shape(weight)
    um = weight**fuzziness
    tile_X = np.tile(np.expand_dims(values, 1), [1, cluster_count, 1])
    tile_um = np.tile(np.expand_dims(um, -1), [1, 1, feature_count])
    temp = tile_X * tile_um
    new_C = np.sum(temp, axis=0) / np.expand_dims(np.sum(um, axis=0), axis=-1)
    return new_C


def _update_memberships(values, centers, fuzziness):
    """
    Calculate the membership weights for each point based on distance to centers.

    Parameters:
    X (ndarray): An array of shape (N, D) where D is the dimensionality of data points.
    centers (ndarray): An array of shape (C, D) containing the coordinates of cluster centers.
    m (float): Fuzziness parameter, typically greater than 1.

    Returns:
    ndarray: An array of shape (N, C) containing membership weights for each point to each cluster.
    """
    if fuzziness <= 1:
        raise ValueError("fuzziness must be greater than 1")
    distances = _euclidean_distances(values, centers)
    memberships = np.zeros_like(distances, dtype=float)
    exponent = 2.0 / (fuzziness - 1.0)
    for index, row in enumerate(distances):
        zero_distance = np.isclose(row, 0.0)
        if zero_distance.any():
            memberships[index, zero_distance] = 1.0 / zero_distance.sum()
        else:
            inverse = row ** (-exponent)
            memberships[index] = inverse / inverse.sum()
    return memberships


def cluster_regulatory_interactions(
    input_path,
    fate_map,
    threshold,
    regulator_names,
    target_gene_names,
    regulator_number,
    target_number,
    n_clusters,
    m,
    max_iter=100,
    theta=1e-5,
    seed=0,
):
    """
    Run the Fuzzy C-Means clustering algorithm.

    Parameters:
    input_path (str): The path to the input files containing regulatory data.
    fate_map (object): An object that contains information about nodes and edges.
    threshold (float): A threshold to filter regulatory interactions.
    regulator_names (list[str]): A list of names of regulators.
    target_gene_names (list[str]): A list of names of target genes.
    regulator_number (int): The number of regulators.
    target_number (int): The number of target genes.
    n_clusters (int): The number of clusters to form.
    m (float): Fuzziness parameter, typically greater than 1.
    max_iter (int): Maximum number of iterations to run the algorithm.
    theta (float): Convergence threshold for the weight change.
    seed (int): Random seed for reproducibility.

    Returns:
    tuple: Input data matrix, cluster centers, and membership weight matrix.
    """
    rng = np.random.RandomState(seed)  # Set the random seed
    grns_dict = get_dynamic_networks(
        input_path, fate_map, threshold, regulator_names, target_gene_names
    )

    # Initialize data matrix X
    X = pd.DataFrame(
        0,
        columns=list(grns_dict.keys()),
        index=list(range(regulator_number * target_number)),
    )

    # Populate the data matrix based on regulatory interactions
    for node in list(grns_dict.keys()):
        grn = grns_dict[node]
        for regulator in range(regulator_number):
            for target in range(target_number):
                i = regulator * target_number + target
                if abs(grn.iloc[target, regulator]) > threshold:
                    X.loc[i, node] = 1
                else:
                    X.loc[i, node] = 0

    N, D = np.shape(X)
    weight = rng.uniform(size=(N, n_clusters))  # Initialize weights randomly
    weight = weight / np.sum(weight, axis=1, keepdims=True)  # Normalize weights

    # Main iteration loop for updating weights and centers
    for i in range(max_iter):
        weight_old = weight.copy()  # Save old weights for convergence check
        centers = _update_centers(weight, X, m)
        weight = _update_memberships(X, centers, m)
        if np.linalg.norm(weight - weight_old) < theta:  # Check for convergence
            break

    # Create edge ID mapping for results
    edge_id_dict = {}
    for i in range(target_number * regulator_number):
        edge_id_dict.update(
            {
                i: regulator_names[i // target_number]
                + "->"
                + target_gene_names[i % target_number]
            }
        )

    X = pd.DataFrame(X)
    X.columns = list(grns_dict.keys())
    centers = pd.DataFrame(centers)
    weight_matrix = pd.DataFrame(weight)
    weight_matrix.columns = [
        "EdgeCluster_" + str(i) for i in range(1, (n_clusters + 1))
    ]
    weight_matrix.index = list(edge_id_dict.values())

    return X, centers, weight_matrix


def summarize_interaction_cluster_specificity(
    input_path,
    fate_map,
    threshold,
    regulator_names,
    target_gene_names,
    n_clusters,
    weight_threshold,
    X,
    regulator_number,
    target_number,
    weight_matrix,
):
    """
    Map edge_clusters to nodes based on the regulatory interactions and weights.

    Parameters:
    input_path (str): The path to the input files containing regulatory data.
    fate_map (object): An object that contains information about nodes and edges.
    threshold (float): A threshold to filter regulatory interactions.
    regulator_names (list[str]): A list of names of regulators.
    target_gene_names (list[str]): A list of names of target genes.
    n_clusters (int): The number of clusters formed during clustering.
    weight_threshold (float): A threshold for considering the significance of weights.
    X (DataFrame): A DataFrame representing the regulatory interactions.
    regulator_number (int): The number of regulators.
    target_number (int): The number of target genes.
    weight (DataFrame): A DataFrame containing membership weights for each edge and cluster.

    Returns:
    DataFrame: A DataFrame mapping each cluster to nodes, showing the presence of edge_clusters in nodes.
    """
    # Retrieve the gene regulatory networks for each node
    grns_dict = get_dynamic_networks(
        input_path, fate_map, threshold, regulator_names, target_gene_names
    )

    # Initialize a DataFrame to store the count of edge_clusters in each node
    edge_cluster_to_nodes = pd.DataFrame(
        0, columns=list(grns_dict.keys()), index=list(range(n_clusters))
    )

    # Count the number of edge_clusters associated with each node for each cluster
    for cluster in range(n_clusters):
        for edge in range(target_number * regulator_number):
            node_list = list(
                X.columns[(X.iloc[edge] == 1)]
            )  # Get nodes connected by the edge
            if (
                weight_matrix.iloc[edge, cluster] > weight_threshold
            ):  # Check if weight exceeds threshold
                for j in node_list:
                    edge_cluster_to_nodes.loc[
                        cluster, j
                    ] += 1  # Increment count for the node

    # Create a binary copy of the weight matrix based on the weight threshold
    weight_copy = weight_matrix.copy()
    weight_copy[weight_copy >= weight_threshold] = 1
    weight_copy[weight_copy < weight_threshold] = 0

    # Create a DataFrame to locate edge_clusters in nodes
    locate_edge_cluster_to_nodes_df = pd.DataFrame()
    for cluster in range(n_clusters):
        # Normalize the counts by the sum of weights for the respective cluster
        row = edge_cluster_to_nodes.iloc[cluster, :] / list(weight_copy.sum())[cluster]
        locate_edge_cluster_to_nodes_df = pd.concat(
            [locate_edge_cluster_to_nodes_df, row], axis=1
        )

    # Set the column names to the weight DataFrame's columns
    locate_edge_cluster_to_nodes_df.columns = weight_matrix.columns

    return locate_edge_cluster_to_nodes_df
