import os
import re

import numpy as np
import pandas as pd
from ete3 import Tree

from regarch.cell_fate_map import FateMap, parse_edge_dict
from regarch.utils.data_io import newick_to_edge_length_dict


# Generate GRNs
def generate_root_grn(target_number: int, regulator_number: int, output_path: str):
    """
    Generate a Gene Regulatory Network (GRN).

    Parameters:
    target_number (int): The number of target genes in the GRN.
    regulator_number (int): The number of regulators in the GRN.
    output_path (str): The directory where the GRN CSV file will be saved.

    Returns:
    pd.DataFrame: A DataFrame representing the generated GRN, with regulators as rows and targets as columns.
    """

    # Initialize a random GRN with values between -1 and 1
    grn0 = np.random.uniform(low=-1, high=1, size=(regulator_number, target_number))

    # Determine the number of zero entries to be set in the GRN
    num_zeros = np.random.randint(
        (target_number * regulator_number) / 2, (target_number * regulator_number) / 1.5
    )

    # Randomly select indices to set to zero
    zero_indices = np.random.choice(grn0.size, size=num_zeros, replace=False)
    grn0.flat[zero_indices] = 0

    # Convert the GRN array to a DataFrame with appropriate indices and columns
    grn0 = pd.DataFrame(
        grn0,
        index=range(target_number, target_number + regulator_number),
        columns=range(0, target_number),
    )

    # Calculate the sum of each column (target)
    tar_sums = np.sum(grn0, axis=0)

    # Identify targets that have a sum of zero
    zero_tar = np.where(tar_sums == 0)[0]

    # For each target with a sum of zero, assign a random regulator value
    for j in zero_tar:
        rand_reg = np.random.randint(0, regulator_number)  # Randomly select a regulator
        rand_hill = np.random.uniform(-5, 5)  # Randomly select a hill coefficient
        grn0[rand_reg, j] = rand_hill  # Assign the random value to the GRN

    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define the output file path
    output_file = os.path.join(output_path, "grn0.csv")

    # Save the GRN DataFrame to a CSV file
    grn0.to_csv(output_file, index=False)

    return grn0


def _transition_probability(
    edge: str, edge_dict: dict[str, float], lambda_transition=0.5
):
    """
    Calculate the transition rate for a given edge based on the edge dictionary.

    Parameters:
    edge (str): The edge for which to calculate the transition rate.
    edge_dict (dict): A dictionary containing edge values associated with their transition probabilities.
    lambda_transition (float, optional): A parameter that influences the transition rate. Default is 0.5.

    Returns:
    float: The calculated transition rate for the specified edge.
    """

    # Calculate the transition rate using the formula: rate = exp(-edge_value * lambda_transition)
    rate = np.exp(-edge_dict[edge] * lambda_transition)

    return rate


def generate_descendant_grns(
    nodes_number: int,
    target_number: int,
    regulator_number: int,
    edge_dict: dict[str, float],
    output_path: str,
    grn0: pd.DataFrame,
):
    """
    Generate descendant Gene Regulatory Networks (GRNs) based on an initial GRN.

    Parameters:
    nodes_number (int): The total number of nodes to generate GRNs for.
    target_number (int): The number of target genes in each GRN.
    regulator_number (int): The number of regulators in each GRN.
    edge_dict (dict): A dictionary containing edge values for calculating transition rates.
    output_path (str): The directory where the descendant GRN CSV files will be saved.
    grn0 (pd.DataFrame): The initial GRN from which descendants will be generated.

    Returns:
    dict: A dictionary containing the generated descendant GRNs, indexed by their names.
    """

    # Initialize a dictionary to store the generated GRNs and an index for their relationships
    grn_dict = {"grn0": grn0}
    grn_index = {}

    # Create mappings for descendant GRNs
    for i in range(int((nodes_number - 1) / 2)):
        grn_index.update(
            {
                "grn{}".format(2 * i + 1): "grn{}".format(i),
                "grn{}".format(2 * i + 2): "grn{}".format(i),
            }
        )

    # Generate each descendant GRN
    for key in grn_index:
        G = np.full(
            (regulator_number, target_number), 100.0
        )  # Initialize GRN with high values

        # Retrieve the parent GRN based on the index
        G_par = grn_dict.get(grn_index[key], globals().get(grn_index[key]))

        # Update the descendant GRN based on the parent GRN's values
        for i in range(G_par.shape[0]):
            for j in range(G_par.shape[1]):
                if G_par.iloc[i, j] != 0:
                    samples = [0, G_par.iloc[i, j]]
                    probability = _transition_probability(
                        grn_index[key] + "->" + key, edge_dict
                    )
                    probs = [1 - probability, probability]
                    b = np.random.choice(samples, p=probs)
                    G[i, j] = b
                else:  # When the parent value is zero
                    a = np.random.uniform(-0.5, 0.5)
                    samples = [a, 0]
                    probability = _transition_probability(
                        grn_index[key] + "->" + key, edge_dict
                    )
                    probs = [1 - probability, probability]
                    b = np.random.choice(samples, p=probs)
                    G[i, j] = b

        G = pd.DataFrame(G)  # Convert the numpy array to a DataFrame
        output_file = os.path.join(
            output_path, key + ".csv"
        )  # Define the output file path
        G.to_csv(output_file, index=False)  # Save the GRN to a CSV file
        grn_dict.update({key: G})  # Update the dictionary with the new GRN

    return grn_dict


# Generate ATAC-seq data
def generate_atac_data(
    grn_dict: dict[str, pd.DataFrame],
    nodes_number: int,
    target_number: int,
    regulator_number: int,
    output_path: str,
):
    """
    Generate ATAC-seq data based on the provided GRNs.

    Parameters:
    grn_dict (dict): A dictionary containing GRNs indexed by their names.
    nodes_number (int): The total number of nodes.
    target_number (int): The number of target genes in each GRN.
    regulator_number (int): The number of regulators in each GRN.
    output_path (str): The directory where the ATAC-seq data will be saved.

    Returns:
    dict: A dictionary containing the generated ATAC-seq data, indexed by their GRN names.
    """

    atac_dict = {}  # Initialize a dictionary to store the ATAC-seq data
    atac_index = {"grn0": "ATAC_grn0"}  # Map the GRN names to ATAC-seq names

    for i in range(int((nodes_number - 1) / 2), nodes_number):
        atac_index[f"grn{i}"] = f"ATAC_grn{i}"

    ATAC = []  # Store all valid ATAC-seq rows

    for key in atac_index:
        G = grn_dict[key]

        for j in range(target_number):
            for i in range(regulator_number):
                row = []
                if G.iloc[i, j] != 0:
                    row.extend([f"regulator_{i}", f"target_{j}"])
                    if G.iloc[i, j] > 0:
                        a = np.random.uniform(0, 1)
                        sample = [0, a]
                        prob = [0.1, 0.9]
                        b = np.random.choice(sample, p=prob)
                    else:
                        a = np.random.uniform(-1, 0)
                        sample = [0, a]
                        prob = [0.1, 0.9]
                        b = np.random.choice(sample, p=prob)
                    row.extend([b, key])
                else:
                    row.extend([f"regulator_{i}", f"target_{j}"])
                    a = np.random.uniform(-0.5, 0.5)
                    sample = [0, a]
                    prob = [0.9, 0.1]
                    b = np.random.choice(sample, p=prob)
                    row.extend([b, key])

                if len(row) > 1 and row[2] != 0:
                    ATAC.append(row)

    # Write to file with 'regulator_' and 'target_' prefixes
    with open(output_path, "w") as f:
        for row in ATAC:
            row_str = ",".join(map(str, row))
            f.write(row_str + "\n")

    atac_dict.update({key: ATAC})
    return atac_dict


# File preparation for SERGIO
def generate_expression_data(
    grn_dict: dict[str, pd.DataFrame],
    nodes_number: int,
    target_number: int,
    regulator_number: int,
    cells_number: int,
    sergio_files_path: str,
    output_path: str,
    clusters: int = 1,
):
    """
    Simulate gene expression data for regulatory networks and prepare input files for SERGIO.

    This function generates File1 and File2 formats based on provided GRNs,
    and uses SERGIO to simulate single-cell gene expression data.

    Parameters:
    grn_dict (dict): A dictionary containing gene regulatory networks (GRNs) indexed by their names.
    nodes_number (int): The total number of nodes in the GRNs.
    target_number (int): The number of target genes.
    regulator_number (int): The number of regulator genes.
    cells_number (int): The number of cells for expression data simulation.
    sergio_files_path (str): The directory where input files (File1 and File2) will be saved.
    output_path (str): The directory where simulated expression data will be saved.
    clusters (int, optional): Number of clusters for File2 generation. Default is 1.

    Returns:
    None: Writes File1, File2, and simulated expression data to the specified directories.
    """

    # SERGIO is only needed for synthetic expression generation.  Keeping the
    # import local prevents the optional simulator from becoming a dependency
    # of fate-map reconstruction and GRN inference on experimental data.
    from regarch.simulation.sergio import SergioSimulator

    # Generate File1 data
    def _write_target_configs():
        file1_dict = {}
        file1_index = {"grn0": "File1_0"}
        for i in range(int((nodes_number - 1) / 2), nodes_number):
            file1_index.update({"grn{}".format(i): "File1_{}".format(i)})

        for key in file1_index:
            network = grn_dict[key]
            input_file = []
            tar_genes = range(0, target_number)

            for i in tar_genes:
                row = [i]
                regulator_number_i = np.sum(network.iloc[:, i] != 0)
                row.append(regulator_number_i)
                for j in range(0, regulator_number):
                    if network.iloc[j, i] != 0:
                        row.append(j + target_number)
                for j in range(0, regulator_number):
                    if network.iloc[j, i] != 0:
                        row.append(network.iloc[j, i])
                for k in range(0, regulator_number_i):
                    row.append(2)
                if len(row) > 1:
                    input_file.append(row)

            output_file = os.path.join(sergio_files_path, file1_index[key] + ".txt")
            with open(output_file, "w") as f:
                for row in input_file:
                    row_str = ",".join(map(str, row))
                    f.write(row_str + "\n")
            file1_dict.update({key: input_file})
        return file1_dict

    # Generate File2 data
    def _write_regulator_config():
        input_file_2 = []
        regulator_genes = range(target_number, target_number + regulator_number)

        for i in regulator_genes:
            row_2 = [i]
            for j in range(clusters):
                a = np.random.uniform(low=0, high=1)
                row_2.append(a)
            input_file_2.append(row_2)

        output_file = os.path.join(sergio_files_path, "File2" + ".txt")
        with open(output_file, "w") as f:
            for row in input_file_2:
                row_str = ",".join(map(str, row))
                f.write(row_str + "\n")

    # Call the helper functions
    os.makedirs(sergio_files_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    _write_target_configs()
    _write_regulator_config()

    # Simulate expression data using SERGIO
    expression_index = {}
    file1_index = {}
    for i in range(int((nodes_number - 1) / 2), nodes_number):
        file1_index.update({"grn{}".format(i): "File1_{}".format(i)})
        expression_index.update({"grn{}".format(i): "Exp_grn{}".format(i)})

    for key in expression_index:
        sim = SergioSimulator(
            number_genes=target_number + regulator_number,
            number_bins=1,
            number_sc=cells_number,
            noise_params=1,
            decays=0.8,
            sampling_state=15,
            noise_type="dpd",
        )

        target_config = os.path.join(sergio_files_path, file1_index[key] + ".txt")
        regulator_config = os.path.join(sergio_files_path, "File2.txt")

        sim.build_graph(
            input_file_targets=target_config,
            input_file_regs=regulator_config,
            shared_coop_state=0,
        )
        sim.simulate()
        expr = sim.get_expressions()
        expression = expr[0, :, :]
        output_file = os.path.join(output_path, expression_index[key] + ".csv")
        np.savetxt(output_file, expression, delimiter=",", header="", fmt="%d")


def prepare_expression_data(
    target_number: int,
    target_gene_names,
    regulator_names,
    input_path: str,
    output_path: str,
):
    """Combine per-node simulator matrices into RegArch's expression table."""
    expression_df = pd.DataFrame()
    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        node_id = re.split(r"[_\.]", filename)[1]
        expression = pd.read_csv(file_path, header=None)
        expression.index = list(target_gene_names) + list(regulator_names)
        expression.columns = ["cell_" + str(i) for i in expression.columns]
        raw_data = expression
        expression_dict = {
            "gene": [],
            "cell": [],
            "value": [],
            "node_id": [],
            "gene_type": [],
        }
        expression_data = pd.DataFrame(expression_dict)
        k = 0
        for i in range(raw_data.shape[0]):
            for j in range(raw_data.shape[1]):
                expression_data.loc[k, "gene"] = raw_data.index[i]
                expression_data.loc[k, "cell"] = raw_data.columns[j]
                expression_data.loc[k, "value"] = raw_data.iloc[i, j]
                expression_data.loc[k, "node_id"] = node_id
                if i < target_number:
                    expression_data.loc[k, "gene_type"] = "target_gene"
                else:
                    expression_data.loc[k, "gene_type"] = "regulator_gene"
                k += 1
        expression_df = pd.concat([expression_df, expression_data], axis=0)
    expression_df.to_csv(output_path, header=False, index=False)
    return expression_df


# Perturb fate map branch length
def perturb_branch_lengths(newick_str, sigma, seed=1234):
    """
    Perturb the branch lengths of a Newick-format tree with Gaussian noise,
    while keeping the length of each root-to-leaf path normalized to 1.

    Parameters:
    newick_str (str): The input Newick string that includes branch lengths.
    sigma (float): Standard deviation of the Gaussian noise (e.g., 0.1 = 10% perturbation).
    seed (int): Random seed for reproducibility (default is 1234).

    Returns:
    FateMap: A FateMap object built from the perturbed tree.
    """
    rng = np.random.default_rng(seed)
    tree = Tree(newick_str, format=1)

    # Collect all root-to-leaf paths
    paths = []
    for leaf in tree.iter_leaves():
        path = list(reversed(leaf.get_ancestors())) + [leaf]
        paths.append(path)

    # Perturb and normalize branch lengths along each path
    edge_lengths = {}
    for path in paths:
        original_lengths = []
        edges = []
        for i in range(1, len(path)):
            edge = (path[i - 1], path[i])
            edges.append(edge)
            original_lengths.append(path[i].dist)

        perturbed = [
            max(0.0, length * (1 + rng.normal(0, sigma))) for length in original_lengths
        ]
        total = sum(perturbed)
        if total <= 0:
            raise ValueError("Perturbed branch lengths sum to zero")
        normalized = [l / total for l in perturbed]

        for edge, new_length in zip(edges, normalized):
            if edge not in edge_lengths:
                edge_lengths[edge] = []
            edge_lengths[edge].append(new_length)

    # Average perturbed lengths over all paths
    for edge, lengths in edge_lengths.items():
        avg_length = np.mean(lengths)
        edge[1].dist = avg_length

    tree_newick = tree.write(format=1, format_root_node=True)

    edge_dict = newick_to_edge_length_dict(tree_newick)
    edge = parse_edge_dict(edge_dict)
    fate_map = FateMap(edge)

    return fate_map
