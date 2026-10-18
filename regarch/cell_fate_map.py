"""Time-scaled cell fate-map models, inference, and serialization."""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from regarch.utils.constants import NODE_INTERNAL, NODE_LEAF, NODE_ROOT
from regarch.utils.logging_utils import get_logger

logger = get_logger()


class FateMapEdge:
    """Directed fate-map branch with a non-negative temporal weight."""

    def __init__(self, start: str, end: str, weight: float) -> None:
        self.start = start
        self.end = end
        self.weight = weight

    def __repr__(self):
        return f"{self.start}\t{self.end}\t{self.weight}"


class FateMapNode:
    """Node metadata derived from a validated fate-map topology."""

    def __init__(
        self,
        node_id: str,
        directed_edges: List[FateMapEdge] = None,
        upstream_node_id: str = None,
        node_type: str = None,
        high_expression_genes_in_leaf: List[str] = None,
    ) -> None:
        self.node_id = node_id
        self.node_type = node_type
        self.directed_edges = directed_edges
        self.upstream_node_id = upstream_node_id
        self.high_expression_genes_in_leaf = high_expression_genes_in_leaf

    def __repr__(self):
        return f"{self.node_id}\t{self.node_type}\tDirected_Edges:{len(self.directed_edges)}"


class FateMap:
    """Validated rooted tree describing cell-fate relationships and timing."""

    def __init__(
        self,
        edges: List[FateMapEdge],
        high_expression_genes: Dict[str, List[str]] = None,
    ) -> None:
        self.edges = list(edges)
        self.nodes = self._get_nodes(high_expression_genes)

    def _get_nodes(
        self, high_expression_genes: Dict[str, List[str]]
    ) -> Dict[str, FateMapNode]:
        if not self.edges:
            raise ValueError("A fate map must contain at least one edge")

        node_order = []
        children = {}
        parent = {}
        seen_edges = set()
        for edge in self.edges:
            start, end = str(edge.start), str(edge.end)
            if not start or not end:
                raise ValueError("Fate-map node identifiers must be non-empty")
            if start == end:
                raise ValueError(f"Self edge is not allowed: {start}->{end}")
            try:
                weight = float(edge.weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid edge weight for {start}->{end}") from exc
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(
                    f"Edge weight must be finite and non-negative: {start}->{end}"
                )
            edge.weight = weight
            key = (start, end)
            if key in seen_edges:
                raise ValueError(f"Duplicate edge: {start}->{end}")
            seen_edges.add(key)
            if end in parent and parent[end] != start:
                raise ValueError(
                    f"Node {end!r} has multiple parents: {parent[end]!r} and {start!r}"
                )
            parent[end] = start
            children.setdefault(start, []).append(edge)
            children.setdefault(end, [])
            for node_id in (start, end):
                if node_id not in node_order:
                    node_order.append(node_id)

        roots = [node_id for node_id in node_order if node_id not in parent]
        if len(roots) != 1:
            raise ValueError(f"A fate map must have exactly one root; found {roots}")
        root = roots[0]
        for node_id, node_children in children.items():
            if node_children and len(node_children) != 2:
                raise ValueError(
                    f"Internal node {node_id!r} must have exactly two children"
                )

        visited = set()
        active = set()

        def visit(node_id):
            if node_id in active:
                raise ValueError("The fate map contains a directed cycle")
            if node_id in visited:
                return
            active.add(node_id)
            for directed_edge in children[node_id]:
                visit(directed_edge.end)
            active.remove(node_id)
            visited.add(node_id)

        visit(root)
        if visited != set(node_order):
            missing = sorted(set(node_order) - visited)
            raise ValueError(
                f"The fate map is disconnected; unreachable nodes: {missing}"
            )

        nodes: Dict[str, FateMapNode] = {}
        for node_id in node_order:
            node_children = children[node_id]
            if node_id == root:
                node_type = NODE_ROOT
            elif node_children:
                node_type = NODE_INTERNAL
            else:
                node_type = NODE_LEAF
            nodes[node_id] = FateMapNode(
                node_id=node_id,
                directed_edges=list(node_children),
                upstream_node_id=parent.get(node_id),
                node_type=node_type,
                high_expression_genes_in_leaf=(
                    high_expression_genes.get(node_id)
                    if high_expression_genes is not None
                    else None
                ),
            )
        return nodes

    def get_parent_node(self, child_node_id: str) -> FateMapNode:
        parent_node_id = self.nodes[child_node_id].upstream_node_id
        return self.nodes[parent_node_id]

    def get_path(self, leaf_node_id: str) -> List[FateMapNode]:
        path = []
        child = leaf_node_id
        while child is not None:
            path.append(self.nodes[child].node_id)
            child = self.nodes[child].upstream_node_id
        path.reverse()
        return path

    @property
    def node_leaves(self):
        return [
            node_id
            for node_id in self.nodes
            if self.nodes[node_id].node_type == NODE_LEAF
        ]

    @property
    def node_internals(self):
        return [
            node_id
            for node_id in self.nodes
            if self.nodes[node_id].node_type == NODE_INTERNAL
        ]

    @property
    def node_root(self):
        return [
            node_id
            for node_id in self.nodes
            if self.nodes[node_id].node_type == NODE_ROOT
        ][0]

    def __repr__(self):
        return f"Nodes:{len(self.nodes)}, Edges:{len(self.edges)}"


def generate_newick(fate_map: FateMap) -> str:
    """Serialize a validated binary fate map as a named Newick tree."""
    branch_lengths = {fate_map.node_root: 0.0}
    for edge in fate_map.edges:
        branch_lengths[edge.end] = edge.weight

    def build_subtree(node_id: str) -> str:
        children = fate_map.nodes[node_id].directed_edges
        if not children:
            return f"{node_id}:{branch_lengths[node_id]}"
        if len(children) != 2:
            raise ValueError(
                f"Internal node {node_id!r} must have exactly two children"
            )
        subtrees = ",".join(build_subtree(edge.end) for edge in children)
        return f"({subtrees}){node_id}:{branch_lengths[node_id]}"

    return f"{build_subtree(fate_map.node_root)};"


def _compute_default_allele_prob(sc_mat):
    """
    Generate uniform allele emergence probabilities for each site (column) in a single-cell matrix.

    Parameters
    ----------
    sc_mat : np.ndarray
        A 2D array (cells x sites). Each entry indicates the mutation state for a specific site in a specific cell.

    Returns
    -------
    np.ndarray
        A 2D array of the same shape as `sc_mat`. Each element represents a uniform probability
        of emergence for the corresponding site.
    """
    obs_allele_probs_matrix = np.zeros((sc_mat.shape[0], sc_mat.shape[1]))

    # For each column (mutation site):
    for j in range(obs_allele_probs_matrix.shape[1]):
        # Extract unique nonzero mutation states in this column
        muts = np.sort(np.unique(sc_mat[:, j]))
        muts = muts[muts != 0]
        if len(muts) == 0:
            # If no nonzero mutations exist, set the probability to 0
            obs_prob = 0.0
        else:
            # Uniform probability among all observed (nonzero) mutations
            obs_prob = 1.0 / len(muts)

        # Fill the column with the uniform probability
        obs_allele_probs_matrix[:, j] = obs_prob * np.ones(sc_mat.shape[0])

    return obs_allele_probs_matrix


def _compute_mrca_times(t_S, sc_mat, mut_prob, alpha_vector, beta):
    """
    Compute the Most Recent Common Ancestor (MRCA) times for all cell pairs.

    Parameters
    ----------
    t_S : float
        The sampling time.
    sc_mat : np.ndarray
        A 2D array (cells x sites). Each entry indicates the mutation state for a specific site in a specific cell.
    mut_prob : np.ndarray
        A 2D array (cells x sites). Each entry indicates the probability of a mutation at a site for a particular cell.
    alpha_vector : np.ndarray
        A 1D array of alpha parameters (one per site).
    beta : float
        A parameter for the (1 + beta * t)^(-alpha) model (scale factor in the integrals).

    Returns
    -------
    np.ndarray
        A 2D array (cells x cells), where entry (i, j) is the estimated MRCA time between cell i and j.
    """
    sc_mat = np.asarray(sc_mat)
    mut_prob = np.asarray(mut_prob, dtype=float)
    alpha_vector = np.asarray(alpha_vector, dtype=float)
    if sc_mat.ndim != 2 or sc_mat.shape[0] < 2 or sc_mat.shape[1] < 1:
        raise ValueError("sc_mat must contain at least two cells and one site")
    if mut_prob.shape != sc_mat.shape or not np.all(np.isfinite(mut_prob)):
        raise ValueError("mut_prob must be finite and have the same shape as sc_mat")
    if alpha_vector.shape != (sc_mat.shape[1],):
        raise ValueError("alpha_vector must contain one value per mutation site")
    if not np.all(np.isfinite(alpha_vector)) or np.any(alpha_vector <= 0):
        raise ValueError("alpha_vector must contain finite positive values")
    if not np.isfinite(t_S) or float(t_S) <= 0:
        raise ValueError("t_S must be finite and positive")
    if not np.isfinite(beta) or float(beta) <= 0:
        raise ValueError("beta must be finite and positive")

    def compute_integrate(
        val, t_S, gene1, gene2, case1, case2, case3, case4, case5, alpha_vector, beta
    ):
        """
        Inner function to compute the negative log-likelihood given a candidate t_MRCA value.
        """
        t_MRCA = val

        # Compute terms (1 + beta*t)^{-alpha_i} for relevant timepoints
        # Note: I1 is always 1, so we don't store it
        I2 = (1 + beta * t_MRCA) ** (-alpha_vector)  # (1 + beta*t_MRCA)^(-alpha)
        I3 = (1 + beta * t_S) ** (-alpha_vector)  # (1 + beta*t_S)^(-alpha)
        I4 = (1 + 2 * beta * t_S - beta * t_MRCA) ** (-alpha_vector)

        # case1: Both cells share the same nonzero mutation
        prob_case1 = (
            gene1[case1]
            + (-gene1[case1] + gene1[case1] ** 2) * I2[case1]
            - 2 * (gene1[case1] ** 2) * I3[case1]
            + (gene1[case1] ** 2) * I4[case1]
        )

        # case2: Both cells have no mutation (0)
        prob_case2 = I4[case2]

        # case3: Cells have different nonzero mutations
        prob_case3 = (
            (I2[case3] - 2 * I3[case3] + I4[case3]) * gene1[case3] * gene2[case3]
        )

        # case4: First cell mutated, second cell not mutated
        prob_case4 = (I3[case4] - I4[case4]) * gene1[case4]

        # case5: First cell not mutated, second cell mutated
        prob_case5 = (I3[case5] - I4[case5]) * gene2[case5]

        probabilities = (prob_case1, prob_case2, prob_case3, prob_case4, prob_case5)
        if any(
            np.any(~np.isfinite(probability)) or np.any(probability <= 0)
            for probability in probabilities
        ):
            return np.inf

        # Sum the log probabilities across all sites.
        loglikelihood = sum(np.log(probability).sum() for probability in probabilities)

        # We return negative log-likelihood for minimization
        return -loglikelihood

    MRCA_matrix = np.full((sc_mat.shape[0], sc_mat.shape[0]), float(t_S))

    # For each pair of cells (k, j):
    for k in range(sc_mat.shape[0]):
        for j in range(k + 1, sc_mat.shape[0]):
            cell1 = sc_mat[k, :]
            cell2 = sc_mat[j, :]

            gene1 = mut_prob[k, :]
            gene2 = mut_prob[j, :]

            # Define cases for each site
            case1 = (cell1 == cell2) & (cell1 != 0)
            case2 = (cell1 == cell2) & (cell1 == 0)
            case3 = (cell1 != cell2) & (cell1 != 0) & (cell2 != 0)
            case4 = (cell1 != 0) & (cell2 == 0)
            case5 = (cell1 == 0) & (cell2 != 0)

            # Minimize negative log-likelihood wrt t_MRCA
            pred_t_M = minimize_scalar(
                compute_integrate,
                args=(
                    t_S,
                    gene1,
                    gene2,
                    case1,
                    case2,
                    case3,
                    case4,
                    case5,
                    alpha_vector,
                    beta,
                ),
                method="bounded",
                bounds=(0.0, float(t_S)),
            )
            if not pred_t_M.success or not np.isfinite(pred_t_M.fun):
                raise RuntimeError(
                    f"MRCA optimization failed for cell pair ({k}, {j}): "
                    f"{pred_t_M.message}"
                )

            # Update symmetric matrix
            MRCA_matrix[k, j] = pred_t_M.x
            MRCA_matrix[j, k] = pred_t_M.x

    return MRCA_matrix


def _compute_alpha(t_S, sc_mat, mut_prob, beta, MRCA_matrix):
    """
    Compute the alpha parameters for each mutation site, given the MRCA matrix and other model parameters.

    Parameters
    ----------
    t_S : float
        The sampling time.
    sc_mat : np.ndarray
        A 2D array (cells x sites). Each entry indicates the mutation state for a specific site in a specific cell.
    mut_prob : np.ndarray
        A 2D array (cells x sites). Each entry indicates the probability of a mutation at a site for a particular cell.
    beta : float
        A parameter for the (1 + beta * t)^(-alpha) model (scale factor in the integrals).
    MRCA_matrix : np.ndarray
        A 2D array (cells x cells). The MRCA times between each pair of cells.

    Returns
    -------
    np.ndarray
        A 1D array of optimized alpha parameters (one per mutation site).
    """
    sc_mat = np.asarray(sc_mat)
    mut_prob = np.asarray(mut_prob, dtype=float)
    if sc_mat.ndim != 2 or sc_mat.shape[0] < 2 or sc_mat.shape[1] < 1:
        raise ValueError("sc_mat must contain at least two cells and one site")
    if mut_prob.shape != sc_mat.shape or not np.all(np.isfinite(mut_prob)):
        raise ValueError("mut_prob must be finite and have the same shape as sc_mat")
    if not np.isfinite(beta) or float(beta) <= 0:
        raise ValueError("beta must be finite and positive")
    mrca_time_to_distance(MRCA_matrix, t_S)
    MRCA_matrix = np.asarray(MRCA_matrix, dtype=float)

    def compute_alpha_likelihood(val, t_S, sc_mat, mut_prob, MRCA_matrix, beta, i):
        """
        Inner function to compute the negative log-likelihood for a single alpha_i value,
        given site i and the MRCA matrix for all cell pairs.
        """
        alpha_i = val
        loglikelihood = 0

        # For each pair of cells (k, j):
        for k in range(sc_mat.shape[0]):
            for j in range(k + 1, sc_mat.shape[0]):
                cell1 = sc_mat[k, i]
                cell2 = sc_mat[j, i]
                gene1 = mut_prob[k, i]
                gene2 = mut_prob[j, i]

                I2 = (1 + beta * MRCA_matrix[k, j]) ** (-alpha_i)
                I3 = (1 + beta * t_S) ** (-alpha_i)
                I4 = (1 + 2 * beta * t_S - beta * MRCA_matrix[k, j]) ** (-alpha_i)

                # Evaluate the probability using the same cases as _compute_mrca_times.
                if cell1 == cell2:
                    if cell1 != 0:
                        prob = (
                            gene1
                            + (-gene1 + gene1**2) * I2
                            - 2 * (gene1**2) * I3
                            + (gene1**2) * I4
                        )
                    else:
                        prob = I4
                else:
                    if (cell1 != 0) and (cell2 != 0):
                        prob = (I2 - 2 * I3 + I4) * gene1 * gene2
                    elif (cell1 != 0) and (cell2 == 0):
                        prob = (I3 - I4) * gene1
                    elif (cell1 == 0) and (cell2 != 0):
                        prob = (I3 - I4) * gene2

                if not np.isfinite(prob) or prob <= 0:
                    return np.inf
                loglikelihood += np.log(prob)

        # Return negative log-likelihood
        return -loglikelihood

    alpha_vector = np.zeros(sc_mat.shape[1])

    # Optimize alpha_i for each site i
    for i in range(len(alpha_vector)):
        pred_alpha = minimize_scalar(
            compute_alpha_likelihood,
            args=(t_S, sc_mat, mut_prob, MRCA_matrix, beta, i),
            method="bounded",
            bounds=(1e-6, 10.0),
        )
        if not pred_alpha.success or not np.isfinite(pred_alpha.fun):
            raise RuntimeError(
                f"Alpha optimization failed for mutation site {i}: "
                f"{pred_alpha.message}"
            )
        alpha_vector[i] = pred_alpha.x

    return alpha_vector


def _alternating_optimization(t_S, sc_mat, mut_prob, beta, max_iter=100, tol=1e-6):
    """
    Perform an alternating optimization of MRCA_matrix and alpha_vector.

    Parameters
    ----------
    t_S : float
        Sampling time.
    sc_mat : np.ndarray
        A 2D array (cells x sites). Each entry indicates the mutation state for a specific site in a specific cell.
    mut_prob : np.ndarray
        A 2D array (cells x sites). Each entry indicates the probability of a mutation at a site for a particular cell.
    beta : float
        A parameter for (1 + beta * t)^(-alpha) (scale factor in the integrals).
    max_iter : int, optional
        Maximum number of iterations for alternating optimization. Default is 100.
    tol : float, optional
        Convergence threshold for alpha_vector. Default is 1e-6.

    Returns
    -------
    alpha_vector : np.ndarray
        The optimized alpha_vector.
    MRCA_matrix : np.ndarray
        The optimized MRCA_matrix.
    """
    sc_mat = np.asarray(sc_mat)
    mut_prob = np.asarray(mut_prob, dtype=float)
    if sc_mat.ndim != 2 or sc_mat.shape[0] < 2 or sc_mat.shape[1] < 1:
        raise ValueError("sc_mat must contain at least two cells and one site")
    if mut_prob.shape != sc_mat.shape or not np.all(np.isfinite(mut_prob)):
        raise ValueError("mut_prob must be finite and have the same shape as sc_mat")
    if not np.isfinite(t_S) or float(t_S) <= 0:
        raise ValueError("t_S must be finite and positive")
    if not np.isfinite(beta) or float(beta) <= 0:
        raise ValueError("beta must be finite and positive")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or float(tol) <= 0:
        raise ValueError("tol must be finite and positive")

    num_cells = sc_mat.shape[0]
    # Initialize alpha_vector (one alpha per site)
    alpha_vector = np.ones(sc_mat.shape[1])
    # Initialize MRCA_matrix as a zero matrix
    MRCA_matrix = np.zeros((num_cells, num_cells))

    converged = False
    for iteration in range(max_iter):
        logger.info("Fate-map optimization iteration %s", iteration + 1)

        # Step 1: Fix alpha_vector, optimize MRCA_matrix
        MRCA_matrix = _compute_mrca_times(t_S, sc_mat, mut_prob, alpha_vector, beta)

        # Step 2: Fix MRCA_matrix, optimize alpha_vector
        new_alpha_vector = _compute_alpha(t_S, sc_mat, mut_prob, beta, MRCA_matrix)

        # Check convergence
        diff = np.linalg.norm(new_alpha_vector - alpha_vector)
        logger.info("Alpha-vector difference: %s", diff)
        if diff < tol:
            logger.info("Fate-map optimization converged")
            alpha_vector = new_alpha_vector
            converged = True
            break

        alpha_vector = new_alpha_vector

    if not converged:
        logger.warning(
            "Fate-map alternating optimization did not converge after %s iterations; "
            "recomputing MRCA from the final alpha vector",
            max_iter,
        )
        MRCA_matrix = _compute_mrca_times(t_S, sc_mat, mut_prob, alpha_vector, beta)

    return alpha_vector, MRCA_matrix


class _UPGMANode:
    """
    A Node class used for building the UPGMA tree.
    Attributes:
        name (str): Name of the node (leaf name or internal node name like "Node1").
        children (list): List of child nodes (empty if leaf).
        height (float): The distance from this node down to the leaves in the UPGMA context.
    """

    def __init__(self, name, children=None, height=0.0):
        self.name = name
        self.children = children if children else []
        self.height = height

    def is_leaf(self):
        """Check if the node is a leaf."""
        return len(self.children) == 0


def _upgma(distance_df):
    """
    Perform UPGMA on the given distance matrix (pandas DataFrame).
    Rows and columns must represent the same set of leaf labels.

    Args:
        distance_df (pd.DataFrame): Symmetric distance matrix for leaves.

    Returns:
        _UPGMANode: The root node of the constructed UPGMA tree.
    """
    if not isinstance(distance_df, pd.DataFrame):
        raise TypeError("distance_df must be a pandas DataFrame")
    if distance_df.shape[0] < 2 or distance_df.shape[0] != distance_df.shape[1]:
        raise ValueError("distance_df must be a square matrix with at least two leaves")
    if not distance_df.index.is_unique or not distance_df.columns.is_unique:
        raise ValueError("distance_df labels must be unique")
    if not distance_df.index.equals(distance_df.columns):
        raise ValueError(
            "distance_df rows and columns must have identical ordered labels"
        )
    try:
        values = distance_df.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("distance_df must contain only numeric values") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("distance_df contains non-finite values")
    if np.any(values < 0):
        raise ValueError("distance_df cannot contain negative distances")
    if not np.allclose(values, values.T, rtol=1e-10, atol=1e-12):
        raise ValueError("distance_df must be symmetric")
    if not np.allclose(np.diag(values), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("distance_df diagonal must be zero")
    distance_df = pd.DataFrame(
        (values + values.T) / 2.0,
        index=distance_df.index.copy(),
        columns=distance_df.columns.copy(),
    )

    # Each initial cluster is a single leaf node
    clusters = {
        name: (_UPGMANode(name=name), 1)  # (node_object, cluster_size)
        for name in distance_df.index
    }

    # Copy the distance matrix to update dynamically
    cur_df = distance_df.copy()

    internal_node_count = 0

    # While more than one cluster remains, merge the closest pair
    while len(clusters) > 1:
        # 1) Find the two clusters with the smallest distance
        min_dist = float("inf")
        pair_to_merge = (None, None)

        names_list = list(clusters.keys())
        for i in range(len(names_list)):
            for j in range(i + 1, len(names_list)):
                n1, n2 = names_list[i], names_list[j]
                d = cur_df.loc[n1, n2]
                if d < min_dist:
                    min_dist = d
                    pair_to_merge = (n1, n2)

        # 2) Merge these two clusters
        n1, n2 = pair_to_merge
        (node1, size1) = clusters[n1]
        (node2, size2) = clusters[n2]

        internal_node_count += 1
        new_node_name = f"IP{internal_node_count}"

        # In UPGMA, the new internal node's height is half the distance
        new_height = min_dist / 2.0
        new_node = _UPGMANode(
            name=new_node_name, children=[node1, node2], height=new_height
        )

        # Combined cluster size
        new_size = size1 + size2

        # Remove merged clusters, add new cluster
        del clusters[n1]
        del clusters[n2]
        clusters[new_node_name] = (new_node, new_size)

        # 3) Update the distance matrix for the newly formed cluster
        for other_name in clusters.keys():
            if other_name == new_node_name:
                continue
            (other_node, other_size) = clusters[other_name]

            # UPGMA average distance formula
            dist_new_other = (
                size1 * cur_df.loc[n1, other_name] + size2 * cur_df.loc[n2, other_name]
            ) / new_size

            cur_df.loc[new_node_name, other_name] = dist_new_other
            cur_df.loc[other_name, new_node_name] = dist_new_other

        # Remove rows and columns of the merged clusters
        cur_df = cur_df.drop(index=[n1, n2], columns=[n1, n2])

    # Only one cluster (the root) remains
    root_name = list(clusters.keys())[0]
    root_node, _ = clusters[root_name]
    return root_node


def _build_normalized_upgma_edges(root):
    """
    Given a parsed tree (root), where each node's 'height' is
    the distance from this node down to the leaves, compute the
    normalized edge length for every Parent->Child.

    For each Parent->Child:
      edge_length = (parent.height - child.height) / root.height

    Args:
        root (ParsedNode): Root of the parsed Newick tree.

    Returns:
        dict: Key: "ParentName->ChildName", Value: normalized edge length (float).
    """
    edge_dict = {}
    total_height = root.height  # The distance from root to leaves

    def dfs(parent, node):
        if parent is not None:
            raw_length = parent.height - node.height
            # Normalize by the total height at the root
            if total_height > 0:
                norm_length = raw_length / total_height
            else:
                norm_length = 0.0

            key = f"{parent.name}->{node.name}"
            edge_dict[key] = norm_length

        for ch in node.children:
            dfs(node, ch)

    dfs(None, root)
    return edge_dict


def mrca_time_to_distance(MRCA_matrix, t_S):
    """Convert MRCA times to leaf-to-leaf patristic distances.

    For contemporaneously sampled leaves at time ``t_S``, the path from leaf
    ``i`` to leaf ``j`` contains two branches of length
    ``t_S - t_MRCA(i, j)``.  Hence ``d(i, j) = 2 * (t_S - t_MRCA(i, j))``.
    """
    mrca = np.asarray(MRCA_matrix, dtype=float)
    if not np.isfinite(t_S) or float(t_S) <= 0:
        raise ValueError("t_S must be finite and positive")
    if mrca.ndim != 2 or mrca.shape[0] != mrca.shape[1]:
        raise ValueError("MRCA_matrix must be a square matrix")
    if not np.all(np.isfinite(mrca)):
        raise ValueError("MRCA_matrix contains non-finite values")
    if np.any(mrca < 0) or np.any(mrca > float(t_S)):
        raise ValueError("MRCA times must lie between 0 and t_S")
    if not np.allclose(mrca, mrca.T, rtol=1e-10, atol=1e-12):
        raise ValueError("MRCA_matrix must be symmetric")
    if not np.allclose(np.diag(mrca), float(t_S), rtol=1e-10, atol=1e-12):
        raise ValueError("MRCA_matrix diagonal must equal t_S")

    distance = 2.0 * (float(t_S) - mrca)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    return distance


def cell_types_distance(MRCA_matrix, cell_types, t_S):
    """
    Construct a cell-type distance matrix from an MRCA (Most Recent Common Ancestor) matrix.

    This function:
      1. Converts MRCA times to pairwise leaf distances as ``2*(t_S-MRCA)``.
      2. Averages distances across all cell pairs belonging to each pair of
         cell types.

    Parameters
    ----------
    MRCA_matrix : np.ndarray
        A square matrix representing pairwise MRCA times among cells.
        Its shape is [num_cells x num_cells].
    cell_types : dict
        A dictionary mapping leaf (cell) labels to their corresponding cell types.
        For example: {"Cell0": "TypeA", "Cell1": "TypeB", ...}.

    Returns
    -------
    distance_df : pd.DataFrame
        A symmetric DataFrame where both rows and columns represent unique cell types.
        Each entry is the average separation distance (or time) between two cell types.
    """
    cell_distance = mrca_time_to_distance(MRCA_matrix, t_S)
    expected_labels = [f"Cell{i}" for i in range(cell_distance.shape[0])]
    missing = [label for label in expected_labels if label not in cell_types]
    if missing:
        raise ValueError(f"cell_types is missing cell labels: {missing}")
    labels = [cell_types[label] for label in expected_labels]
    if any(pd.isna(label) for label in labels):
        raise ValueError("cell_types contains missing type labels")
    types = sorted(set(labels))
    type_distance = np.zeros((len(types), len(types)), dtype=float)

    for i, type_i in enumerate(types):
        indices_i = np.flatnonzero(np.asarray(labels) == type_i)
        for j in range(i + 1, len(types)):
            indices_j = np.flatnonzero(np.asarray(labels) == types[j])
            value = float(cell_distance[np.ix_(indices_i, indices_j)].mean())
            type_distance[i, j] = value
            type_distance[j, i] = value

    return pd.DataFrame(type_distance, index=types, columns=types)


def upgma_to_edge_dict(distance_df):
    """
    Builds a UPGMA tree from the input distance matrix, then generates a dictionary
    of normalized edge lengths in the format { "Parent->Child": normalized_length }.

    Steps:
      1. Perform UPGMA on distance_df to get a tree.
      2. Convert the resulting tree to a Newick string.
      3. Parse the Newick string to interpret node heights as distance-to-leaves.
      4. Build a dictionary of normalized edge lengths, where each edge length is
         (parent.height - child.height) / root.height.

    Parameters
    ----------
    distance_df : pd.DataFrame
        A symmetric distance matrix where rows and columns represent entities
        to be clustered.

    Returns
    -------
    dict
        A dictionary where each key is "ParentName->ChildName" and
        each value is the normalized edge length (float) between them.
    """
    return _build_normalized_upgma_edges(_upgma(distance_df))


def load_fate_map_topology(fate_map_path):
    """
    Load a dictionary from a CSV file.

    Parameters:
        fate_map_path (str): Path to the input CSV file.

    Returns:
        dict: The reconstructed dictionary.
    """
    df = pd.read_csv(fate_map_path)
    required = {"Parent", "Child", "Length"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Fate-map CSV is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Fate-map CSV contains no edges")
    if df[["Parent", "Child"]].isna().any().any():
        raise ValueError("Fate-map CSV contains missing node identifiers")
    duplicated = df.duplicated(subset=["Parent", "Child"], keep=False)
    if duplicated.any():
        pairs = df.loc[duplicated, ["Parent", "Child"]].drop_duplicates()
        raise ValueError(
            f"Fate-map CSV contains duplicate edges: {pairs.values.tolist()}"
        )
    try:
        lengths = pd.to_numeric(df["Length"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Fate-map CSV contains non-numeric edge lengths") from exc
    if not np.all(np.isfinite(lengths)) or np.any(lengths < 0):
        raise ValueError("Fate-map CSV edge lengths must be finite and non-negative")
    edge_dict = {
        f"{row['Parent']}->{row['Child']}": float(length)
        for (_, row), length in zip(df.iterrows(), lengths)
    }
    return edge_dict


def parse_edge_dict(edge_dict: Dict[str, float]):
    """
    Convert an edge dictionary (key = "Parent->Child", value = weight)
    into a list of FateMapEdge objects.

    Parameters
    ----------
    edge_dict : Dict[str, float]
        A dictionary where each key has the format "ParentName->ChildName"
        and the value is the edge weight or distance.

    Returns
    -------
    list
        A list of FateMapEdge objects constructed from the key-value pairs.
    """
    edges = []
    for key, weight in edge_dict.items():
        if not isinstance(key, str) or key.count("->") != 1:
            raise ValueError(f"Invalid edge key: {key!r}")
        start, end = key.split("->", 1)
        grn_edge = FateMapEdge(start, end, weight)
        edges.append(grn_edge)
    return edges


def construct_fate_map(
    sc_mat, cell_types, t_S, beta=1, max_iter=100, tol=1e-2, output_edges=False
):
    """
    Construct a fate map based on single-cell matrix ``sc_mat``, cell types,
    and inferred alpha/MRCA matrices. This is a high-level pipeline.

    **Steps:**

    1. **Compute default allele probabilities.**
    2. **Perform alternating optimization to get final alpha and MRCA_matrix.**
    3. **Build a cell types distance DataFrame using ``cell_types_distance``.**
    4. **Convert that distance DataFrame into a dictionary of normalized UPGMA edges.**
    5. **Parse the edge dictionary into ``GRNEdge`` objects.**
    6. **Wrap edges into a ``FateMap`` object.**

    :param np.ndarray sc_mat:
        Single-cell matrix, where rows typically represent cells
        and columns represent sites.
    :param dict cell_types:
        Mapping from cell labels to their cell type names.
    :param float t_S:
        A sampling time or similar parameter for the optimization.
    :param float beta:
        A parameter for the (1 + beta * t) model in the integrals.
        **Default:** 1.
    :param int max_iter:
        Maximum number of iterations for alternating optimization.
        **Default:** 1000.
    :param float tol:
        Convergence threshold.
        **Default:** 1e-5.
    :param bool output_edges:
        If True, returns a tuple of (edge, fate_map).
        Otherwise returns just the fate_map.

    :returns:
        - If ``output_edges`` is False:

          A ``FateMap`` object containing the inferred edges
          for further analysis/visualization.

        - If ``output_edges`` is True:

          A tuple ``(edge, fate_map)`` where ``edge`` represents
          the GRNEdge objects.
    :rtype: FateMap or tuple
    """

    # 1. Compute default allele probabilities
    mut_prob = _compute_default_allele_prob(sc_mat)

    # 2. Perform alternating optimization to get alpha and MRCA
    _, MRCA_matrix = _alternating_optimization(
        t_S, sc_mat, mut_prob, beta, max_iter, tol
    )

    # 3. Build a cell-type distance DataFrame
    distance_df = cell_types_distance(MRCA_matrix, cell_types, t_S)

    # 4. Convert distance DataFrame to edge dict via UPGMA
    edge_dict = upgma_to_edge_dict(distance_df)

    # 5. Parse edge dict to create GRNEdge objects
    edge = parse_edge_dict(edge_dict)

    # 6. Construct the FateMap (wrap the edges)
    fate_map = FateMap(edge)

    if output_edges:
        return edge, fate_map
    else:
        return fate_map
