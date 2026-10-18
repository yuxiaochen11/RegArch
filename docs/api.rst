API reference
=============

Core inference
--------------

.. autoclass:: regarch.cell_fate_map.FateMap
   :members:

.. autoclass:: regarch.cell_fate_map.FateMapEdge
   :members:

.. autofunction:: regarch.cell_fate_map.construct_fate_map

.. autofunction:: regarch.cell_fate_map.generate_newick

.. autofunction:: regarch.cell_fate_map.load_fate_map_topology

.. autofunction:: regarch.cell_fate_map.parse_edge_dict

.. autoclass:: regarch.gene_regulatory_network.GRNInference
   :members: infer_leaf_grn, infer_ancestor_grn, infer_grn, get_target_networks

Network analysis
----------------

.. autofunction:: regarch.downstream_analysis.get_regulators_for_target_gene

.. autofunction:: regarch.downstream_analysis.get_targets_for_regulator_gene

.. autofunction:: regarch.downstream_analysis.identify_differentiation_regulators

.. autofunction:: regarch.downstream_analysis.identify_fate_bias_regulators

.. autofunction:: regarch.downstream_analysis.cluster_regulatory_interactions

.. autofunction:: regarch.downstream_analysis.summarize_interaction_cluster_specificity

Plotting
--------

.. autofunction:: regarch.plotting.plot_regulatory_interactions_along_fate_map

.. autofunction:: regarch.plotting.plot_regulatory_genes_along_fate_map

.. autofunction:: regarch.plotting.plot_dynamic_regulator_count

.. autofunction:: regarch.plotting.plot_target_genes_along_fate_map

.. autofunction:: regarch.plotting.plot_regulatory_network_along_fate_map

.. autofunction:: regarch.plotting.plot_regulator_activity_across_lineages

.. autofunction:: regarch.plotting.plot_regulatory_strength_along_fate_map

.. autofunction:: regarch.plotting.plot_key_genes_differentiation

.. autofunction:: regarch.plotting.plot_key_genes_fate_bias

.. autofunction:: regarch.plotting.plot_regulatory_interaction_clusters

.. autofunction:: regarch.plotting.plot_regulatory_interactions_by_cell_type
