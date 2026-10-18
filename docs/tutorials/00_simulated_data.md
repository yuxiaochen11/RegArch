# Example of Simulated Data

In this tutorial, we demonstrate the basic usage of **RegArch** using simulated data. This example highlights how RegArch combines scATAC-seq and scRNA-seq data with cell lineage information to infer dynamic gene regulatory networks.

## Model-misspecification sensitivity

The benchmark tests whether ancestral reconstruction remains stable when the
inference model is deliberately misspecified. The primary analysis evaluates
`P(S_ancestors | D_leaves, T, Q, pi_root)` with fixed underlying simulated leaf
GRN weights, so parameter sensitivity is not confounded by a new RNA-regression
fit. Only the temperature conditions recalibrate the evidence probabilities
derived from those weights. The one-factor-at-a-time grid changes CTMC
gain/loss/sign-switch rates,
positive-state/gain odds, edge-specific root-prior odds, branch lengths,
topology, and terminal-evidence temperatures. Rates use `0.25x`, `0.5x`, `2x`,
and `4x`; the remaining continuous factors use a compact `0.5x`/`2x` grid.
Probability odds are scaled on the logit scale and are never clipped.

```shell
python -m benchmark.cli generate \
  --config benchmark/configs/lineage_signal_10leaf_10rep.json \
  --output benchmark/runs/lineage_signal_10leaf

python -m benchmark.run_fixed_evidence_sensitivity \
  --runs-dir benchmark/runs/lineage_signal_10leaf \
  --output-dir benchmark/results/lineage_signal_10leaf_model_misspecification

python -m benchmark.plot_model_misspecification \
  --results-dir benchmark/results/lineage_signal_10leaf_model_misspecification \
  --output-dir benchmark/results/journal_figures \
  --prefix 10leaf_model_misspecification
```

Results are reported separately for root, internal, and leaf nodes. The primary
figure shows replicate-paired change in AUPRC with 95% bootstrap confidence
intervals; a local-scale companion resolves non-topology effects, and signed
precision at true K is supplementary. The topology condition changes topology
only. Root-prior perturbations preserve the reference ATAC prior source and
ranking; the reference uses the generating `Q` and branch lengths with the
unperturbed ATAC-derived root prior, rather than an oracle root state. An
optional end-to-end refit is available through
`benchmark.run_model_misspecification` and additionally reports optimizer
convergence.

<div style="text-align: center;">
  <img src="../_static/10leaf_model_misspecification_delta_auprc.png"
       alt="Ten-replicate paired model-misspecification sensitivity in RegArch"
       style="width: 100%;">
</div>

The same AUPRC results excluding the topology panel are shown below on one
shared enlarged y-scale; this is a view of the same prespecified analysis, not
an additional hypothesis test.

<div style="text-align: center;">
  <img src="../_static/10leaf_model_misspecification_local_delta_auprc.png"
       alt="Local-scale detail of non-topology RegArch sensitivity conditions"
       style="width: 100%;">
</div>

<!-- Workflow Visualization -->
<div style="text-align: center;">
  <img src="../_static/simulation_workflow.png" alt="Simulation Workflow" style="width: 300%;">
</div>

---

## Data Generation

### 1. Time-scaled Cell Fate Map

In this example, we use the topology structure of a time-scaled fate map (represented by `fate_map_edge_dict` with 7 nodes/cell clusters) as input. We also set a state transition rate (`lambda`) to measure the probability that a child GRN inherits the known regulatory relationships of its parent GRN.

#### Load and Process Fate Map

```python
# Load the fate map topology from CSV and process it into a FateMap object.
fate_map_path = 'examples/data/simulation/additional_input_data/grn_dict.csv'
edge_dict = load_fate_map_topology(fate_map_path)
edge = parse_edge_dict(edge_dict)
fate_map_topology = FateMap(edge)
newick_format = generate_newick(fate_map_topology)
```

**Sample Newick Format Output:**

```text
'((T3:0.6,T4:0.6)P1:0.4,(T5:0.9,T6:0.9)P2:0.1)P0:0;'
```

#### Fate Map Visualization (Rscript)

```R
# Use R to generate a fate map plot.
Rscript plot_fate_map.R "((T3:0.6,T4:0.6)P1:0.4,(T5:0.9,T6:0.9)P2:0.1)P0:0;" 1.2 5 0.3 fate_map_plot.pdf
```
---

### 2. Ground Truth GRNs

#### Generate the Root GRN

- **Parameters**: 150 target genes, 100 regulators  
- **Output**: A pandas DataFrame `grn0` where:
  - **Rows**: represent regulators  
  - **Columns**: represent target genes  
  - **Elements**: range from -1 to 1, with the absolute value indicating regulatory strength, and the sign indicating activation (+) or repression (-)

```python
# Define output path for the ground truth GRN.
grns_path = 'examples/results/simulation/additional_output/ground_truth_grn'

# Generate the root GRN.
grn0 = generate_root_grn(150, 100, grns_path)
```

**Sample DataFrame Output:**

```text
      0        1         2         3   ...   148       149
150  -0.7318   0.5520    0.0000    0.0000 ...   0.2533    0.0000
151   0.0000   0.3824    0.4575    0.4303 ...  -0.8496    0.0000
...
```

#### Generate Descendant GRNs

Load the dynamic GRN topology and generate descendant GRNs for each cell cluster based on the fate map.

```python
# Load additional GRN topology and generate descendant GRNs.
grn_dict_path = 'examples/data/simulation/additional_input_data/grn_dict.csv'
grn_dict = load_fate_map_topology(grn_dict_path)
descendant_grns = generate_descendant_grns(7, 150, 100, grn_dict, grns_path, grn0)
```

---

### 3. scRNA-seq Data 

Generate expression data required by SERGIO. For each node, the output DataFrame has:
- **Rows**: regulators and target genes  
- **Columns**: sampled cells  
- **Elements**: gene expression values

```python
# Generate scRNA-seq expression data.
sergio_files_path = 'examples/results/simulation/additional_output/sergio_files'
expr_matrix_path = 'examples/results/simulation/additional_output/expression_files'
generate_expression_data(descendant_grns, 7, 150, 100, 300, sergio_files_path, expr_matrix_path, 1)
```

**Console Output (Sample):**

```text
Start simulating new level
There are 100 genes to simulate in this layer
Done with current level
...
Start simulating new level
There are 150 genes to simulate in this layer
Done with current level
```

Convert the generated expression data into the RegArch input format. The final DataFrame includes five columns:  
`gene_id`, `cell_id`, `gene expression_value`, `node_id`, `gene_type` (target_gene or regulator_gene)

```python
# Convert expression files to RegArch input format.
expression_file_path = 'examples/data/simulation/input/expression_data.csv'

# Load gene names.
target_genes_name = pd.read_csv('examples/data/simulation/additional_input_data/target_genes_name.csv')['x'].tolist()
regulatory_genes_name = pd.read_csv('examples/data/simulation/additional_input_data/regulatory_genes_name.csv')['x'].tolist()

prepare_expression_data(150, target_genes_name, regulatory_genes_name, expr_matrix_path, expression_file_path)
```

**Sample Converted Data (First Few Rows):**

```text
gene         cell      value   node_id    gene_type
target_0     cell_0    10.0    grn3       target_gene
target_0     cell_1    18.0    grn3       target_gene
target_0     cell_2    15.0    grn3       target_gene
...
```

---

### 4. scATAC-seq Data 

Generate scATAC-seq data as a pandas DataFrame with the following columns:  
`target_gene_id`, `regulator_id`, `co-accessibility score`, `node_id`

```python
# Generate scATAC-seq data.
atac_file_path = 'examples/data/simulation/input/atac_data.csv'
atac_data = generate_atac_data(descendant_grns, 7, 150, 100, atac_file_path)
```

---

## Inference of Gene Regulatory Networks

Use the generated scATAC-seq and scRNA-seq data to infer GRNs.

```python
# Setup GRN inference.
saved_dir = 'examples/results/simulation/inferred_grns/inferred_grn_defult_params'
fate_map = FateMap(parse_edge_dict(grn_dict))

# Run GRN inference.
grn_inference_result = GRNInference(atac_file_path, expression_file_path, fate_map, saved_dir)
grn_inference_result.infer_grn(20)
```

**Sample Log Output:**

```text
-2024-09-24 11:32:58,551-INFO-52272 Start fitting target_gene_id:1
-2024-09-24 11:34:56,762-INFO-25168 Finish inferencing leaves grn value for target_gene_id:1
-2024-09-24 11:34:56,763-INFO-25168 Saved grn values for target_gene_id:1
...
```

Retrieve target gene networks by setting a threshold (e.g., 0.1) to filter effective regulatory relationships.

```python
# Get target networks with a threshold of 0.1.
target_networks_dict = grn_inference.get_target_networks(0.1)
```

For an alternative output format, generate a dictionary where:
- **Key**: `node_id`  
- **Value**: A DataFrame with rows for target genes and columns for regulators

```python
# Generate dynamic networks for each fate map node.
dynamic_networks_dict = get_dynamic_networks(saved_dir, grns, 0.1, regulatory_genes_name, target_genes_name)
```

### Visualizing the Fate Map Path

You can visualize the inferred GRN along a specific path using the `.get_path` method.

```python
# Example: Get the path for node 'grn5'.
path = fate_map.get_path('grn5')
print(path)  # Expected output: ['grn0', 'grn2', 'grn5']
```

Extract and format the GRN along the selected path into a three-column DataFrame using `get_positive_interactions`.

```python
# Extract networks for grn0, grn2, and grn5.
network0 = dynamic_networks_dict['grn0']
network2 = dynamic_networks_dict['grn2']
network5 = dynamic_networks_dict['grn5']

# Assign gene names to columns and index.
network0.columns = network2.columns = network5.columns = regulatory_genes_name
network0.index = network2.index = network5.index = target_genes_name

# Get gene interactions.
gene0_interaction = get_positive_interactions(network0)
gene2_interaction = get_positive_interactions(network2)
gene5_interaction = get_positive_interactions(network5)

# Save interaction results.
gene0_interaction.to_csv('examples/results/simulation/inferred_grns/interaction_format/gene0_interaction.csv', header=None)
gene2_interaction.to_csv('examples/results/simulation/inferred_grns/interaction_format/gene2_interaction.csv', header=None)
gene5_interaction.to_csv('examples/results/simulation/inferred_grns/interaction_format/gene5_interaction.csv', header=None)
```

Visualize GRNs on the selected pathways using R scripts:

```R
# Plot GRN for grn0.
Rscript plot_grn.R examples/results/simulation/additional_output/expression_files/Exp_grn5.csv \
examples/data/simulation/additional_input_data/target_genes_name.csv \
examples/data/simulation/additional_input_data/regulatory_genes_name.csv \
examples/results/simulation/inferred_grns/interaction_format/gene0_interaction.csv grn0_plot.pdf

# Plot GRN for grn2.
Rscript plot_grn.R examples/results/simulation/additional_output/expression_files/Exp_grn5.csv \
examples/data/simulation/additional_input_data/target_genes_name.csv \
examples/data/simulation/additional_input_data/regulatory_genes_name.csv \
examples/results/simulation/inferred_grns/interaction_format/gene2_interaction.csv grn2_plot.pdf

# Plot GRN for grn5.
Rscript plot_grn.R examples/results/simulation/additional_output/expression_files/Exp_grn5.csv \
examples/data/simulation/additional_input_data/target_genes_name.csv \
examples/data/simulation/additional_input_data/regulatory_genes_name.csv \
examples/results/simulation/inferred_grns/interaction_format/gene5_interaction.csv grn5_plot.pdf
```

<div style="text-align: center;">
  <img src="../_static/grns.png" alt="GRNs Visualization" style="width: 80%;">
</div>

---

## Validation of RegArch Performance

Traverse all fate map nodes to compute AUROC and AUPRC by comparing inferred GRNs with the ground truth.

```python
# Evaluate performance for each fate map node.
nodes = list(fate_map.nodes.keys())
auroc_list = []
auprc_list = []

for node_id in nodes:
    grn_infer = dynamic_networks_dict[node_id]
    grn_true = pd.read_csv('examples/results/simulation/additional_output/ground_truth_grn/'+node_id+'.csv')
    grn_true = grn_true.T
    auroc = compute_auroc(grn_infer, grn_true, 0.1)
    auprc = compute_auprc(grn_infer, grn_true, 0.1)
    print([node_id, auroc, auprc])
    auroc_list.append([node_id, auroc])
    auprc_list.append([node_id, auprc])
    
auroc_df = pd.DataFrame(auroc_list)
auroc_df.columns = ['node_id','value']
auprc_df = pd.DataFrame(auprc_list)
auprc_df.columns = ['node_id','value']
auprc_df.to_csv('examples/results/simulation/evaluation_result/auprc.csv', index=None)
auroc_df.to_csv('examples/results/simulation/evaluation_result/auroc.csv', index=None)
```

**Sample Output:**

```text
['grn0', 0.6080135899404416, 0.6076961007516475]
['grn1', 0.6778555297246805, 0.6693773066892854]
['grn2', 0.6118844499627417, 0.5970852889617932]
['grn3', 0.820360973156812, 0.8165970672496445]
['grn4', 0.8171958121109225, 0.8123780232426353]
['grn5', 0.8240320911215709, 0.8254550385925934]
['grn6', 0.8197851536821597, 0.8222811570776326]
```

## RegArch performance verification under perturbed fate map

Infer GRNs based on the perturbed fate map.

```python
## Perturb the length of the fate map, the strengths are 0.05, 0.1, 0.3, 0.5, 0.7 respectively
saved_dir_pert005 = 'examples/results/simulation/inferred_grns/inferred_grn_pert005'
fate_map_pert005 = perturb_branch_lengths(newick_format, sigma=0.05, seed=1234)

grn_inference_result_pert005 = GRNInference(atac_file_path, expression_file_path, fate_map_pert005, saved_dir_pert005)
grn_inference_result_pert005.infer_grn(20)
grn_dict_pert005 = get_dynamic_networks(saved_dir_pert005, fate_map_pert005, 0.1, regulatory_genes_name, target_genes_name)

#...

## Perturb the topology of the fate map, but keep the triplets unchanged
newick_format_pert_topology = '(((grn3:0.6,grn4:0.6)grn1:0.4,grn5:0.9)grn2:0.1,grn6:1.0)grn0:0;'
saved_dir_pert_topology = 'examples/results/simulation/inferred_grns/inferred_grn_pert_topology'

edge_dict_pert_topology = newick_to_edge_length_dict(newick_format_pert_topology)
edge_pert_topology = parse_edge_dict(edge_dict_pert_topology)
fate_map_pert_topology = FateMap(edge_pert_topology)

grn_inference_result_pert_str = GRNInference(atac_file_path, expression_file_path, fate_map_pert_topology, saved_dir_pert_topology)
grn_inference_result_pert_str.infer_grn(20)
grn_dict_pert_topology = get_dynamic_networks(saved_dir_pert_topology, fate_map_pert_topology, 0.1, regulatory_genes_name, target_genes_name)
```
The **Jaccard index** and **key regulators overlap** are used to compare the differences in GRN inference based on fate maps before and after perturbation.

```python
jaccard_means = []
overlap_means = []
grn_pert = [grn_dict_pert005, grn_dict_pert01, grn_dict_pert03, grn_dict_pert05, grn_dict_pert07, grn_dict_pert_topology]
for id in range(len(grn_pert)):
    jaccard_means.append(metric_perturbation_impact(grn_pert[id], dynamic_networks_dict)[0])
    overlap_means.append(metric_perturbation_impact(grn_pert[id], dynamic_networks_dict)[1])

print(jaccard_means)
print(overlap_means)
```

**Sample Output:**

```text
[0.9928021771748436, 0.9306423926821185, 0.9301414871605946, 0.9299776335481349, 0.9281831137652954, 0.8616158941999136]
[0.9714285714285714, 0.880952380952381, 0.880952380952381, 0.8857142857142858, 0.8857142857142858, 0.8238095238095238]
```

**Visualization:**

```python
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches

perturbation_labels = ["0.05", "0.1", "0.3", "0.5", "0.7", "Topology"]
colors = ['#EFEFEF', '#DCDDDD', '#C9CACA', '#B5B5B6', '#9FA0A0', '#A3CA89']
metric_labels = ["Jaccard index", "Key regulator overlap"]

x = np.arange(len(metric_labels)) 
bar_width = 0.1
fig, ax = plt.subplots(figsize=(3.2, 1.6))

for i, (jac, over, color, label) in enumerate(zip(jaccard_means, overlap_means, colors, perturbation_labels)):
    ax.bar(x[0] + i * bar_width - 0.25, jac, width=bar_width, color=color)
    ax.bar(x[1] + i * bar_width - 0.25, over, width=bar_width, color=color)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=6, fontname='Arial')

ax.set_ylabel("GRN similarity", fontsize=6, fontname='Arial')
ax.set_ylim(0.5, 1.05)
ax.tick_params(axis='y', labelsize=5)
for label in ax.get_yticklabels():
    label.set_fontname('Arial')

legend_handles = [mpatches.Patch(color=col, label=lab) for col, lab in zip(colors, perturbation_labels)]
legend_font = FontProperties(family='Arial', size=5.5)
ax.legend(
    handles=legend_handles,
    loc='center left',
    bbox_to_anchor=(1.0, 0.7),
    prop=legend_font,
    markerscale=0.8,
    handlelength=1.0,
    handleheight=0.8,
    labelspacing=0.3,
    borderpad=0.3,
    borderaxespad=0.4
)

ax.set_axisbelow(True)
ax.grid(True, axis='y', linestyle='--', alpha=0.7, linewidth=0.5)
for spine in ax.spines.values():
    spine.set_color('#666666')
ax.tick_params(axis='x', color='#666666')
ax.tick_params(axis='y', color='#666666')

output_path = 'examples/results/simulation/additional_output/figures'
filename = "pert_grouped.eps"
os.makedirs(output_path, exist_ok=True)
save_path = os.path.join(output_path, filename)
plt.tight_layout()
plt.savefig(save_path, format='eps', bbox_inches='tight')
plt.show()
```
<div style="text-align: center;">
  <img src="../_static/pert_result.svg" alt="Perturbation Result Visualization" style="width: 50%;">
</div>
