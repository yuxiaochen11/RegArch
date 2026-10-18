# RegArch
***
RegArch infers lineage-resolved gene regulatory networks (GRNs) by integrating lineage tracing, scRNA-seq, and scATAC-seq data on a time-scaled cell fate map. Its Python package is `regarch`.

RegArch supports two GRN inference modes: the signed-CTMC mode provides the formal probabilistic model used for controlled benchmarking, whereas `alt` is a transition-agnostic lineage-regularized mode for empirical analyses when regulatory transition rates are not independently calibrated.
***

### Features
The currently available features include:
- Inferring time-scaled cell fate maps
- Inferring lineage-resolved gene regulatory networks
- Reconstructing regulatory states across terminal and ancestral lineage nodes
- Characterizing inferred regulatory reorganization along cell lineages
- Prioritizing differentiation-associated regulators
- Prioritizing branch-biased regulators
- Identifying constitutive and lineage-associated regulatory modules

### Environment
Python 3.11 or later.

#### Installation
```shell
python -m pip install -e ".[demo]"
```

If an offline or incomplete package mirror cannot create the isolated build
environment, use `python -m pip install -e ".[demo]" --no-build-isolation`.

Run commands and notebooks from the project root. The demo notebooks also
contain a small bootstrap cell, so they remain runnable when Jupyter opens
them from `examples/demos/`.

#### Repository layout

```text
regarch/                 Installable Python package
  downstream_analysis/  Regulator and interaction-cluster analyses
  plotting/              Analysis plotting API
  simulation/            Synthetic-data generation and internal simulator
  utils/                 Shared I/O, metrics, constants, and logging
benchmark/               Reproducible simulation and comparison pipeline
docs/                    Sphinx documentation source
examples/data/           Bundled inputs
examples/demos/          Executable notebooks
examples/results/        Reference outputs used by notebook analysis cells
tests/                   Package and benchmark regression tests
```

#### Reproduce the bundled examples

```shell
jupyter lab
```

Open any notebook in `examples/demos/` and choose **Run All**. Input data and the
reference outputs used by the analysis cells are bundled under
`examples/data/` and `examples/results/`. Some inference cells are
computationally intensive and are disabled by default; the stored outputs let
all downstream analysis and plotting cells run immediately. Set the notebook's
`RECOMPUTE_*` flags to `True` only when intentionally regenerating results.

***
### Usage overview

#### Inferring the time-scaled cell fate map
You have two options for specifying the cell fate map. First, lineage tracing data can be supplied as a NumPy array:

```python
from regarch import construct_fate_map
import numpy as np

# Input the lineage tracing data
elements = np.array([0, 1, 2, 3, 4, 5])
sc_mat = np.random.choice(elements, size=(30, 15)) 
t_S=15

# Input the cell type of each sampled cell if each fate-map terminal node
# should correspond to a cell cluster with the same cell type.
cell_types = {
            "Cell0":"C","Cell1":"A","Cell2":"A","Cell3":"B","Cell4":"C",
            "Cell5":"A","Cell6":"B","Cell7":"D","Cell8":"D","Cell9":"B",
            "Cell10":"B","Cell11":"C","Cell12":"C","Cell13":"A","Cell14":"A",
            "Cell15":"A","Cell16":"D","Cell17":"D","Cell18":"D","Cell19":"B",
            "Cell20":"C","Cell21":"A","Cell22":"C","Cell23":"C","Cell24":"B",
            "Cell25":"B","Cell26":"A","Cell27":"A","Cell28":"D","Cell29":"B"
            }

# Reconstruct the time-scaled cell fate map
fate_map = construct_fate_map(sc_mat, cell_types, t_S, beta=1, max_iter=20, tol=1e-2)
```

You can also directly provide the topology of a cell fate map:

```python
from regarch import FateMap, parse_edge_dict

# Input the cell fate map directly
edge_dict = {"grn0->grn1":0.4,"grn0->grn2":0.1,"grn1->grn3":0.6,"grn1->grn4":0.6,"grn2->grn5":0.9,"grn2->grn6":0.9}
parse_edge = parse_edge_dict(edge_dict)
fate_map = FateMap(parse_edge)
```

#### Inferring lineage-resolved gene regulatory networks
Prepare the transcriptomic and chromatin-derived regulatory data in the format of the bundled examples, together with a time-scaled fate map:

```python
from regarch import FateMap, GRNInference, load_fate_map_topology, parse_edge_dict

# Input the transcriptome and chromatin accessibility data
atac_file_path = "examples/data/C.elegans/C.elegans_Core/input_data/EMS/atac_data.csv"
expression_file_path = "examples/data/C.elegans/C.elegans_Core/input_data/EMS/expression_data.csv"
fate_map_path = "examples/data/C.elegans/C.elegans_Core/input_data/EMS/edge_dict_EMS.csv"

# Define an output path
saved_dir = 'examples/results/my_first_run'

edge_dict = load_fate_map_topology(fate_map_path)
fate_map = FateMap(parse_edge_dict(edge_dict))
grn_inference = GRNInference(atac_file_path, expression_file_path, fate_map, saved_dir)
grn_inference.infer_grn()
```

### Validation

Run the complete automated validation suite from the project root:

```shell
python -m pytest -q
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

### More
More information about RegArch can be found at https://regarch.readthedocs.io, including the API reference and usage tutorials.
***

### Citation
Citation information will be added when the associated manuscript is published.
