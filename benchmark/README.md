# RegArch simulation benchmark

This directory contains a reproducible simulation factory and common evaluation
layer for the revised RegArch benchmarks. It is independent of the legacy
functions in `regarch/utils/synthetic_data_generation.py`.

## What is generated

1. A balanced binary fate map with explicit branch lengths.
2. A sparse root GRN with signed continuous weights.
3. Node-specific GRNs propagated with a three-state continuous-time Markov
   process. States are `-1`, `0`, and `1`, and branch transitions are computed
   exactly as `P(t) = exp(Qt)`.
4. A shared enhancer-mediated truth layer:
   `TF -> peak -> target`.
5. Paired leaf-cell RNA and peak-level ATAC observations.
6. A noisy motif layer and noisy peak-gene candidate links.
7. A RegArch ATAC prior derived only from observed accessibility, motif
   scores, and peak-gene links.
8. Shared input manifests for CellOracle, SCENIC, SCENIC+, GENIE3, GRNBoost,
   FigR, Pando, GRaNIE, LINGER, Dictys, and scMTNI.
9. A lineage-signal diagnostic comparing sibling leaves with cross-clade leaves
   and checking observed branch transitions against the generating CTMC.

The primary benchmark compares leaf-node GRNs. Ancestor-node recovery is a
separate RegArch-only analysis.

## Generate data

Generate the ten independent benchmark replicates with:

```powershell
python -m benchmark.cli generate `
  --config benchmark/configs/lineage_signal_10leaf_10rep.json `
  --output benchmark/runs/lineage_signal_10leaf
```

The configuration has 19 total nodes (10 observed leaves), 8 regulators,
20 total genes, 30 cells per leaf, nominal root density 0.08, 60% positive
edges, short terminal branches, and stronger between-clade turnover.
Regulators are included in the target-gene set. Raw run directories are
generated on demand and intentionally not versioned; compact aggregate results
and SHA-256 input manifests are retained under `benchmark/results/`.

## Output layout

Each replicate contains:

```text
truth/
  fate_map.nwk
  fate_map_edges.tsv
  node_metadata.tsv
  grn_matrices/
  true_grn_all_nodes.tsv
  true_grn_leaf_nodes.tsv
  true_tf_region_gene_edges.tsv
  rewiring_transition_log.tsv
intermediate/
  metadata/
  rna/
  atac/
  evaluation/
    edge_universe.tsv
    edge_universe_all_nodes.tsv
    edge_universe_ancestor_nodes.tsv
method_inputs/
  RegArch/
  CellOracle/
  ...
ablations/
  wrong_fate_map/
  atac_prior_coverage_*.tsv
  atac_prior_no_atac.tsv
method_outputs/
  raw/
  standardized/
evaluation/
diagnostics/
  lineage_signal.tsv
  lineage_signal_summary.tsv
```

Method directories contain concrete text inputs where a stable text format is
available. Methods requiring version-specific objects such as Seurat, AnnData,
cisTopic, loom, or cisTarget databases receive an `input_manifest.json` that
points to the shared matrices without duplicating them. Those objects must be
built inside the environment containing the corresponding external method.

## Run RegArch

```powershell
python -m benchmark.run_regarch benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001 `
  --variant full --processes 1

python -m benchmark.run_regarch benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001 `
  --variant no_fate_map --processes 1

python -m benchmark.run_regarch benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001 `
  --variant no_atac --processes 1

python -m benchmark.run_regarch benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001 `
  --variant wrong_fate_map --processes 1
```

The benchmark runner defaults to `--lineage-model ctmc_signed`. It uses the
same three states as the simulator (`-1`, `0`, `+1`) and the same gain, loss,
sign-switch, positive-fraction, and branch-length parameters. Exact tree
sum-product passes couple all observed leaves during optimization and then
produce posterior state probabilities for every leaf and ancestor. At every
node, absolute output score is posterior edge-presence probability and score
sign is the posterior MAP activation/repression state. This common output
definition makes leaf-to-leaf and leaf-to-ancestor evaluation comparable.
Use `ctmc_presence` as an unsigned ablation and `alt` only as a
model-misspecification control.

The default lineage weight is `0.5`, with an edge-specific ATAC-derived root
prior. These settings are frozen in the retained `lineage_signal_10leaf`
configuration and should not be tuned on the reported benchmark outputs.
The objective weights are explicit command-line parameters:
`--lineage-weight`, `--atac-weight`, and `--likelihood-weight`. Select them on
separate calibration replicates and freeze them before evaluating paper test
replicates. Do not choose a weight from the reported test benchmark.

The primary claim should be supported only when `diagnostics/lineage_signal_summary.tsv`
shows stronger sibling similarity than cross-clade similarity and the full
model improves over `no_fate_map` and `wrong_fate_map`. The generator does not
artificially degrade competitor inputs or force RegArch to rank first.

`no_fate_map` disables the lineage prior while retaining the data containers
required by the current implementation. It is valid for the leaf-node
ablation only. Its ancestor outputs must not be interpreted.

ATAC quality variants can be selected with `--prior-file`:

```powershell
python -m benchmark.run_regarch benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001 `
  --variant full `
  --run-name atac_cov050_cor075 `
  --prior-file benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001/ablations/atac_prior_coverage_0.50_correctness_0.75.tsv
```

Optimization sensitivity is controlled with `--optimizer L-BFGS-B`, `SLSQP`,
or `COBYLA`.

## Run the matched ablation study

The complete fate-map and ATAC-prior ablation study reuses the retained
ten-replicate dataset and changes one inference input at a time:

```powershell
$reps = Get-ChildItem benchmark/runs/lineage_signal_10leaf `
  -Directory -Filter "lineage_signal_10leaf_rep_*"

python -m benchmark.run_ablation_study $reps.FullName `
  --output-dir benchmark/results/lineage_signal_10leaf_ablations `
  --processes 4
```

The study includes full, no-fate-map, wrong-fate-map, no-ATAC, uniform-root,
coverage-only, correctness-only, and joint ATAC-quality conditions.
`prior_cov100_cor100` is a truth-controlled oracle prior used as an upper
bound; it is not a realistic method input. Results, parameters, paired tests,
convergence diagnostics, and figures are written under
`results/lineage_signal_10leaf_ablations/`.

## Standardize and evaluate

All methods must produce:

```text
method  cell_type  regulator  target  score  sign_optional
```

Missing edges are assigned score zero over the complete shared edge universe.
The primary score is unsigned AUPRC. AUROC, precision/recall at the true edge
count, signed precision/recall at the true edge count, sign accuracy among
recovered true edges, sign coverage, and top-regulator overlap are also
reported. External methods are evaluated only on observed leaves. Ancestor
recovery is a separate RegArch-only analysis over
`edge_universe_ancestor_nodes.tsv`.

The unified runner writes the comparison-ready leaf output to
`method_outputs/standardized/RegArch.tsv` and retains the same run's
full-node standardized posterior in `evaluation/RegArch_all_nodes.tsv`.
The full-node file is excluded from the 11-method leaf comparison.

```powershell
python -m benchmark.standardize_regarch `
  --inferred-dir benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001/method_outputs/raw/RegArch_full/inferred_grns `
  --regulator-list benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001/method_inputs/RegArch/regulator_list.txt `
  --method-name RegArch_full `
  --output benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001/method_outputs/standardized/RegArch_full.tsv

python -m benchmark.cli evaluate `
  --edge-universe benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001/intermediate/evaluation/edge_universe.tsv `
  --scores benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001/method_outputs/standardized/*.tsv `
  --output benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001/evaluation/benchmark_metrics.tsv
```

For ancestor-node recovery, evaluate the same standardized RegArch output
against `edge_universe_ancestor_nodes.tsv`. For a wrong fate map, standardize
with the MRCA/descendant-set mapping:

```powershell
python -m benchmark.standardize_regarch `
  --inferred-dir <wrong-map-inferred-directory> `
  --regulator-list <regulator-list> `
  --node-mapping <replicate>/ablations/wrong_fate_map/wrong_to_truth_ancestor_mapping.tsv `
  --output <replicate>/method_outputs/standardized/RegArch_wrong_fate_map.tsv
```

For the retained ten-replicate benchmark:

```powershell
python -m benchmark.audit_ancestor_recovery `
  benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_* `
  --output-dir benchmark/results/lineage_signal_10leaf_ancestor_recovery `
  --run-name ancestor_time1

python -m benchmark.summarize_dataset `
  --config benchmark/configs/lineage_signal_10leaf_10rep.json `
  benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_* `
  --output-dir benchmark/results/lineage_signal_10leaf_benchmark
```

## Sensitivity analysis

### CTMC model misspecification

The primary ancestral-reconstruction analysis evaluates the conditional target
`P(S_ancestors | D_leaves, T, Q, pi_root)` directly. It holds the ten simulated
regulatory histories and underlying leaf GRN weights fixed, then changes one
inference assumption at a time. Only the terminal-temperature conditions alter
the calibration that maps those fixed weights to evidence probabilities. This
separates ancestral model misspecification from RNA regression optimization.
The 22-condition grid contains:

- a reference condition with the generating `Q` and branch lengths plus the
  unperturbed edge-specific ATAC-derived root prior;
- gain, loss, and sign-switch rates at `0.25x`, `0.5x`, `2x`, and `4x` truth;
- shared positive-state/gain odds at `0.5x` and `2x` (for reference `pi+=0.60`,
  the inferred fractions are `0.4286` and `0.75`);
- edge-specific ATAC root-prior odds at `0.5x` and `2x`, preserving the prior
  source and edge ranking so this remains a one-factor-at-a-time comparison;
- all branch lengths at `0.5x` and `2x` their true values;
- terminal edge/sign evidence temperatures at `0.5x` and `2x` defaults;
- a topology-only cross-clade perturbation. Branch lengths are not compressed in
  this condition.

Probability parameters are changed by multiplying odds rather than probabilities:
`p' = c p / (1 - p + c p)`. Thus `c` and `1/c` are symmetric on the logit scale,
remain strictly valid probabilities, and never require clipping. The analysis
deliberately avoids a combinatorial Cartesian product.

After running the generation command above, execute the exact analysis. The
`--runs-dir` form performs deterministic replicate discovery on Windows, macOS,
and Linux without relying on shell wildcard expansion:

```powershell
python -m benchmark.run_fixed_evidence_sensitivity `
  --runs-dir benchmark/runs/lineage_signal_10leaf `
  --output-dir benchmark/results/lineage_signal_10leaf_model_misspecification

python -m benchmark.plot_model_misspecification `
  --results-dir benchmark/results/lineage_signal_10leaf_model_misspecification `
  --output-dir benchmark/results/journal_figures `
  --prefix 10leaf_model_misspecification
```

The primary figure reports replicate-paired change in AUPRC relative to the
reference condition; dots are independent replicates and error bars are
percentile 95% bootstrap confidence intervals over replicate-level differences.
A companion local-scale figure resolves the smaller non-topology effects while
keeping one common scale across those seven panels. Leaves are a fixed-weight
negative control. Signed precision at true K is supplementary. Formal plotting
refuses fewer than three replicates. Smoke data can only be plotted with
`--preview`, which adds a visible warning watermark.

An optional, substantially slower end-to-end sensitivity reruns the full RNA
optimization for every condition:

```powershell
python -m benchmark.run_model_misspecification `
  --runs-dir benchmark/runs/lineage_signal_10leaf `
  --output-dir benchmark/results/model_misspecification_end_to_end `
  --processes 4 --optimizer-maxfun 180000
```

That runner additionally writes optimizer convergence diagnostics and uses a
fingerprinted cache, so results from obsolete condition definitions cannot be
silently reused. Generated branch/topology inputs remain under
`ablations/model_misspecification_inputs/`; the saved fate map is never modified.

### Data-size and optimizer sensitivity

Create additional JSON configurations by changing:

- `fate_map.node_count` and the corresponding `node_count - 1` branch lengths;
- `grn.loss_rate`, `grn.gain_rate`, and `grn.sign_switch_rate`;
- `grn.regulator_count`, `grn.target_count`, and `grn.density`;
- `observations.cells_per_leaf`;
- optimizer selection in `benchmark.run_regarch`.

Balanced node counts supported directly are 7, 15, 31, and so on.

Ready-to-edit sensitivity configurations can be generated with:

```powershell
python -m benchmark.sensitivity `
  --base-config benchmark/configs/main.json `
  --output benchmark/configs/generated_sensitivity
```

This creates 7-node and 15-node topology settings plus low, medium, and high
rewiring-rate settings. Run each inference with all three optimizer options.

## Tests

```powershell
python -m pytest -q
```

External methods run in pinned Conda, R, or Docker environments. The
implementation level of every method is recorded in
`method_implementations.json`; synthetic-resource adapters are never described
as native end-to-end workflows.

## Installed external environments

The current Windows workstation uses isolated environments because several
methods do not support native Windows installation:

- CellOracle 0.18.0: official Docker image
  `kenjikamimoto126/celloracle_ubuntu:0.18.0`;
- pySCENIC 0.12.1: Conda environment `lg_pyscenic`, with NumPy 1.23.5;
- Dictys: official CPU Docker image `lfwa/dictys-cpu:latest`;
- FigR 0.1.0, Pando 1.1.1, and GRaNIE 1.6.1: local R library;
- LINGER: reproducible image recipe in `benchmark/docker/linger.Dockerfile`.
- SCENIC+ 1.0a2: reproducible minimal eGRN image in
  `benchmark/docker/scenicplus.Dockerfile`;
- scMTNI: official source compiled by `benchmark/docker/scmtni.Dockerfile`.

Run CellOracle on the shared leaf matrices and ATAC-derived base GRN:

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace `
  kenjikamimoto126/celloracle_ubuntu:0.18.0 `
  python benchmark/methods/run_celloracle.py `
  benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001 --bagging-number 50 --n-jobs 1

python -m benchmark.standardize_external `
  benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001 --method CellOracle `
  --output benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001/method_outputs/standardized/CellOracle.tsv
```

SCENIC, SCENIC+, FigR, Pando, GRaNIE, LINGER, and Dictys require motif,
genome-annotation, pretrained-model, or method-specific preprocessing
resources beyond package installation. For synthetic TF names, use generated
synthetic resources rather than human or mouse databases. An installed method
is therefore reported as `installed_not_run` until native output tables exist.

### Synthetic-resource benchmark runs

The pilot benchmark uses the generated motif hits and peak-gene candidates in
place of human or mouse databases. No method reads ground-truth GRN files.

```powershell
# SCENIC: reuse the executed GRNBoost adjacency, then apply leaf-specific
# synthetic cis-regulatory pruning.
conda run -n lg_pyscenic python benchmark/methods/run_scenic.py `
  benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001

# FigR, Pando, and GRaNIE inference cores.
Rscript benchmark/methods/run_r_multiome_methods.R `
  benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001

# Dictys OU reconstruction after preparing the shared binary chromatin mask.
python benchmark/methods/prepare_dictys.py benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001

# LINGER baseline trans-regulatory matrix reconstruction.
docker run --rm -v "${PWD}:/workspace" -w /workspace `
  regarch-linger:1.105 python benchmark/methods/run_linger.py `
  benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001

# SCENIC+: official build_grn/create_emodules implementation with the
# generated motif cistromes and region-gene candidates.
docker build -f benchmark/docker/scenicplus.Dockerfile `
  -t regarch-scenicplus:latest ../../external_tools/scenicplus
docker run --rm -v "${PWD}:/repo" regarch-scenicplus:latest `
  /repo/benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001 --source /opt/scenicplus

# scMTNI: official C++ implementation and an observed-leaf tree projection.
docker build -f benchmark/docker/scmtni.Dockerfile `
  -t regarch-scmtni:latest ../../external_tools/scMTNI
python -m benchmark.methods.run_scmtni benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_001
```

`run_r_multiome_methods.R` uses FigR DORC scoring, Pando elastic-net models,
and the GRaNIE TF-peak/peak-gene correlation structure. GRaNIE's standard
object builder cannot annotate synthetic gene identifiers through Ensembl, so
the adapter supplies the generated annotations directly. LINGER receives its
standard TF-RE and RE-target intermediate matrices. Dictys receives a square
ATAC-derived edge mask and is fitted independently for each leaf.

SCENIC+ first uses its native GSEA eGRN builder. With the 20-gene validation
network, GSEA can return no retained eRegulons. The preregistered fallback then
scores candidate TF-target pairs from the product of motif, region-gene, and
TF-gene evidence. It reads only the shared method inputs, covers at least 25%
of the gene universe, and never reads truth GRNs.

scMTNI requires expression for every node in its lineage tree, so latent
ancestors cannot be supplied without giving it extra observations. The adapter
therefore uses a leaf-only projection that preserves the two sister pairs and
uses path-length-derived gain/loss probabilities. Input files are forced to LF
line endings because the official parser treats Windows carriage returns as
part of cell-type names.

## Task 1 multi-method runner

The unified runner covers RegArch, a seeded random negative control, and the
11 comparison methods in the current Task 1 request. Every biological method
manifest points to leaf subsets of the same RNA and ATAC matrices.
`shared_input_contract.json` stores SHA-256 hashes; the runner aborts if any
shared observation changes between method runs.

Generate the retained ten-repeat leaf benchmark and execute all methods:

```powershell
python -m benchmark.cli generate `
  --config benchmark/configs/lineage_signal_10leaf_10rep.json `
  --output benchmark/runs/lineage_signal_10leaf

python -m benchmark.run_benchmark run `
  benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_* `
  --processes 4 --lineage-weight 0.5 --force
```

Use `--force` to discard and recompute method outputs for those replicate
directories. Raw outputs remain under `method_outputs/raw/<method>/`, and
common-schema outputs are written under `method_outputs/standardized/`.

Aggregate, diagnose output coverage, and draw the benchmark figure:

```powershell
python -m benchmark.run_benchmark aggregate `
  benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_* `
  --output-dir benchmark/results/lineage_signal_10leaf_benchmark
```

Audit every completed method output:

```powershell
python -m benchmark.audit_benchmark `
  benchmark/runs/lineage_signal_10leaf/lineage_signal_10leaf_rep_* `
  --output benchmark/results/lineage_signal_10leaf_benchmark/method_audit_summary.tsv
```

The comparison reports per-leaf AUPRC/AUROC and top-K metrics, paired mean
differences against RegArch, bootstrap confidence intervals, win rates, and
one-sided Wilcoxon tests with Holm correction. `advantage_supported` requires a
positive mean difference, a bootstrap confidence interval above zero, and
Holm-adjusted p-value below 0.05. `method_output_diagnostics.tsv` additionally
reports each method's nonzero edge coverage and the corresponding recall
ceiling.

`Random` assigns an independent seeded `Uniform(0, 1)` score to every edge in
the shared universe. It never reads truth labels, RNA, or ATAC. Its expected
AUROC is 0.5 and its expected AUPRC/precision-at-K equal edge prevalence.

The validated benchmark outputs are under
`results/lineage_signal_10leaf_benchmark/`. The directory includes aggregate
metrics, paired comparisons against RegArch and Random, output diagnostics,
SCENIC+ fallback metadata, simulation metadata, and the benchmark figure.
Ancestor recovery is stored separately under
`results/lineage_signal_10leaf_ancestor_recovery/`, and ablation outputs are
stored under `results/lineage_signal_10leaf_ablations/`.

`METHOD_AUDIT.md` documents the independent reproducibility audit for
GENIE3, GRNBoost2, and SCENIC. The audit corrected SCENIC score-column
selection and removed prohibited self-regulatory candidates from the common
evaluation universe.

`method_comparability.tsv` defines the exact label that may be used for each
method. Adapter-qualified entries must not be presented as unrestricted
end-to-end evaluations of their complete official packages.




