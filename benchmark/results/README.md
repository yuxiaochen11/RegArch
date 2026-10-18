# Benchmark Results

The retained benchmark outputs are organized around the current
`lineage_signal` simulation series.

Start with `BENCHMARK_RESULTS_OVERVIEW.md` for a compact map of all retained
experiments, main conclusions, and recommended remaining analyses.
Use `EXPERIMENT_COVERAGE_MATRIX.md` to check which experiment types are matched
between the 10-leaf and 15-leaf datasets.
`LINEAGE_SIGNAL_10LEAF_RESULTS.md` is the scenario-level narrative report that
links the 10-leaf benchmark, ancestor recovery, and ablation outputs.

Primary result:

- `lineage_signal_10leaf_benchmark/`: 10-leaf cross-method benchmark, method
  diagnostics, simulation metadata, and figures.
- `lineage_signal_15leaf_benchmark/`: 15-leaf cross-method benchmark using the
  same `lineage_signal_15leaf` data.
- `lineage_signal_10leaf_ancestor_recovery/`: RegArch-only ancestor-node
  recovery on the 10-leaf data.
- `lineage_signal_10leaf_ablations/`: 10-leaf fate-map and ATAC-prior
  ablation study.

Sensitivity results:

- `lineage_signal_10leaf_model_misspecification/`: 10-leaf exact conditional
  sensitivity to CTMC rates, positive-state odds, root-prior calibration,
  branch lengths, terminal-evidence calibration, and topology.
- `lineage_signal_10leaf_atac_sensitivity/`: 10-leaf ATAC coverage/correctness
  sensitivity.
- `lineage_signal_15leaf_sensitivity/`: 15-leaf RegArch scalability and
  runtime sensitivity.
- `lineage_signal_15leaf_atac_sensitivity/`: 15-leaf ATAC coverage/correctness
  sensitivity.
- `hyperparameter_10leaf_sensitivity/`: 10-leaf cell-number, gene-number, and
  optimizer sensitivity.

