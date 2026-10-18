# 10-Leaf vs 15-Leaf Experiment Coverage Matrix

This file checks whether the 10-leaf and 15-leaf `lineage_signal` datasets have
the same categories of experiments.

## Summary

The 10-leaf and 15-leaf datasets both have cross-method benchmarks and ATAC
coverage/correctness sensitivity analyses. They are therefore matched for the
main benchmark claim and for the ATAC-prior quality sensitivity claim.

They are not yet fully matched for every experiment type:

- The full fate-map/ATAC ablation study exists for 10 leaves only.
- The 15-leaf dataset has RegArch scalability/runtime sensitivity, but the
  corresponding leaf-count comparison is naturally defined as 10 vs 15 rather
  than duplicated inside 10 leaves.
- The hyperparameter sensitivity study currently exists for 10 leaves only.

## Coverage Table

| Experiment type | 10-leaf location | 15-leaf location | Matched? | Notes |
|---|---|---|---|---|
| Cross-method leaf benchmark | `lineage_signal_10leaf_benchmark/` | `lineage_signal_15leaf_benchmark/` | Yes | Both include RegArch, 11 comparison methods, and Random on the same replicate-specific RNA/ATAC data. |
| Random control | `lineage_signal_10leaf_benchmark/random_control_summary.tsv` | `lineage_signal_15leaf_benchmark/random_control_summary.tsv` | Yes | Both benchmarks report random AUPRC/AUROC references. |
| Method diagnostics | `lineage_signal_10leaf_benchmark/method_output_diagnostics.tsv` | `lineage_signal_15leaf_benchmark/method_output_diagnostics.tsv` | Yes | 15-leaf additionally has `method_output_diagnostics_summary.tsv`. |
| RegArch ancestor recovery | `lineage_signal_10leaf_ancestor_recovery/` | `lineage_signal_15leaf_sensitivity/` | Partly | Both have RegArch ancestor metrics, but they are stored under different analysis types. |
| Fate-map ablation: full/no fate/wrong fate | `lineage_signal_10leaf_ablations/` | Missing | No | 15-leaf has not yet run the same no-fate-map/wrong-fate-map ablation grid. |
| ATAC ablation/sensitivity: coverage/correctness | `lineage_signal_10leaf_atac_sensitivity/` | `lineage_signal_15leaf_atac_sensitivity/` | Yes | Both include matched controlled ATAC-prior quality conditions. |
| No-ATAC ablation | `lineage_signal_10leaf_ablations/` | Missing as formal ablation | No | 15-leaf ATAC sensitivity includes low-quality priors, but not the same formal `no_atac` condition from the 10-leaf ablation table. |
| Leaf-count sensitivity | Baseline side of comparison | `lineage_signal_15leaf_sensitivity/` | Yes for 10 vs 15 comparison | Stored primarily in the 15-leaf sensitivity directory as `sensitivity_10leaf_vs_15leaf.tsv`. |
| Cell/gene/optimizer hyperparameter sensitivity | `hyperparameter_10leaf_sensitivity/` | Missing | No | Hyperparameter grid has not yet been repeated for 15 leaves. |

## Key Benchmark Results

### 10 leaves

| Rank | Method | Mean AUPRC | Mean AUROC | Mean Precision@K | Paired leaves |
|---:|---|---:|---:|---:|---:|
| 1 | RegArch | 0.6791 | 0.9385 | 0.6008 | 100 |
| 2 | LINGER | 0.5819 | 0.9284 | 0.5358 | 100 |
| 3 | SCENIC | 0.5668 | 0.9309 | 0.5326 | 100 |
| 4 | SCENIC+ | 0.5551 | 0.9201 | 0.5078 | 100 |
| 5 | CellOracle | 0.5169 | 0.9010 | 0.4721 | 100 |

### 15 leaves

| Rank | Method | Mean AUPRC | Mean AUROC | Mean Precision@K | Paired leaves |
|---:|---|---:|---:|---:|---:|
| 1 | RegArch | 0.5847 | 0.9038 | 0.5431 | 150 |
| 2 | SCENIC+ | 0.4966 | 0.8876 | 0.4717 | 150 |
| 3 | SCENIC | 0.4943 | 0.8954 | 0.4674 | 150 |
| 4 | CellOracle | 0.4903 | 0.8745 | 0.4530 | 150 |
| 5 | LINGER | 0.4825 | 0.8898 | 0.4554 | 150 |

## Key Ablation and Sensitivity Results

### 10-leaf formal ablation

| Condition | Node group | Mean AUPRC | Mean AUROC | Mean Precision@K |
|---|---|---:|---:|---:|
| full | leaf | 0.6791 | 0.9385 | 0.6008 |
| no_fate_map | leaf | 0.6371 | 0.8949 | 0.5961 |
| wrong_fate_map | leaf | 0.6361 | 0.9288 | 0.5469 |
| uniform_root | leaf | 0.6496 | 0.8757 | 0.5981 |
| no_atac | leaf | 0.2757 | 0.6302 | 0.2802 |
| full | ancestor | 0.6364 | 0.9381 | 0.6001 |
| no_atac | ancestor | 0.2547 | 0.6302 | 0.2444 |

### ATAC-prior quality sensitivity

Both 10-leaf and 15-leaf datasets include controlled ATAC-prior coverage and
correctness conditions. The 15-leaf example shows the expected degradation from
oracle-quality priors to low-coverage/low-correctness priors:

| 15-leaf condition | Mean AUPRC | Mean AUROC | Mean Precision@K |
|---|---:|---:|---:|
| prior_cov100_cor100 | 0.7402 | 0.9366 | 0.6314 |
| prior_cov075_cor100 | 0.6688 | 0.9174 | 0.5512 |
| prior_cov050_cor100 | 0.6090 | 0.8952 | 0.5129 |
| prior_cov025_cor100 | 0.5429 | 0.8466 | 0.4863 |
| prior_cov100_cor050 | 0.6239 | 0.8716 | 0.5591 |
| prior_cov025_cor050 | 0.3800 | 0.7577 | 0.3564 |

## Recommendation

For a fully symmetric final result package, run the following additional
15-leaf analyses:

1. A 15-leaf formal ablation grid matching the 10-leaf conditions:
   `full`, `no_fate_map`, `wrong_fate_map`, `uniform_root`, and `no_atac`.
2. A 15-leaf hyperparameter sensitivity subset, at minimum:
   baseline optimizer, low/high cells per leaf, larger gene count, and one
   reduced-iteration optimizer stress condition.

For the current manuscript, the existing results are already sufficient for:

- 10-leaf and 15-leaf cross-method benchmark comparison.
- 10-leaf formal ablation conclusions.
- 10-leaf and 15-leaf ATAC-prior quality sensitivity.
- 10-to-15 leaf scalability/runtime sensitivity.

The only claim that should not be stated as matched across both leaf counts yet
is: "formal fate-map/no-ATAC ablation was performed identically for both 10 and
15 leaves."
