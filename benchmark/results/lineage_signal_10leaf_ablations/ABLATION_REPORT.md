# RegArch ablation study

## Design

- Ten independently generated final-benchmark replicates were reused.
- RNA, truth GRNs, candidate edges, optimization settings, and evaluation metrics were fixed.
- Statistical tests average correlated nodes within each replicate before paired testing.
- `no_fate_map` is evaluated only at leaves because it has no identifiable ancestor output.
- Wrong-topology internal clades cross the true clades; only leaves and the shared Root are evaluated.
- Controlled ATAC priors use truth only to create known-quality perturbations before inference.
  The 100%/100% prior is an oracle upper bound, not a realistic benchmark input.

## Primary AUPRC

| Condition | Leaf | Ancestor |
|---|---:|---:|
| full | 0.6791 | 0.6364 |
| no_fate_map | 0.6259 | NA |
| wrong_fate_map | 0.6367 | 0.6027 |

## Interpretation safeguards

- High sign accuracy in a weak condition is conditional on the small subset of true edges recovered;
  signed precision at K should be used to assess joint edge-and-sign recovery.
- AUPRC and AUROC assess the complete ranking. Precision at one cutoff can occasionally improve
  even when the global ranking deteriorates.
- Holm-adjusted paired tests are available in `paired_comparisons.tsv`.

All target-gene fits and convergence counts are recorded in `optimization_convergence.tsv`.
