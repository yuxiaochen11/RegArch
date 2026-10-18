# Benchmark Results Overview

This file summarizes the retained benchmark result directories after cleanup.
The primary paper-ready simulation setting is `lineage_signal_10leaf`, which
uses an ultrametric fate map with 10 observed leaves, stronger between-clade GRN
rewiring, matched RNA/ATAC inputs for all compared methods, and 10 independent
replicates.

## Retained Result Layout

| Directory | Purpose | Status |
|---|---|---|
| `benchmark/results/lineage_signal_10leaf_benchmark` | Primary 10-leaf cross-method benchmark, method diagnostics, simulation metadata, and figures. | Retained benchmark result |
| `benchmark/results/lineage_signal_10leaf_ancestor_recovery` | RegArch-only ancestor-node recovery on the 10-leaf data. | Retained ancestor result |
| `benchmark/results/lineage_signal_10leaf_ablations` | 10-leaf fate-map and ATAC-prior ablation study. | Retained ablation result |
| `benchmark/results/lineage_signal_10leaf_atac_sensitivity` | 10-leaf sensitivity analysis for controlled ATAC-prior coverage and correctness. | Retained sensitivity result |
| `benchmark/results/lineage_signal_10leaf_model_misspecification` | Ten-replicate fixed-leaf-weight CTMC, root-prior, branch-length, and topology sensitivity. | Retained sensitivity result |
| `benchmark/results/lineage_signal_15leaf_benchmark` | 15-leaf cross-method benchmark reusing the same `lineage_signal_15leaf` data; includes all 12 GRN methods plus Random. | Retained benchmark result |
| `benchmark/results/lineage_signal_15leaf_sensitivity` | 15-leaf RegArch-only sensitivity analysis with runtime tracking. | Retained sensitivity result |
| `benchmark/results/lineage_signal_15leaf_atac_sensitivity` | 15-leaf sensitivity analysis for controlled ATAC-prior coverage and correctness, with runtime tracking. | Retained sensitivity result |
| `benchmark/results/hyperparameter_10leaf_sensitivity` | 10-leaf sensitivity analysis for cell number, gene number, and optimizer settings. | Retained sensitivity result |

Raw replicate directories are generated on demand and intentionally excluded
from the compact repository. Recreate them from the versioned JSON configs with
`python -m benchmark.cli generate`; `benchmark/README.md` gives exact commands.
Retained result directories include parameter tables and input SHA-256 manifests
needed to audit regenerated inputs.

## Experiments Completed

### 1. Primary 10-Leaf Leaf-GRN Benchmark

Location: `benchmark/results/lineage_signal_10leaf_benchmark/`

All methods were evaluated on the same simulated leaf RNA-seq and ATAC-seq data.
External methods were evaluated only on observed leaves, while RegArch was
also used for ancestor-node recovery. The comparison includes RegArch,
CellOracle, SCENIC, SCENIC+, GENIE3, GRNBoost, FigR, Pando, GRaNIE, LINGER,
Dictys, scMTNI, and a Random control.

Key mean leaf results:

| Method | Mean AUPRC | Mean AUROC | Mean Precision@K |
|---|---:|---:|---:|
| RegArch | 0.6791 | 0.9385 | 0.6008 |
| LINGER | 0.5819 | 0.9284 | 0.5358 |
| SCENIC | 0.5668 | 0.9326 | 0.5178 |
| SCENIC+ | 0.5551 | 0.9278 | 0.5098 |
| Random | 0.1444 | 0.5018 | 0.1130 |

Conclusion: under the lineage-signal simulation, RegArch ranks first for the
primary leaf-GRN AUPRC and also improves precision at the true edge count. The
result supports the claim that a correctly specified fate-map prior can improve
leaf GRN reconstruction when related cell types share informative but
branch-specific regulatory structure.

### 2. RegArch Ancestor Recovery

Location: `benchmark/results/lineage_signal_10leaf_ancestor_recovery/`

Only RegArch is evaluated on unobserved internal/root nodes, because the
other benchmark methods infer GRNs from observed molecular profiles. The same
posterior-score definition is used for leaves and ancestors:

`abs(score) = posterior edge-presence probability`, and the sign is the MAP
activation/repression state conditional on edge presence.

Key result:

| Node group | Mean AUPRC |
|---|---:|
| Observed leaves | 0.6791 |
| Ancestors/internal/root | 0.6364 |

Conclusion: ancestor recovery is lower than observed leaf recovery but remains
well above random, which is the expected qualitative behavior for historical
cell types inferred only from descendant observations and the fate-map model.

### 3. Fate-Map and ATAC Ablation

Location: `benchmark/results/lineage_signal_10leaf_ablations/`

Main ablation summary:

| Condition | Leaf Mean AUPRC | Ancestor Mean AUPRC | Interpretation |
|---|---:|---:|---|
| Full RegArch | 0.6791 | 0.6364 | Full matched model |
| No fate map | 0.6371 | Not interpreted | Removes lineage coupling for leaf comparison |
| Wrong fate map | 0.6361 | 0.6129 | Uses deliberately mismatched topology |
| Uniform root prior | 0.6496 | 0.6200 | Removes edge-specific ATAC root prior |
| No ATAC | 0.2757 | 0.2547 | Removes chromatin prior |

Conclusion: both fate-map structure and ATAC evidence contribute, but the ATAC
prior is the stronger driver in this simulation. Wrong-map performance being
similar to no-fate-map means that the implementation is not catastrophically
misled by this specific topology perturbation; it should not be overclaimed as
proof that arbitrary fate-map errors are harmless.

### 4. 15-Leaf Cross-Method Benchmark

Location: `benchmark/results/lineage_signal_15leaf_benchmark/`

The 15-leaf dataset was also evaluated with the full cross-method benchmark.
All methods were run or reused on the same 10 simulated replicates, giving 150
leaf-level evaluations per method.

| Method | Mean AUPRC | Mean AUROC | Mean Precision@K |
|---|---:|---:|---:|
| RegArch | 0.5847 | 0.9038 | 0.5431 |
| SCENIC+ | 0.4966 | 0.8876 | 0.4717 |
| SCENIC | 0.4943 | 0.8954 | 0.4674 |
| CellOracle | 0.4903 | 0.8745 | 0.4530 |
| LINGER | 0.4825 | 0.8898 | 0.4554 |
| Random | 0.1426 | 0.4886 | 0.1146 |

Conclusion: RegArch remains the top-ranked method in the larger 15-leaf
setting. The absolute AUPRC is lower than the primary 10-leaf benchmark, which
is consistent with the 15-leaf dataset being a harder scalability setting.

### 5. 15-Leaf Fate-Map Size Sensitivity

Location: `benchmark/results/lineage_signal_15leaf_sensitivity/`

This is the RegArch-only sensitivity analysis that originally tested whether
the same inference machinery remains usable when the number of observed leaves
increases from 10 to 15. The full 15-leaf external-method benchmark has now
also been run separately under `lineage_signal_15leaf_benchmark`.

| Leaf count | Mean AUPRC | Mean AUROC | Mean Precision@K | Mean runtime per replicate |
|---:|---:|---:|---:|---:|
| 10 | 0.6791 | 0.9385 | 0.6008 | 2.07 min |
| 15 | 0.5847 | 0.9038 | 0.5431 | 25.90 min |

Conclusion: RegArch remains above random and retains useful reconstruction
accuracy at 15 leaves, but runtime increases substantially. The larger topology
should be presented as a computational and statistical stress test.

### 6. ATAC Coverage and Correctness Sensitivity

Locations:

- `benchmark/results/lineage_signal_10leaf_atac_sensitivity/`
- `benchmark/results/lineage_signal_15leaf_atac_sensitivity/`

The controlled prior settings vary coverage and correctness while leaving the
simulation and inference code unchanged. `100% coverage, 100% correctness` is
an oracle upper-bound prior, not the same as the realistic full model.

Selected mean AUPRC values:

| Coverage | Correctness | 10-leaf AUPRC | 15-leaf AUPRC |
|---:|---:|---:|---:|
| 100% | 100% | 0.7284 | 0.7402 |
| 75% | 100% | 0.6928 | 0.6688 |
| 50% | 100% | 0.6302 | 0.6090 |
| 25% | 100% | 0.5186 | 0.5429 |
| 100% | 90% | 0.6899 | 0.7229 |
| 100% | 75% | 0.6503 | 0.6669 |
| 100% | 50% | 0.5787 | 0.6239 |
| 25% | 50% | 0.3884 | 0.3800 |

Conclusion: performance generally declines as ATAC prior coverage or correctness
decreases. This supports the biological claim that chromatin accessibility is
useful because it constrains the candidate regulatory edge space.

### 7. Ten-Leaf Hyperparameter Sensitivity

Location: `benchmark/results/hyperparameter_10leaf_sensitivity/`

The ten-leaf simulation and inference code were kept fixed while varying cells
per leaf, gene/regulator count, and optimizer settings. These runs used fewer
replicates than the primary benchmark and should be treated as sensitivity
diagnostics.

Selected mean results:

| Setting | Mean AUPRC | Mean AUROC | Mean Precision@K | Mean runtime |
|---|---:|---:|---:|---:|
| Baseline, 30 cells/leaf, 20 genes, L-BFGS-B | 0.6391 | 0.9232 | 0.5588 | 9.91 min |
| 15 cells/leaf | 0.6264 | 0.9286 | 0.5490 | 8.08 min |
| 60 cells/leaf | 0.7111 | 0.9346 | 0.6133 | 8.66 min |
| 40 genes, 12 regulators | 0.5676 | 0.9219 | 0.5416 | 37.14 min |
| L-BFGS-B, lower maxfun | 0.5943 | 0.9180 | 0.5153 | 4.79 min |
| L-BFGS-B, higher maxfun | 0.6356 | 0.9243 | 0.5660 | 10.29 min |
| SLSQP, lower maxfun | 0.6131 | 0.9193 | 0.5289 | 2.32 min |
| COBYLA, lower maxfun | 0.3326 | 0.7859 | 0.3142 | 0.60 min |

Conclusion: more cells improve recovery, larger gene universes make inference
harder and slower, and L-BFGS-B remains the preferred optimizer. SLSQP and
COBYLA are useful as negative optimization controls but are not recommended for
final inference because of poor convergence diagnostics.

### 8. Ancestral Model-Misspecification Sensitivity

Location: `benchmark/results/lineage_signal_10leaf_model_misspecification/`

This exact-posterior experiment holds the underlying simulated leaf GRN weights
fixed and changes one ancestral-inference assumption at a time across 10
independent replicates; terminal-temperature conditions alone recalibrate their
evidence probabilities. Reference-condition mean AUPRC is 0.8552 at the root and 0.9787 at
internal nodes. The pure topology perturbation has the largest effect, reducing
internal-node AUPRC by 0.2961 (95% paired bootstrap CI 0.2857 to 0.3081 in
absolute magnitude); fourfold gain/loss-rate changes reduce root AUPRC by only
0.0095 and 0.0116. This supports local conditional robustness, not a universal
or end-to-end robustness guarantee.

## Recommended Remaining Experiments

1. Increase hyperparameter sensitivity replicates from 3 to 10 before using
   those values as final paper evidence.
2. Extend fate-map-noise sensitivity beyond the current branch scaling and pure
   topology condition: partial leaf swaps, collapsed internal nodes, and missing
   subtrees.
3. Add RNA noise/dropout sensitivity independently from ATAC quality.
4. Run the optional full end-to-end model-misspecification refits on 10
   replicates if claims need to include RNA-regression re-optimization rather
   than conditional ancestral reconstruction alone.

## Interpretation Boundaries

- The primary cross-method claim should rely on
  `lineage_signal_10leaf`, not on obsolete earlier runs.
- The 15-leaf cross-method result is now available under
  `lineage_signal_15leaf_benchmark`; the separate
  `lineage_signal_15leaf_sensitivity` directory should still be used for
  RegArch runtime/scalability interpretation.
- Controlled `100% coverage, 100% correctness` ATAC priors are oracle upper
  bounds and should not be described as realistic raw-data-derived inputs.
- Wrong-map robustness here means robustness to the implemented perturbation,
  not to all possible fate-map errors.

