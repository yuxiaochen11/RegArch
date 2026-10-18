# Fixed-leaf-weight model-misspecification sensitivity

These retained results were regenerated with the current RegArch repository and
`benchmark/configs/lineage_signal_10leaf_10rep.json`. They contain 10 independent
replicates, 22 one-factor-at-a-time inference conditions, and separate root,
internal, and leaf evaluations. No low-budget optimizer smoke result is included.

The analysis isolates
`P(S_ancestors | D_leaves, T, Q, pi_root)`: simulated leaf regulatory weights are
held fixed, and exact three-state CTMC sum-product inference is rerun under each
perturbed assumption. Only the temperature conditions change the mapping from
those weights to evidence probabilities. Therefore this result supports claims
about ancestral reconstruction conditional on leaf evidence; it is not an
end-to-end RNA-regression sensitivity result.

## Conditions

The reference condition uses the generating CTMC rates and branch lengths plus
the unperturbed edge-specific ATAC-derived root prior. It is a data-informed
reference, not an oracle copy of the simulated root state.

- `lambda_gain`, `lambda_loss`, and `lambda_switch`: `0.25x`, `0.5x`, `1x`,
  `2x`, and `4x` the generating values.
- Shared positive-state/gain fraction `pi+`: odds multiplied by `0.5` or `2`
  (`pi+=0.4286`, `0.60`, and `0.75` in this benchmark).
- Edge-specific ATAC root prior: odds multiplied by `0.5` or `2`; the ATAC
  source and edge ranking remain unchanged.
- All branch lengths and both terminal-evidence temperatures: `0.5x`, `1x`,
  and `2x`.
- Topology: a cross-clade terminal reassignment with developmental branch
  lengths retained. It does not use the separate short-branch negative control.

## Figure interpretation

The primary figure is
`../journal_figures/10leaf_model_misspecification_delta_auprc.pdf`. Each faint
point is one independent replicate. Solid markers show the mean paired change
from the reference condition; error bars are deterministic percentile 95% bootstrap
confidence intervals over the 10 replicate-level differences. Root, ancestral,
and leaf nodes are averaged separately within each replicate before inference
conditions are compared. The fixed leaf weights form a negative control; leaf
AUPRC remains at zero paired change in this design.

The companion
`../journal_figures/10leaf_model_misspecification_local_delta_auprc.pdf` uses a
common enlarged y-scale across the seven non-topology factors so their smaller
effects and confidence intervals remain readable. The primary figure retains a
single global scale, making the much larger topology effect directly visible.

Reference-condition mean AUPRC was 0.8552 at the root, 0.9787 across internal ancestors, and
1.0000 at leaves. The topology-only perturbation produced the dominant effect:
internal-node AUPRC changed by -0.2961 (95% bootstrap CI -0.3081 to -0.2857).
In contrast, even fourfold gain/loss-rate perturbations changed root AUPRC by
-0.0095 and -0.0116, respectively. These values describe this simulation regime
and should not be generalized as universal robustness guarantees.

The signed precision-at-true-K figure is supplementary. Plot-ready paired
estimates and intervals are stored beside each figure; raw per-node metrics,
replicate summaries, actual inference parameters, condition definitions, and
SHA-256 manifests for both source inputs and derived topology artifacts are
retained in this directory. Some topology-condition sign-accuracy/coverage
entries are structurally undefined when no recovered true edge has an available
sign; the primary AUPRC and signed-precision metrics remain defined throughout.
