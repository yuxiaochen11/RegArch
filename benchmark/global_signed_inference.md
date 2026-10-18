# Global signed RegArch inference

## State model

Each regulator-target edge has state `S(v) in {-1, 0, +1}` at fate-map node
`v`. For a branch of length `t`, inference uses `P(t) = exp(Qt)` with the same
generator as simulation:

```text
             child -1       child 0       child +1
parent -1  -(loss+switch)     loss          switch
parent  0   gain*(1-pos)     -gain          gain*pos
parent +1      switch          loss       -(loss+switch)
```

The rows above describe rates before exponentiation. `gain`, `loss`,
`switch`, `pos`, and every branch length are read from the simulation
configuration used to create that replicate.

## Global inference

RNA likelihood and ATAC evidence are observed only at leaves. Leaf continuous
coefficients are initialized with ATAC magnitude and RNA ridge-regression sign,
then optimized jointly with one global tree likelihood per candidate edge.
Exact upward and downward sum-product messages produce posterior probabilities
for all three states at every node.

The standardized score has the same meaning at every node:

```text
abs(score) = P(S(v) != 0 | all observed leaves)
sign(score) = MAP sign conditional on edge presence
```

The method therefore infers ancestors from all descendant evidence instead of
averaging independently fitted leaf networks. Continuous ancestral effect
magnitude is not identifiable from leaf-only observations and is not reported.

## Fair comparison contract

- All 12 methods receive leaf subsets of the same simulated RNA and ATAC data.
- The 11 comparison methods are inferred and evaluated only at observed leaves.
- RegArch leaf scores are evaluated on the identical leaf edge universe.
- RegArch ancestor scores are evaluated separately on the all-node or
  ancestor-only edge universe.
- Presence metrics use absolute posterior score. Signed metrics additionally
  require the predicted activation/repression state to be correct.
- Input hashes, optimizer diagnostics, and per-node posterior files are retained
  for every replicate.

## Reproducible validation

```powershell
$reps = Get-ChildItem benchmark/runs/lineage_signal_10leaf `
  -Directory -Filter "lineage_signal_10leaf_rep_*" |
  Sort-Object Name | Select-Object -ExpandProperty FullName
python -m benchmark.validate_matched_simulation @reps `
  --output-dir benchmark/results/lineage_signal_10leaf_ancestor_recovery `
  --run-name full
```


