# Benchmark 与消融实验结果：`lineage_signal_10leaf_10rep`

本文档总结 `lineage_signal_10leaf_10rep` 场景下的叶节点 GRN 重建 benchmark、RegArch 祖先节点恢复结果，以及 fate map/ATAC 消融实验。结果已拆分存储：

- benchmark：`benchmark/results/lineage_signal_10leaf_benchmark/`
- 祖先节点恢复：`benchmark/results/lineage_signal_10leaf_ancestor_recovery/`
- 消融实验：`benchmark/results/lineage_signal_10leaf_ablations/`

## 1. 评估任务

### 1.1 叶节点 GRN 重建

叶节点 `T1-T10` 有模拟 scRNA-seq 和 scATAC-seq 观测数据。比较对象包括：

- RegArch；
- CellOracle；
- SCENIC；
- SCENIC+；
- GENIE3；
- GRNBoost；
- FigR；
- Pando；
- GRaNIE；
- LINGER；
- Dictys；
- scMTNI；
- Random control。

主要问题是：在同一批叶节点观测数据上，各方法能否恢复真实叶节点 TF-target GRN。

### 1.2 祖先节点 GRN 推断

祖先节点和 Root 没有 RNA/ATAC 观测。只有 RegArch 使用 fate map 和叶节点观测进行全局推断，因此祖先节点恢复作为 RegArch-only 分析报告。

### 1.3 消融实验

消融实验只改变 RegArch 的输入或先验，保持 RNA、ATAC、truth GRN、candidate edge universe 与评价指标不变。关键条件包括：

- Full：正确 fate map + ATAC prior；
- No fate map：关闭谱系先验，只保留叶节点 RNA/ATAC；
- Wrong fate map：使用拓扑错误的 fate map；
- Uniform root：将 root prior 改为均匀/密度先验；
- No ATAC：移除 ATAC prior，并使用 uniform root prior。

Wrong fate map 使用 `cross_clade_split` 构造：每个叶节点的直接父节点都被改变，并且每对真实兄弟叶被拆散；该过程只使用拓扑，不使用 GRN truth。

## 2. 评估指标

所有方法输出标准化为：

```text
method, cell_type, regulator, target, score, sign_optional
```

评估时将每个方法的输出补全到完整 edge universe。未报告边的 score 设为 0。主评价使用 unsigned score，即 `abs(score)`。

### 2.1 AUPRC

AUPRC 是主指标，用于衡量方法在高度稀疏网络中对真实边的排序能力。对于每个节点，将所有候选 TF-target pair 按 score 从高到低排序，以 true edge label 计算 average precision：

```math
\mathrm{AUPRC}
= \sum_k
\left(
R_k - R_{k-1}
\right) P_k.
```

其中 `P_k` 和 `R_k` 是第 `k` 个阈值下的 precision 与 recall。由于 GRN 很稀疏，AUPRC 比 AUROC 更能反映高排名边是否富集真实调控边。

### 2.2 AUROC

AUROC 衡量真实边与非边在全局排序中的可分性：

```math
\mathrm{AUROC}
= P(s_{positive} > s_{negative}).
```

AUROC 对类别不平衡相对不敏感，因此作为辅助指标。

### 2.3 Precision at true K / Recall at true K

对每个节点，令：

```math
K_v = \sum_e y_e^{(v)}
```

即该节点真实边数量。取 score 最高的 `K_v` 条边，计算：

```math
\mathrm{Precision@K}
= \frac{\mathrm{TP@K}}{K_v}.
```

由于 cutoff 正好等于真实边数，因此 Recall@K 与 Precision@K 数值相同。该指标反映如果研究者只保留与真实网络稀疏度相同数量的边，恢复了多少真实边。

### 2.4 Signed precision at true K

对于提供调控符号的方法，还评估 top-K 边中符号也正确的比例：

```math
\mathrm{SignedPrecision@K}
= \frac{
\#\{e \in \mathrm{TopK}: y_e=1,\ \hat s_e=s_e\}
}{K}.
```

RegArch 输出 signed posterior MAP state，因此可以评估符号。许多比较方法只提供 unsigned score 或相关性符号；因此 signed 指标用于补充分析，不作为所有方法排名的唯一依据。

### 2.5 Random control

Random control 在相同 edge universe 上随机排序。其理论 AUROC 约为 `0.5`，理论 AUPRC 约为 edge prevalence。本场景 random control 的观测结果为：

| 指标 | 观测均值 | 理论参考 |
|---|---:|---:|
| AUPRC | 0.1444 | 0.1097 |
| AUROC | 0.5110 | 0.5 |
| Precision@K | 0.1258 | 0.1097 |

有限样本和节点间真实边数差异可使 random AUPRC 略高于理论 prevalence。

## 3. 统计检验

### 3.1 为什么按 replicate 配对

同一模拟重复中的 10 个叶节点共享同一个 Root GRN 和同一套随机生成过程，因此叶节点不是完全独立样本。为避免夸大样本量，统计检验先在每个 replicate 内对叶节点指标取均值，然后在 10 个独立 replicate 上做配对检验。

### 3.2 RegArch 与其他方法比较

对每个 competitor 和每个指标，计算每个 replicate 的差值：

```math
d_r
= \bar m_{r,\mathrm{RegArch}}
- \bar m_{r,\mathrm{competitor}}.
```

使用 one-sided paired Wilcoxon signed-rank test 检验：

```math
H_1: \mathrm{median}(d_r)>0.
```

对同一指标下的多个 competitor 做 Holm correction。均值差的置信区间通过 5000 次 bootstrap 在 replicate 差值上计算。

### 3.3 消融实验

消融实验同样先按 replicate 聚合节点指标，再用 paired Wilcoxon test 比较 Full 与每个消融条件。叶节点、祖先节点和 Root 分别报告。

## 4. 叶节点 Benchmark 结果

### 4.1 主结果

| Rank | Method | AUPRC mean | AUROC mean | Precision@K |
|---:|---|---:|---:|---:|
| 1 | RegArch | 0.6791 | 0.9385 | 0.6008 |
| 2 | LINGER | 0.5819 | 0.9284 | 0.5358 |
| 3 | SCENIC | 0.5668 | 0.9309 | 0.5326 |
| 4 | SCENIC+ | 0.5551 | 0.9201 | 0.5078 |
| 5 | CellOracle | 0.5169 | 0.9010 | 0.4721 |
| 6 | GRaNIE | 0.4310 | 0.8936 | 0.3991 |
| 7 | FigR | 0.4010 | 0.8776 | 0.3714 |
| 8 | Pando | 0.3465 | 0.8239 | 0.3338 |
| 9 | Dictys | 0.3259 | 0.8398 | 0.2820 |
| 10 | scMTNI | 0.2875 | 0.7166 | 0.3527 |
| 11 | GENIE3 | 0.2062 | 0.6069 | 0.1902 |
| 12 | GRNBoost | 0.1605 | 0.5331 | 0.1419 |
| 13 | Random | 0.1444 | 0.5110 | 0.1258 |

RegArch 在 AUPRC、AUROC 和 Precision@K 上均为最高。AUPRC 相比最强 competitor LINGER 的均值差为 `0.0972`。

### 4.2 配对统计检验

RegArch 在 10/10 个独立 replicate 的 AUPRC 上均高于每个 competitor。对于 AUPRC，所有 competitor 的 Holm-adjusted one-sided Wilcoxon p-value 均为 `0.0117`，均满足预设优势判断。

部分 AUPRC 差值如下：

| Competitor | RegArch - competitor | 95% bootstrap CI |
|---|---:|---:|
| LINGER | 0.0972 | [0.0759, 0.1169] |
| SCENIC | 0.1123 | [0.0926, 0.1316] |
| SCENIC+ | 0.1239 | [0.1043, 0.1435] |
| CellOracle | 0.1622 | [0.1322, 0.1909] |
| Random | 0.5347 | [0.5105, 0.5606] |

这说明在该模拟场景下，RegArch 对叶节点 GRN 的排序恢复能力显著优于 RNA-only 方法、多组学方法和动态/树模型相关方法。

## 5. 祖先节点恢复

RegArch 的全局 signed CTMC posterior 在所有节点上具有相同解释：score 绝对值表示 edge-presence posterior probability，符号表示 activation/repression 的 MAP state。因此叶节点与祖先节点可在同一指标下比较。

| Node group | Node runs | AUPRC mean | AUROC mean | Signed Precision@K |
|---|---:|---:|---:|---:|
| Root | 10 | 0.5841 | 0.9246 | 0.4596 |
| Internal time 0.4 | 20 | 0.6253 | 0.9339 | 0.5138 |
| Internal time 0.6 | 10 | 0.6674 | 0.9483 | 0.5693 |
| Internal time 0.8 | 50 | 0.6451 | 0.9404 | 0.5537 |
| Leaves time 1.0 | 100 | 0.6791 | 0.9385 | 0.5619 |

在该 higher-turnover 模拟中，Root 的恢复准确率低于叶节点，符合生物学直觉：Root 距离所有观测叶节点较远，祖先状态必须从多个后代网络反推，边的丢失、获得和符号改变都会累积不确定性。

内部节点整体 AUPRC 为 `0.6364`，明显高于 random ancestor baseline，同时略低于叶节点。这说明 RegArch 能在没有祖先观测数据的情况下恢复相当一部分历史 GRN，但祖先节点推断仍比直接观测叶节点更困难。

## 6. 消融实验结果

### 6.1 主消融表

| Condition | Leaf AUPRC | Ancestor AUPRC | Leaf AUROC |
|---|---:|---:|---:|
| Full | 0.6791 | 0.6364 | 0.9385 |
| No fate map | 0.6371 | NA | 0.8949 |
| Wrong fate map | 0.6361 | 0.6129 (Root only) | 0.9288 |
| Uniform root | 0.6496 | 0.6200 | 0.8757 |
| No ATAC | 0.2757 | 0.2547 | 0.6302 |

注意：`Wrong fate map` 的 ancestor AUPRC 只包含 Root。错误拓扑下的内部节点混合了真实跨分支叶节点，没有唯一的真实祖先身份，因此不能与 Full 的 P1-P8 内部节点逐一比较。该数值应解读为 Root-only diagnostic，而不是“wrong fate map 在祖先节点整体优于 Full”。

### 6.2 正确 fate map 的贡献

Full 相比 No fate map 的叶节点 AUPRC 提升为：

```text
0.6791 - 0.6371 = 0.0419
```

配对检验结果：

- independent replicates：10；
- Full win rate：0.8；
- one-sided Wilcoxon p = `0.00684`；
- Holm-adjusted p = `0.00684`。

这说明正确 fate map 在该场景中提供了稳定且统计显著的增益。其生物学含义是：当真实 GRN 沿谱系平滑演化时，共享祖先结构可帮助模型在叶节点之间借用信息，提高叶节点 GRN 恢复。

### 6.3 错误 fate map 的影响

Full 相比 Wrong fate map 的叶节点 AUPRC 提升为：

```text
0.6791 - 0.6361 = 0.0430
```

配对检验结果：

- independent replicates：10；
- Full win rate：1.0；
- one-sided Wilcoxon p = `0.00098`；
- Holm-adjusted p = `0.00391`。

这说明错误 fate map 会消除正确 fate map 带来的收益。与上一版 10 叶节点模拟相比，本场景通过缩短末端枝和提高跨分支分化，使 wrong fate map 的负面影响更清晰。

Root-only diagnostic 中，Wrong fate map 的 Root AUPRC 为 `0.6129`，略高于 Full 的 `0.5841`；但这是未观测 Root 在不同平滑路径下的非显著差异，不能解释为错误谱系更好。配对检验方向为 Full > condition，该比较的 one-sided p = `0.9902` 只说明没有证据支持 Full 在 Root-only 指标上更高；它不是 wrong > Full 的显著性检验。主结论应以叶节点为准：Full 在 AUPRC、AUROC、Precision@K 和 signed Precision@K 上均高于 Wrong fate map。

但 Wrong fate map 与 No fate map 的直接差异不显著：

```text
No fate - Wrong fate = 0.0010
two-sided Wilcoxon p = 0.7695
```

因此应谨慎表述：当前结果支持 RegArch 对一定程度错误 fate map 具有鲁棒性，即错误谱系会使模型近似退化为无谱系情形，而不是显著低于无谱系基线。但不能据此断言所有错误 fate map 都不会产生反效果；在更高 lineage weight、更弱 RNA/ATAC 或更极端错误拓扑下，错误谱系仍可能伤害推断。

### 6.4 ATAC prior 的贡献

No ATAC 的叶节点 AUPRC 为 `0.2757`，相比 Full 下降：

```text
0.6791 - 0.2757 = 0.4034
```

配对检验：

- Full win rate：1.0；
- one-sided Wilcoxon p = `0.00098`；
- Holm-adjusted p = `0.00391`。

这说明 ATAC-derived prior 是当前叶节点 GRN 恢复的主要信息来源。生物学上，染色质可及性、motif 和 peak-gene links 共同约束了哪些 TF-target pair 更可能存在；在稀疏网络恢复问题中，这种 cis-regulatory prior 对候选边排序非常关键。

### 6.5 Root prior 的贡献

Uniform root 将 edge-specific root prior 替换为基于整体密度的均匀先验。叶节点 AUPRC 为 `0.6496`，相比 Full 下降 `0.0294`，Holm-adjusted p = `0.00391`。

这说明从叶节点 ATAC prior 汇总得到的 edge-specific root prior 也有贡献，但贡献小于直接移除 ATAC。

## 7. 生物学与算法结论

### 7.1 RegArch 的优势来源

本场景支持以下结论：

1. RegArch 在叶节点 GRN 恢复上优于 11 个比较方法和 random control。
2. 主要优势来自多组学先验、表达似然和谱系相关性的联合建模，而不是任一单一信息源。
3. ATAC-derived prior 贡献最大；移除 ATAC 后性能大幅下降。
4. 正确 fate map 提供额外、统计显著的增益。
5. 错误 fate map 会消除正确谱系的收益，但在当前设置下未显著低于 no fate map，说明模型对中等拓扑错误具有一定鲁棒性。

### 7.2 对祖先 GRN 的意义

其他比较方法只使用叶节点观测数据，无法自然输出 Root 或内部祖先节点 GRN。RegArch 通过 signed CTMC fate-map prior 将叶节点观测信息传播到整棵树上，从而可以推断未观测历史细胞状态的 GRN。

本场景中，祖先节点 AUPRC 为 `0.6364`，Root AUPRC 为 `0.5841`，均显著高于 random baseline。由于 Root 与叶节点距离最大，其准确率低于叶节点是合理的，也符合“历史越久远，不确定性越高”的生物学直觉。

### 7.3 不应夸大的地方

该模拟不应被解读为：

- RegArch 在所有真实数据场景中必然排名第一；
- 错误 fate map 永远不会伤害推断；
- synthetic-resource adapter 完全等价于各外部工具在真实基因组注释下的最佳流程；
- 祖先节点真值在真实数据中可直接观测。

更准确的结论是：在一个显式控制真值、具有谱系相关 GRN rewiring、paired RNA/ATAC 观测和统一 peak-level regulatory prior 的模拟系统中，RegArch 能有效整合表达、染色质可及性和 fate map 信息，在叶节点 GRN 恢复和未观测祖先 GRN 推断上表现出优势。

