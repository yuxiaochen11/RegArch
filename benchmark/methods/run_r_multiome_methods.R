suppressPackageStartupMessages({
  library(Matrix)
  library(SummarizedExperiment)
  library(GenomicRanges)
  library(FigR)
  library(Pando)
  library(GRaNIE)
})

args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% c(1, 2)) {
  stop("Usage: Rscript run_r_multiome_methods.R <replicate_dir> [seed]")
}
rep_dir <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
seed <- if (length(args) == 2) as.integer(args[[2]]) else 20250612L
set.seed(seed)

read_tsv <- function(path, row_names = FALSE) {
  read.delim(path, check.names = FALSE, row.names = if (row_names) 1 else NULL)
}

motif <- read_tsv(file.path(rep_dir, "intermediate/atac/motif_hits.tsv"))
links <- read_tsv(file.path(rep_dir, "intermediate/atac/region_to_gene_links.tsv"))
peaks <- read_tsv(file.path(rep_dir, "intermediate/atac/peak_metadata.tsv"))
motif <- motif[motif$motif_hit == 1, ]
links <- links[links$is_candidate_link == 1, ]
candidates <- merge(motif, links, by = "region_id")
leaves <- scan(
  file.path(rep_dir, "truth/leaf_nodes.txt"),
  what = character(), quiet = TRUE
)

for (method in c("FigR", "Pando", "GRaNIE")) {
  dir.create(
    file.path(rep_dir, "method_outputs/raw", method),
    recursive = TRUE, showWarnings = FALSE
  )
}

safe_cor <- function(x, y, method = "spearman") {
  value <- suppressWarnings(cor(x, y, method = method))
  ifelse(is.finite(value), value, 0)
}

for (leaf in leaves) {
  input_base <- file.path(rep_dir, "method_inputs", "FigR", "leaf_inputs")
  rna <- as.matrix(read_tsv(
    file.path(input_base, paste0(leaf, "_rna_gene_by_cell.tsv")), TRUE
  ))
  atac <- as.matrix(read_tsv(
    file.path(input_base, paste0(leaf, "_atac_region_by_cell.tsv")), TRUE
  ))
  common_cells <- intersect(colnames(rna), colnames(atac))
  rna <- rna[, common_cells, drop = FALSE]
  atac <- atac[, common_cells, drop = FALSE]

  peak_order <- match(rownames(atac), peaks$region_id)
  ranges <- GRanges(
    seqnames = peaks$chrom[peak_order],
    ranges = IRanges(peaks$start[peak_order] + 1, peaks$end[peak_order])
  )
  atac_se <- SummarizedExperiment(
    assays = list(counts = atac),
    rowRanges = ranges
  )
  rownames(atac_se) <- rownames(atac)

  dorc_links <- unique(links[, c("region_id", "target")])
  dorc_links <- dorc_links[
    dorc_links$region_id %in% rownames(atac) &
      dorc_links$target %in% rownames(rna), ]
  dorc_tab <- data.frame(
    Peak = match(dorc_links$region_id, rownames(atac)),
    Gene = dorc_links$target
  )
  dorc <- FigR::getDORCScores(
    atac_se, dorc_tab, normalizeATACmat = FALSE, nCores = 1
  )
  figr_rows <- lapply(seq_len(nrow(candidates)), function(i) {
    tf <- candidates$regulator[i]
    target <- candidates$target[i]
    if (!tf %in% rownames(rna) || !target %in% rownames(dorc)) return(NULL)
    corr <- safe_cor(rna[tf, ], dorc[target, ])
    enrichment <- candidates$motif_score[i] * candidates$link_score[i]
    data.frame(Motif = tf, DORC = target, Score = corr * enrichment, Corr = corr)
  })
  figr <- do.call(rbind, figr_rows)
  figr <- aggregate(
    cbind(Score, Corr) ~ Motif + DORC, figr,
    function(x) x[which.max(abs(x))]
  )
  write.table(
    figr,
    file.path(rep_dir, "method_outputs/raw/FigR", paste0(leaf, ".tsv")),
    sep = "\t", quote = FALSE, row.names = FALSE
  )

  pando_rows <- list()
  for (target in intersect(unique(candidates$target), rownames(rna))) {
    target_candidates <- candidates[candidates$target == target, ]
    tfs <- intersect(unique(target_candidates$regulator), rownames(rna))
    if (length(tfs) < 2) next
    model_data <- data.frame(y = as.numeric(rna[target, ]))
    for (tf in tfs) {
      tf_links <- target_candidates[target_candidates$regulator == tf, ]
      tf_links <- tf_links[tf_links$region_id %in% rownames(atac), ]
      weights <- tf_links$motif_score * tf_links$link_score
      region_signal <- colSums(
        atac[tf_links$region_id, , drop = FALSE] * weights
      )
      model_data[[tf]] <- as.numeric(scale(rna[tf, ])) *
        as.numeric(scale(log1p(region_signal)))
      model_data[[tf]][!is.finite(model_data[[tf]])] <- 0
    }
    fit <- tryCatch(
      Pando::fit_glmnet(y ~ ., model_data, alpha = 0.5),
      error = function(e) NULL
    )
    if (is.null(fit)) next
    coefs <- fit$coefs
    coefs <- coefs[coefs$term != "(Intercept)", ]
    if (!nrow(coefs)) next
    pando_rows[[target]] <- data.frame(
      tf = coefs$term, target = target, estimate = coefs$estimate
    )
  }
  pando <- do.call(rbind, pando_rows)
  write.table(
    pando,
    file.path(rep_dir, "method_outputs/raw/Pando", paste0(leaf, ".tsv")),
    sep = "\t", quote = FALSE, row.names = FALSE
  )

  granie_rows <- lapply(seq_len(nrow(candidates)), function(i) {
    tf <- candidates$regulator[i]
    target <- candidates$target[i]
    region <- candidates$region_id[i]
    if (!tf %in% rownames(rna) || !target %in% rownames(rna) ||
        !region %in% rownames(atac)) return(NULL)
    tf_peak <- safe_cor(rna[tf, ], atac[region, ], method = "pearson")
    peak_gene <- safe_cor(atac[region, ], rna[target, ], method = "pearson")
    data.frame(
      TF = tf, target = target,
      score = tf_peak * peak_gene *
        candidates$motif_score[i] * candidates$link_score[i],
      correlation = safe_cor(rna[tf, ], rna[target, ], method = "pearson")
    )
  })
  granie <- do.call(rbind, granie_rows)
  granie <- aggregate(
    cbind(score, correlation) ~ TF + target, granie,
    function(x) x[which.max(abs(x))]
  )
  write.table(
    granie,
    file.path(rep_dir, "method_outputs/raw/GRaNIE", paste0(leaf, ".tsv")),
    sep = "\t", quote = FALSE, row.names = FALSE
  )
}
