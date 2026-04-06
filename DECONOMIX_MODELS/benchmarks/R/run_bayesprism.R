#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse)
  library(SingleCellExperiment)
  library(zellkonverter)
  library(BayesPrism)
})

option_list <- list(
  make_option("--h5ad", type="character", help="Path to scRNA h5ad"),
  make_option("--bulk", type="character", help="Bulk CSV (samples x genes)"),
  make_option("--genes", type="character", help="Pipe-separated gene symbols"),
  make_option("--out", type="character", help="Output CSV of predicted props")
)
opt <- parse_args(OptionParser(option_list=option_list))

message("[BayesPrism] Loading scRNA from ", opt$h5ad)
sce <- readH5AD(opt$h5ad)
if (!"cell_type" %in% colnames(colData(sce))) stop("cell_type column not found in scRNA obs")

genes <- strsplit(opt$genes, "\\|")[[1]]

ref_mat <- as.matrix(counts(sce))
ref_genes <- rownames(ref_mat)
keep <- intersect(ref_genes, genes)
ref_mat <- ref_mat[keep, , drop=FALSE]
ct <- as.character(colData(sce)$cell_type)

message("[BayesPrism] Loading bulk from ", opt$bulk)
bulk_df <- read.csv(opt$bulk, check.names = FALSE)
bulk_mat <- t(as.matrix(bulk_df)) # genes x samples
rownames(bulk_mat) <- colnames(bulk_df)
bulk_mat <- bulk_mat[intersect(rownames(bulk_mat), keep), , drop=FALSE]

# Construct BayesPrism object
bp <- new_BayesPrism(reference = ref_mat, input.type = "count.mat", cell.type.label = ct)
bp <- run.bayesprism(bp, mixture = bulk_mat)
props <- t(bp@post.props$final_gibbs) # samples x celltypes
write.csv(props, opt$out, row.names = FALSE)
message("[BayesPrism] Saved ", opt$out)

