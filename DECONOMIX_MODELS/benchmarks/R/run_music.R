#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse)
  library(SingleCellExperiment)
  library(zellkonverter)
  library(MuSiC)
  library(Biobase)
})

option_list <- list(
  make_option("--h5ad", type="character", help="Path to scRNA h5ad"),
  make_option("--bulk", type="character", help="Bulk CSV (samples x genes)"),
  make_option("--genes", type="character", help="Pipe-separated gene symbols"),
  make_option("--out", type="character", help="Output CSV of predicted props")
)
opt <- parse_args(OptionParser(option_list=option_list))

message("[MuSiC] Loading scRNA from ", opt$h5ad)
sce <- readH5AD(opt$h5ad)
if (!"cell_type" %in% colnames(colData(sce))) stop("cell_type column not found in scRNA obs")

genes <- strsplit(opt$genes, "\\|")[[1]]

# Build ExpressionSet for reference
ref_mat <- as.matrix(counts(sce))
ref_genes <- rownames(ref_mat)
keep <- intersect(ref_genes, genes)
ref_mat <- ref_mat[keep, , drop=FALSE]
pheno <- data.frame(cell_type = as.character(colData(sce)$cell_type))
rownames(pheno) <- colnames(ref_mat)
ref_eset <- ExpressionSet(assayData = ref_mat, phenoData = AnnotatedDataFrame(pheno))

message("[MuSiC] Loading bulk from ", opt$bulk)
bulk_df <- read.csv(opt$bulk, check.names = FALSE)
bulk_mat <- t(as.matrix(bulk_df)) # genes x samples for MuSiC
rownames(bulk_mat) <- colnames(bulk_df)
bulk_eset <- ExpressionSet(assayData = bulk_mat)

message("[MuSiC] Running music_prop ...")
res <- music_prop(bulk.eset = bulk_eset, sc.eset = ref_eset, clusters = "cell_type", verbose = TRUE)
props <- res$Est.prop.weighted
props <- t(props) # samples x celltypes
write.csv(props, opt$out, row.names = FALSE)
message("[MuSiC] Saved ", opt$out)

