#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse)
})

option_list <- list(
  make_option("--cibersort", type="character", help="Path to official CIBERSORT.R"),
  make_option("--signature", type="character", help="Signature matrix CSV (genes as columns)"),
  make_option("--mixture", type="character", help="Bulk CSV (samples x genes)"),
  make_option("--out", type="character", help="Output CSV of predicted props")
)
opt <- parse_args(OptionParser(option_list=option_list))

if (is.null(opt$cibersort) || !file.exists(opt$cibersort)) stop("Provide --cibersort path to CIBERSORT.R")

# CIBERSORT expects mixture with genes as rownames and signature as genes x celltypes in some releases.
# Here we transpose mixture accordingly and source the official script.

sig <- read.csv(opt$signature, check.names = FALSE, row.names = 1)
mix <- read.csv(opt$mixture, check.names = FALSE)
mix_t <- t(as.matrix(mix))
colnames(mix_t) <- rownames(mix_t) <- NULL

source(opt$cibersort)
# Assuming CIBERSORT(sig_matrix, mixture_matrix) returns a data.frame with proportions per cell type
res <- CIBERSORT(as.matrix(sig), mix_t)
write.csv(res, opt$out, row.names = FALSE)

