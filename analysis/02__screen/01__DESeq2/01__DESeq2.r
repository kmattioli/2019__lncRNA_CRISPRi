# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")

# BiocManager::install("DESeq2")

suppressMessages(library("DESeq2"))

cts <- read.csv("../../../data/02__screen/00__counts/Biol_Reps.sgRNA_counts.txt", sep="\t", row.names="sgRNA")
cts <- as.matrix(cts)
head(cts)

cols <- read.csv("../../../data/02__screen/01__normalized_counts/col_info.txt", sep="\t", row.names="column")
cols$condition <- as.factor(cols$condition)
cols$rep <- as.factor(cols$rep)
cols$time <- as.factor(cols$time)
head(cols)

fc_cts <- cts[, c(5,6,3,4)]
head(fc_cts)

fc_cts_rep1 <- cts[, c(5,3)]
head(fc_cts_rep1)

fc_cts_rep2 <- cts[, c(6,4)]
head(fc_cts_rep2)

fc_cols <- cols[3:6, ]
fc_cols <- fc_cols[c(3,4,1,2), ]
fc_cols

fc_cols_rep1 <- fc_cols[c(1,3), ]
fc_cols_rep1

fc_cols_rep2 <- fc_cols[c(2,4), ]
fc_cols_rep2

dds <- DESeqDataSetFromMatrix(countData = cts,
                              colData = cols,
                              design = ~ condition)

dds <- estimateSizeFactors(dds)

norm_cts <- counts(dds, normalized=TRUE)
head(norm_cts)

rld <- rlog(dds)
head(assay(rld))

vsd <- vst(dds, blind=FALSE)
head(assay(vsd))

dds <- estimateSizeFactors(dds)

dds <- DESeqDataSetFromMatrix(countData = fc_cts,
                              colData = fc_cols,
                              design = ~ condition)

dds <- DESeq(dds)

res <- results(dds)
head(res)

plotMA(res)

write.table(res, file = "../../../data/02__screen/01__normalized_counts/l2fcs.DESeq2.txt", 
            sep = "\t", quote = FALSE)


