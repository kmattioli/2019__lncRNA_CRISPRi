
# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")

# BiocManager::install("DESeq2")

# install.packages("gridExtra")

suppressMessages(library("DESeq2"))

library("ggplot2")
library("gridExtra")
library("gtable")
library("cowplot")

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

fc_dz_cts <- cts[, c(1,2,3,4)]
head(fc_dz_cts)

fc_cols <- cols[3:6, ]
fc_cols <- fc_cols[c(3,4,1,2), ]
fc_cols

fc_dz_cols <- cols[0:4, ]
fc_dz_cols

dds <- DESeqDataSetFromMatrix(countData = cts,
                              colData = cols,
                              design = ~ condition + rep)

dds <- estimateSizeFactors(dds)

norm_cts <- counts(dds, normalized=TRUE)
head(norm_cts)

vsd <- vst(dds, blind=FALSE)
head(assay(vsd))

g1 <- plotPCA(vsd, ntop=100, intgroup=c("condition", "rep"))

mat <- assay(vsd)
mat <- limma::removeBatchEffect(mat, vsd$rep)
assay(vsd) <- mat

g2 <- plotPCA(vsd, ntop=100, intgroup=c("condition", "rep"))

g1grob <- ggplotGrob(g1)
g2grob <- ggplotGrob(g2)

pdf("FigS5E.pdf", height = 4, width = 6)
grid::grid.draw(cbind(g1grob, g2grob, size = "first"))
dev.off()

dds <- estimateSizeFactors(dds)

dds <- DESeqDataSetFromMatrix(countData = fc_cts,
                              colData = fc_cols,
                              design = ~ rep + condition)

dds <- DESeq(dds, betaPrior=TRUE)

res <- results(dds, addMLE=TRUE)
head(res)

write.table(res, file = "../../../data/02__screen/01__normalized_counts/l2fcs.DESeq2.with_batch.txt", 
            sep = "\t", quote = FALSE)

dds_dz <- DESeqDataSetFromMatrix(countData = fc_dz_cts,
                              colData = fc_dz_cols,
                              design = ~ rep + condition)

dds_dz <- estimateSizeFactors(dds_dz)

dds_dz <- DESeq(dds_dz, betaPrior=TRUE)

res_dz <- results(dds_dz, addMLE=TRUE)
head(res_dz)

write.table(res_dz, file = "../../../data/02__screen/01__normalized_counts/l2fcs_DZ.DESeq2.with_batch.txt", 
            sep = "\t", quote = FALSE)
