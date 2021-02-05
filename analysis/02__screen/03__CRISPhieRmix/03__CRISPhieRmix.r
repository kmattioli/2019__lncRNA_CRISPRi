
#install.packages("devtools")

#devtools::install_github("timydaley/CRISPhieRmix")

library(CRISPhieRmix)

data_f = "../../../data/02__screen/02__enrichment_data/data_filt.with_batch.tmp"

data <- read.table(data_f, sep="\t", header=TRUE)
head(data)

names(data)

# x needs to be the l2fcs of target genes
tgts <- data[which(data$ctrl_status != "scramble"), ]
x <- tgts$l2fc_diff

head(x)

# geneIds needs to be the geneIds associated with x
geneIds <- tgts$group_id_rd
head(geneIds)

# negCtrl needs to be the l2fcs of scrambles
negs <- data[which(data$ctrl_status == "scramble"), ]
negCtrl <- negs$l2fc_diff

head(negCtrl)

l2fc_diff.CRISPhieRmix <- CRISPhieRmix(x = x, geneIds = geneIds, negCtrl = negCtrl,
                                       VERBOSE = TRUE, PLOT = TRUE)

hist(l2fc_diff.CRISPhieRmix$FDR, breaks = 100)

sum(l2fc_diff.CRISPhieRmix$FDR < 0.1)

sum(l2fc_diff.CRISPhieRmix$FDR < 0.05)

scores_diff <- data.frame(groups = l2fc_diff.CRISPhieRmix$genes, FDR = l2fc_diff.CRISPhieRmix$FDR)
head(scores_diff[order(scores_diff$FDR, decreasing = FALSE), ], 10)

sig_diff <- scores_diff[which(scores_diff$FDR < 0.1), ]
nrow(sig_diff)

sig_diff[grep("control", sig_diff$groups), ]

scores_diff[grep("DIGIT|FOXA2|SOX17", scores_diff$groups), ]

nrow(scores_diff)

write.table(scores_diff, file = "../../../data/02__screen/02__enrichment_data/CRISPhieRmix_diff.with_batch.txt", 
            quote = FALSE, sep = "\t", row.names = FALSE)

data_f = "../../../data/02__screen/02__enrichment_data/data_filt_dz.with_batch.tmp"

data <- read.table(data_f, sep="\t", header=TRUE)
head(data)

nrow(data)

# x needs to be the l2fcs of target genes
tgts <- data[which(data$ctrl_status != "scramble"), ]
x <- tgts$l2fc_dz
head(x)

# geneIds needs to be the geneIds associated with x
geneIds <- tgts$group_id_rd
head(geneIds)

# negCtrl needs to be the l2fcs of scrambles
negCtrl <- negs$l2fc_dz
head(negCtrl)

l2fc_dz.CRISPhieRmix <- CRISPhieRmix(x = x, geneIds = geneIds, negCtrl = negCtrl,
                                     VERBOSE = TRUE, PLOT = TRUE)

hist(l2fc_dz.CRISPhieRmix$FDR, breaks = 100)

sum(l2fc_dz.CRISPhieRmix$FDR < 0.1)

sum(l2fc_dz.CRISPhieRmix$FDR < 0.05)

scores_dz <- data.frame(groups = l2fc_dz.CRISPhieRmix$genes, FDR = l2fc_dz.CRISPhieRmix$FDR)
head(scores_dz[order(scores_dz$FDR, decreasing = FALSE), ], 10)

sig_dz <- scores_dz[which(scores_dz$FDR < 0.1), ]
nrow(sig_dz)

sig_dz[grep("control", sig_dz$groups), ]

scores_dz[grep("DIGIT", scores_dz$groups), ]

nrow(scores_dz)

write.table(scores_dz, file = "../../../data/02__screen/02__enrichment_data/CRISPhieRmix_dz.with_batch.txt", 
            quote = FALSE, sep = "\t", row.names = FALSE)
