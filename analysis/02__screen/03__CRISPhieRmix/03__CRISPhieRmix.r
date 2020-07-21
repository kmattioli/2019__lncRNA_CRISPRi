#install.packages("devtools")

#devtools::install_github("timydaley/CRISPhieRmix")

library(CRISPhieRmix)

data_f = "../../../data/02__screen/02__enrichment_data/data_filt.tmp"

data <- read.table(data_f, sep="\t", header=TRUE)
head(data)

names(data)

# x needs to be the l2fcs of target genes
tgts <- data[which(data$ctrl_status != "scramble"), ]
x <- tgts$l2fc
head(x)

# geneIds needs to be the geneIds associated with x
geneIds <- tgts$group_id_rd
head(geneIds)

# negCtrl needs to be the l2fcs of scrambles
negs <- data[which(data$ctrl_statu == "scramble"), ]
negCtrl <- negs$l2fc
head(negCtrl)

l2fc.CRISPhieRmix <- CRISPhieRmix(x = x, geneIds = geneIds, negCtrl = negCtrl,
                                  VERBOSE = TRUE, PLOT = TRUE)

hist(l2fc.CRISPhieRmix$FDR, breaks = 100)

sum(l2fc.CRISPhieRmix$FDR < 0.1)

sum(l2fc.CRISPhieRmix$FDR < 0.05)

scores <- data.frame(groups = l2fc.CRISPhieRmix$genes, FDR = l2fc.CRISPhieRmix$FDR)
head(scores[order(scores$FDR, decreasing = FALSE), ], 80)

sig <- scores[which(scores$FDR < 0.1), ]
nrow(sig)

sig[grep("control", sig$groups), ]

scores[grep("DIGIT|FOXA2|SOX17", scores$groups), ]

nrow(scores)

scores[grep("FOXD3-AS1", scores$groups), ]

write.table(scores, file = "../../../data/02__screen/02__enrichment_data/CRISPhieRmix.txt", quote = FALSE, 
            sep = "\t", row.names = FALSE)


