
# install the package
# source("http://bioconductor.org/biocLite.R")
# biocLite("rhdf5")
# install.packages("devtools")
# install.packages("dplyr")
# devtools::install_github("pachterlab/sleuth", force=TRUE)
library("sleuth")

# other libraries
install.packages("cowplot")
library('cowplot')
library("reshape2")
library(dplyr)

# get paths to kallisto results
base_dir <- "/n/home07/kmattioli/valor/expression/jeff_diff/03__kallisto_with_gene_isoform"
sample_ids <- dir(file.path(base_dir, "kallisto_results"))
kal_dirs <- sapply(sample_ids, function(id) file.path(base_dir, "kallisto_results", id))
kal_dirs

# load sample table
s2c <- read.table(file.path(base_dir, "sleuth_files/sample_info.txt"), header = TRUE, stringsAsFactors=FALSE)
s2c <- dplyr::select(s2c, sample = sample, condition)

# sort sample table alphabetically
attach(s2c)
s2c <- s2c[order(sample), ]
detach(s2c)

# add a column with path information to kallisto directories
s2c <- dplyr::mutate(s2c, path = kal_dirs)

# make sure the directories match the sample ids!
s2c

# subset s2c for pairwise comparisons that we care about!
cond_hESC <- which(s2c$condition == "hESC")
cond_endo <- which(s2c$condition == "endo")
cond_meso <- which(s2c$condition == "meso")
s2c_hESC_vs_endo <- s2c[c(cond_hESC,cond_endo),]
s2c_hESC_vs_meso <- s2c[c(cond_hESC,cond_meso),]

#########################
#    all together       #
#########################

# construct sleuth object
so <- sleuth_prep(s2c, ~ condition, read_bootstrap_tpm = TRUE)

# fit the full model
so <- sleuth_fit(so)

# fit the reduced model
so <- sleuth_fit(so, ~1, 'reduced')

# perform the likelihood ratio test of the null (reduced) model vs alternate (full) model
so <- sleuth_lrt(so, 'reduced', 'full')

# load results of this test
reduced_full <- sleuth_results(so, 'reduced:full', 'lrt')

# fix the annoying name stuff that happens with gencode
reduced_full = transform(reduced_full, id = colsplit(reduced_full$target_id, "\\|", names = c("transcript_id", "gene_id", "havana_gene", "havana_transcript", "transcript_name", "gene_name", "unclear", "biotype")))
reduced_full$id.biotype <- gsub('\\|', '', reduced_full$id.biotype)

# find n significant
level <- 0.05
sig <- dplyr::filter(reduced_full, qval < level)

# plot pca
plot_pca(so, text_labels=TRUE, color_by="condition")

# plot sample heatmap
plot_sample_heatmap(so)

# get df instead of object
sm <- sleuth_to_matrix(so, "obs_norm", "tpm")
df <- data.frame(sm)
write.table(df, file=file.path(base_dir, "sleuth_results/sleuth_abundances_norm_tpm.TRANSCRIPTS.txt"), quote=FALSE, sep="\t")

# get df instead of object - raw data this time
sm_raw <- sleuth_to_matrix(so, "obs_raw", "est_counts")
df_raw <- data.frame(sm_raw)
write.table(df, file=file.path(base_dir, "sleuth_results/sleuth_abundances_raw_counts.TRANSCRIPTS.txt"), quote=FALSE, sep="\t")

#########################
#    hESC vs. endo      #
#########################

# construct sleuth object
so_hESC_vs_endo <- sleuth_prep(s2c_hESC_vs_endo, ~ condition)

# fit the model
so_hESC_vs_endo <- sleuth_fit(so_hESC_vs_endo)

# fit the reduced model
so_hESC_vs_endo <- sleuth_fit(so_hESC_vs_endo, ~1, 'reduced')

# lrt test
so_hESC_vs_endo <- sleuth_lrt(so_hESC_vs_endo, "reduced", "full")

results_table_hESC_vs_endo <- sleuth_results(so_hESC_vs_endo, 'reduced:full', test_type = 'lrt')
#sleuth_live(so_hESC_vs_endo)

write.table(results_table_hESC_vs_endo, file=file.path(base_dir, "sleuth_results/diff_hESC_vs_endo.TRANSCRIPTS.txt"), quote=FALSE, sep="\t")

#########################
#    hESC vs. meso      #
#########################

# construct sleuth object
so_hESC_vs_meso <- sleuth_prep(s2c_hESC_vs_meso, ~ condition)

# fit the model
so_hESC_vs_meso <- sleuth_fit(so_hESC_vs_meso)

# fit the reduced model
so_hESC_vs_meso <- sleuth_fit(so_hESC_vs_meso, ~1, 'reduced')

# lrt test
so_hESC_vs_meso <- sleuth_lrt(so_hESC_vs_meso, "reduced", "full")

results_table_hESC_vs_meso <- sleuth_results(so_hESC_vs_meso, 'reduced:full', test_type = 'lrt')
#sleuth_live(so_hESC_vs_meso)

write.table(results_table_hESC_vs_meso, file=file.path(base_dir, "sleuth_results/diff_hESC_vs_meso.TRANSCRIPTS.txt"), quote=FALSE, sep="\t")
