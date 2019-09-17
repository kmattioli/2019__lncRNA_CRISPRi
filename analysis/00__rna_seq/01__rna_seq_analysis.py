
# coding: utf-8

# # 01__rna_seq_analysis

# in this notebook, i look at lncRNA + mRNA expression across the 3 lineages from our RNA-seq data. note that in our RNA-seq data, we use GENCODE v25 (mapped to hg19) as the transcriptome assembly.
# 
# figures in this notebook:
# - Fig 2A: heatmap of expression across samples for lncRNAs and mRNAs
# - Fig 2B: expression of marker genes across lineages
# - Fig 2C: GO term enrichments
# - Fig 2D: count of expressed lncRNAs by biotype
# - Fig 2E: volcano plots of lncRNA expression in endo and meso compared to hESCs
# - Fig 2F: tissue-specificity of lncRNAs compared to all mRNAs and only transcription factors
# - Fig S3A: heatmap showing correlation of abundances across RNA-seq samples
# - Fig S3B: overview of expression profiles for lncRNAs
# 
# supplemental tables in this notebook:
# - Supp Table S1: RNA-seq results for lncRNAs

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import gseapy
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import re
import scipy.stats as stats
import seaborn as sns
import sys
import time

from scipy.stats import spearmanr
from scipy.stats import pearsonr
from statsmodels.sandbox.stats import multicomp

# import utils
sys.path.append("../../utils")
from plotting_utils import *
from classify_utils import *

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
mpl.rcParams['figure.autolayout'] = False


# In[2]:


sns.set(**PAPER_PRESET)
fontsize = PAPER_FONTSIZE


# ### functions

# In[3]:


def is_significant(row, colname, alpha):
    qval = row[colname]
    if qval == "sleuth NA":
        return qval
    else:
        if qval < alpha:
            return "significant"
        else:
            return "not significant"


# In[4]:


def mark_for_volcano(row, exp_col, sig_col):
    log2fc = row[exp_col]
    sig = row[sig_col]
    if sig == "significant":
        return "significant"
    else:
        return "not significant"


# ### variables

# In[5]:


sleuth_dir = "../../data/00__rna_seq/00__sleuth_results"


# In[6]:


expr_f = "%s/sleuth_abundances_norm_tpm.TRANSCRIPTS.txt" % (sleuth_dir)
diff_hESC_endo_f = "%s/diff_hESC_vs_endo.TRANSCRIPTS.txt" % (sleuth_dir)
diff_hESC_meso_f = "%s/diff_hESC_vs_meso.TRANSCRIPTS.txt" % (sleuth_dir)


# In[7]:


lncRNA_gene_biotype_map_f = "../../misc/00__gene_metadata/all_lncRNAs_classified.locus.buffer.with_name.txt"
lncRNA_tx_biotype_map_f = "../../misc/00__gene_metadata/all_lncRNAs_classified.txt"
all_biotype_map_f = "../../misc/00__gene_metadata/gencode.v25lift37.GENE_ID_TO_NAME_AND_BIOTYPE_MAP.txt"
transcript_id_map_f = "../../misc/00__gene_metadata/gencode.v25lift37.GENE_ID_TRANSCRIPT_ID_MAP.with_DIGIT.fixed.txt"


# In[8]:


tfs_f = "../../misc/01__curated_TF_list/curated_motif_map.txt"


# ### 1. import data

# In[9]:


expr = pd.read_table(expr_f, sep="\t").reset_index()
expr.columns = ["transcript_id", "endo_rep1", "endo_rep2", "hESC_rep1", "hESC_rep2", "meso_rep1", "meso_rep2"]


# In[10]:


expr.head()


# In[11]:


transcript_id_map = pd.read_table(transcript_id_map_f, sep="\t", header=None)
transcript_id_map.columns = ["gene_id", "transcript_id"]
transcript_id_map.head()


# In[12]:


print(len(expr))
expr = expr.merge(transcript_id_map, on="transcript_id")
print(len(expr))
expr.head()


# In[13]:


# annotation biotypes
ann_biotype_map = pd.read_table(all_biotype_map_f, sep="\t", header=None)
ann_biotype_map.columns = ["gene_id", "biotype", "gene_name"]
ann_biotype_map.head()


# In[14]:


# find transcript-level biotype
lncRNA_tx_biotype_map = pd.read_table(lncRNA_tx_biotype_map_f, sep="\t", header=None)
lncRNA_tx_biotype_map.columns = ["transcript_id", "csf_class", "biotype"]
lncRNA_tx_biotype_map.head()


# In[15]:


# find gene-level biotype
lncRNA_gene_biotype_map = pd.read_table(lncRNA_gene_biotype_map_f, sep="\t", header=None)
lncRNA_gene_biotype_map.columns = ["gene_name", "gene_id", "csf_class", "gene_biotype", "cage_peak", "strand"]
lncRNA_gene_biotype_map.head()


# ### merge all biotype information (transcript level, gene level) and fix DIGIT which was added manually

# In[16]:


# merge everything
expr_tmp = expr.merge(ann_biotype_map, on="gene_id", how="left", suffixes=("", "_ann"))
expr_tmp = expr_tmp.merge(lncRNA_tx_biotype_map, on="transcript_id", suffixes=("", "_lncRNA"), how="left")
expr_tmp = expr_tmp.merge(lncRNA_gene_biotype_map[["gene_id", "gene_biotype"]], on="gene_id", how="left")
expr_tmp.head()


# In[17]:


# fix digit
expr_tmp.loc[0, "biotype"] = "tss_overlapping__lncRNA"
expr_tmp.loc[0, "gene_name"] = "DIGIT"
expr_tmp.loc[0, "csf_class"] = "lncRNA_good_csf"
expr_tmp.loc[0, "biotype_lncRNA"] = "tss_overlapping__lncRNA"
expr_tmp.loc[0, "gene_biotype"] = "tss_overlapping__lncRNA"
expr_tmp.head()


# In[18]:


expr_tmp["csf"] = expr_tmp.apply(get_csf_class, axis=1)
expr_tmp.head()


# In[19]:


expr_tmp["transcript_biotype"] = expr_tmp.apply(get_transcript_biotype, axis=1)


# In[20]:


expr_tmp["gene_biotype"] = expr_tmp.apply(get_gene_biotype, axis=1)


# In[21]:


expr_tmp.head()


# In[22]:


expr = expr_tmp[["transcript_id", "gene_id", "gene_name", "csf", "transcript_biotype", "gene_biotype",
                 "hESC_rep1", "hESC_rep2", "endo_rep1", "endo_rep2", "meso_rep1", "meso_rep2"]].drop_duplicates()


# In[23]:


len(expr)


# ### 2. find mean expression per gene across condition

# throughout this notebook, the 'expr' df has transcript values, whereas the 'expr_gene' df has gene values (summed transcript values)

# In[24]:


# also get gene expression values, not transcript expression values (for plotting)
expr_gene = expr.groupby(["gene_id", "gene_name", "csf", "gene_biotype"])[["hESC_rep1", "hESC_rep2", "endo_rep1", 
                                                                           "endo_rep2", "meso_rep1", "meso_rep2"]].agg("sum").reset_index()
print(len(expr_gene))
expr_gene.head()


# In[25]:


expr["hESC_mean"] = expr[["hESC_rep1", "hESC_rep2"]].mean(axis=1)
expr["hESC_std"] = expr[["hESC_rep1", "hESC_rep2"]].std(axis=1)

expr["endo_mean"] = expr[["endo_rep1", "endo_rep2"]].mean(axis=1)
expr["endo_std"] = expr[["endo_rep1", "endo_rep2"]].std(axis=1)

expr["meso_mean"] = expr[["meso_rep1", "meso_rep2"]].mean(axis=1)
expr["meso_std"] = expr[["meso_rep1", "meso_rep2"]].std(axis=1)

expr["overall_mean"] = expr[["hESC_mean", "endo_mean", "meso_mean"]].mean(axis=1)
expr.head()


# In[26]:


expr_gene["hESC_mean"] = expr_gene[["hESC_rep1", "hESC_rep2"]].mean(axis=1)
expr_gene["hESC_std"] = expr_gene[["hESC_rep1", "hESC_rep2"]].std(axis=1)

expr_gene["endo_mean"] = expr_gene[["endo_rep1", "endo_rep2"]].mean(axis=1)
expr_gene["endo_std"] = expr_gene[["endo_rep1", "endo_rep2"]].std(axis=1)

expr_gene["meso_mean"] = expr_gene[["meso_rep1", "meso_rep2"]].mean(axis=1)
expr_gene["meso_std"] = expr_gene[["meso_rep1", "meso_rep2"]].std(axis=1)

expr_gene["overall_mean"] = expr_gene[["hESC_mean", "endo_mean", "meso_mean"]].mean(axis=1)
expr_gene.head()


# ### 3. find transcripts & genes expressed at least at 0.1 tpm (in at least 1 condition)

# In[27]:


expr["threshold"] = expr.apply(get_expr_thresh, axis=1)
expr_gene["threshold"] = expr_gene.apply(get_expr_thresh, axis=1)
expr.head()


# In[28]:


expr.groupby(["csf", "threshold"])["transcript_id"].agg("count")


# In[29]:


expr_gene.groupby(["csf", "threshold"])["gene_id"].agg("count")


# In[30]:


expr["clean_gene_biotype"] = expr.apply(clean_biotype, col="gene_biotype", axis=1)
expr["clean_transcript_biotype"] = expr.apply(clean_biotype, col="transcript_biotype", axis=1)
expr_gene["clean_gene_biotype"] = expr_gene.apply(clean_biotype, col="gene_biotype", axis=1)


# ### 4. filter on good biotypes

# for now, keep all biotypes, but note if biotypes include tss_overlapping or divergent

# In[31]:


expr["cleaner_gene_biotype"] = expr.apply(cleaner_biotype, col="gene_biotype", axis=1)
expr["cleaner_transcript_biotype"] = expr.apply(cleaner_biotype, col="transcript_biotype", axis=1)


# In[32]:


expr_gene["cleaner_biotype"] = expr_gene.apply(cleaner_biotype, col="gene_biotype", axis=1)


# In[33]:


bad_biotypes = ["biotype not considered"]
bad_csf = ["lncRNA_bad_csf"]
expr_filt = expr[~expr["cleaner_gene_biotype"].isin(bad_biotypes)]
expr_filt = expr_filt[~expr_filt["csf"].isin(bad_csf)]


# In[34]:


expr_gene_filt = expr_gene[~expr_gene["cleaner_biotype"].isin(bad_biotypes)]
expr_gene_filt = expr_gene_filt[~expr_gene_filt["csf"].isin(bad_csf)]


# In[35]:


order = ["protein_coding", "intergenic", "promoter_overlap", "transcript_overlap", "gene_nearby"]
pal = {"protein_coding": sns.color_palette("deep")[0], "intergenic": "firebrick", "promoter_overlap": "firebrick",
       "transcript_overlap": "firebrick", "gene_nearby": "firebrick"}


# In[36]:


plt.figure(figsize=(1.6,1.75))
ax = sns.countplot(y="cleaner_biotype", data=expr_gene_filt[expr_gene_filt["threshold"] == "expressed"], 
                   palette=pal, order=order)

for p in ax.patches:
    w = p.get_width()
    y = p.get_y()
    h = p.get_height()
    
    ax.text(w + 100, y + h/2, int(w), ha="left", va="center", fontsize=fontsize) 

plt.xlim((0,23000))
plt.legend(loc=4)
plt.xlabel("count of genes expressed at > 0.1 tpm\nin at least 1 lineage")
plt.ylabel("")
ax.set_yticklabels(["protein-coding", "intergenic", "promoter overlap", "transcript overlap", "gene nearby"])
plt.savefig("Fig2D.pdf", dpi="figure", bbox_inches="tight")


# In[37]:


expr_filt["coding_type"] = expr_filt.apply(coding_type, axis=1)
expr_gene_filt["coding_type"] = expr_gene_filt.apply(coding_type, axis=1)


# In[38]:


expr_filt["hESC_thresh"] = expr_filt.apply(hESC_thresh, axis=1)
expr_filt["endo_thresh"] = expr_filt.apply(endo_thresh, axis=1)
expr_filt["meso_thresh"] = expr_filt.apply(meso_thresh, axis=1)
expr_filt.head()


# In[39]:


expr_gene_filt["hESC_thresh"] = expr_gene_filt.apply(hESC_thresh, axis=1)
expr_gene_filt["endo_thresh"] = expr_gene_filt.apply(endo_thresh, axis=1)
expr_gene_filt["meso_thresh"] = expr_gene_filt.apply(meso_thresh, axis=1)


# In[40]:


hESC_nc = set(expr_filt[(expr_filt["hESC_thresh"] == "expressed") & (expr_filt["coding_type"] == "non-coding")]["gene_name"])
endo_nc = set(expr_filt[(expr_filt["endo_thresh"] == "expressed") & (expr_filt["coding_type"] == "non-coding")]["gene_name"])
meso_nc = set(expr_filt[(expr_filt["meso_thresh"] == "expressed") & (expr_filt["coding_type"] == "non-coding")]["gene_name"])

all_nc = set.intersection(hESC_nc, endo_nc, meso_nc)
print("there are %s lncRNAs expressed in all lineages" % (len(all_nc)))

hESC_endo_nc = set.intersection(hESC_nc, endo_nc)
print("there are %s lncRNAs expressed in hESC and endo lineages" % (len(hESC_endo_nc)))

hESC_meso_nc = set.intersection(hESC_nc, meso_nc)
print("there are %s lncRNAs expressed in hESC and meso lineages" % (len(hESC_meso_nc)))

endo_meso_nc = set.intersection(endo_nc, meso_nc)
print("there are %s lncRNAs expressed in endo and meso lineages" % (len(endo_meso_nc)))


# In[41]:


expr_ncRNA_filt = expr_filt[expr_filt["coding_type"] == "non-coding"]
expr_ncRNA_filt["expr_profile"] = expr_ncRNA_filt.apply(get_expr_profile, axis=1)


# In[42]:


len(expr_ncRNA_filt[expr_ncRNA_filt["expr_profile"] != "not expressed"])


# In[43]:


# check for size of libraries right now
expr_ncRNA_gene_filt = expr_gene_filt[expr_gene_filt["coding_type"] == "non-coding"]
expr_ncRNA_gene_filt["expr_profile"] = expr_ncRNA_gene_filt.apply(get_expr_profile, axis=1)


# In[44]:


len(expr_ncRNA_gene_filt)


# In[45]:


fig = plt.figure(figsize=(2,2))
ax = sns.countplot(y="expr_profile", data=expr_ncRNA_gene_filt,
                   order=["hESC only", "endo only", "meso only", "hESC and endo", "endo and meso", "hESC and meso",
                          "hESC, endo, and meso", "not expressed"], color=sns.color_palette("Set2")[2])

for p in ax.patches:
    w = p.get_width()
    y = p.get_y()
    h = p.get_height()
    
    ax.text(w + 100, y + h/2, int(w), ha="left", va="center", fontsize=fontsize) 

plt.xlim((0,7500))
plt.title("count of lncRNA genes\nby expression profile")
plt.ylabel("")
fig.savefig("FigS3B.pdf", bbox_inches="tight", dpi="figure")


# ### 5. check correlations between replicates

# In[46]:


expr_reps = expr[["hESC_rep1", "hESC_rep2", "endo_rep1", "endo_rep2", "meso_rep1", "meso_rep2"]]
expr_reps_corr = expr_reps.corr(method="spearman")


# In[47]:


# cg = sns.clustermap(expr_reps_corr, annot=True, vmin=0.7, cmap=sns.cubehelix_palette(as_cmap=True), figsize=(2.25,2.25))
# plt.suptitle("spearman correlation of replicates (all transcript tpms, not filtered)")
# plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
# plt.savefig("replicate_corrs_spearman.pdf", bbox_inches="tight", dpi="figure")


# In[48]:


expr_gene_reps = expr_gene[["hESC_rep1", "hESC_rep2", "endo_rep1", "endo_rep2", "meso_rep1", "meso_rep2"]]
expr_gene_reps_corr = expr_gene_reps.corr(method="spearman")


# In[49]:


cg = sns.clustermap(expr_gene_reps_corr, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), figsize=(2.25,2.25))
plt.suptitle("spearman correlation of replicates (all gene tpms, not filtered)")
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.savefig("FigS3A.pdf", bbox_inches="tight", dpi="figure")


# In[50]:


expr_thresh = expr[expr["threshold"] == "expressed"]
expr_gene_thresh = expr_gene[expr_gene["threshold"] == "expressed"]


# In[51]:


# expr_reps = expr_thresh[["hESC_rep1", "hESC_rep2", "endo_rep1", "endo_rep2", "meso_rep1", "meso_rep2"]]
# expr_reps_corr = expr_reps.corr(method="spearman")

# cg = sns.clustermap(expr_reps_corr, annot=True, vmin=0.7, cmap=sns.cubehelix_palette(as_cmap=True), figsize=(2.25,2.25))
# plt.suptitle("spearman correlation of replicates\n(transcript tpms filtered >0.1 tpm in at least 1 lineage)")
# plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
# plt.savefig("replicate_corrs_spearman_filtered.pdf", bbox_inches="tight", dpi="figure")


# In[52]:


# expr_gene_reps = expr_gene_thresh[["hESC_rep1", "hESC_rep2", "endo_rep1", "endo_rep2", "meso_rep1", "meso_rep2"]]
# expr_gene_reps_corr = expr_gene_reps.corr(method="spearman")

# cg = sns.clustermap(expr_gene_reps_corr, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), figsize=(2.25,2.25))
# plt.suptitle("spearman correlation of replicates\n(gene tpms filtered >0.1 tpm in at least 1 lineage)")
# plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
# plt.savefig("replicate_corrs_spearman_filtered.genes.pdf", bbox_inches="tight", dpi="figure")


# ### 6. make some heatmaps

# In[53]:


expr_filt_ex = expr_filt[expr_filt["threshold"] == "expressed"]
print("there are %s transcripts expressed at at least 0.1 tpm in any of the 4 lineages" % (len(expr_filt_ex)))


# In[54]:


expr_gene_filt_ex = expr_gene_filt[expr_gene_filt["threshold"] == "expressed"]
print("there are %s genes expressed at at least 0.1 tpm in any of the 4 lineages" % (len(expr_gene_filt_ex)))


# In[55]:


expr_ncRNA = expr_filt[(expr_filt["cleaner_gene_biotype"] != "protein_coding") & (expr_filt["threshold"] == "expressed")]
print("there are %s ncRNA transcripts expressed at at least 0.1 tpm in any of the 4 lineages" % (len(expr_ncRNA)))


# In[56]:


expr_gene_ncRNA = expr_gene_filt[(expr_gene_filt["cleaner_biotype"] != "protein_coding") & (expr_gene_filt["threshold"] == "expressed")]
print("there are %s ncRNA genes expressed at at least 0.1 tpm in any of the 4 lineages" % (len(expr_gene_ncRNA)))


# In[57]:


cmap = sns.diverging_palette(220, 20, as_cmap=True)


# In[58]:


expr_ncRNA_heatmap = expr_gene_ncRNA[["hESC_rep1", "hESC_rep2", "endo_rep1", "endo_rep2", "meso_rep1", "meso_rep2"]]
g = sns.clustermap(expr_ncRNA_heatmap, yticklabels=False, figsize=(2, 3.5), cmap=cmap, metric="correlation",
                   **{"rasterized": True}, linewidths=0.0, z_score=0)
#g.ax_row_dendrogram.set_visible(False)
plt.suptitle("expression of all ncRNA genes\n[z-score plotted]")
plt.savefig("Fig2A_1.pdf", bbox_inches="tight", dpi="figure")


# In[59]:


expr_mRNA = expr_filt[(expr_filt["gene_biotype"] == "protein_coding") & (expr_filt["threshold"] == "expressed")]
print("there are %s protein-coding transcripts expressed at at least 0.1 tpm in any of the 4 lineages" % (len(expr_mRNA)))


# In[60]:


expr_gene_mRNA = expr_gene_filt[(expr_gene_filt["gene_biotype"] == "protein_coding") & (expr_gene_filt["threshold"] == "expressed")]
print("there are %s protein-coding genes expressed at at least 0.1 tpm in any of the 4 lineages" % (len(expr_gene_mRNA)))


# In[61]:


expr_mRNA_heatmap = expr_gene_mRNA.sort_values(by="overall_mean", ascending=False).head(10000)[["hESC_rep1", "hESC_rep2", "endo_rep1", "endo_rep2", "meso_rep1", "meso_rep2"]]
sns.clustermap(expr_mRNA_heatmap, yticklabels=False, z_score=0, cmap=cmap, figsize=(2, 3.5), metric="correlation",
               **{"rasterized": True}, linewidths=0.0)
plt.suptitle("expression of all mRNAs\n[z-score plotted]")
plt.savefig("Fig2A_2.pdf", bbox_inches="tight")


# ### 7. plot expression of certain markers

# In[62]:


marker_genes = ["POU5F1", "NANOG", "EOMES", "GATA6", "T"]
sub = expr_gene[expr_gene["gene_name"].isin(marker_genes)]
sub


# In[63]:


fig, axarr = plt.subplots(figsize=(0.75, 3.5), ncols=1,  nrows=5, sharex=True)

for i, gene in enumerate(marker_genes):
    sub = expr_gene[expr_gene["gene_name"] == gene][["gene_name", "hESC_rep1", "hESC_rep2", "endo_rep1", "endo_rep2",
                                                     "meso_rep1", "meso_rep2"]]
    sub = pd.melt(sub, id_vars="gene_name")
    sub["sample"] = sub["variable"].str.split("_", expand=True)[0]
    sub["log_val"] = np.log10(sub["value"]+1)
    ax = axarr[i]
    sns.barplot(data=sub, x="sample", y="value", color=sns.color_palette("deep")[0], ax=ax, errwidth=1.5, ci=90,
                errcolor="gray")
    ax.set_yscale("symlog")
    
    ax.set_ylabel("%s\ntpm" % gene)
    ax.set_xlabel("")
    if i == 4:
        ax.set_xticklabels(["hESC", "endo", "meso"], rotation=50, va="top", ha="right")
        
fig.savefig("Fig2B.pdf", dpi="figure", bbox_inches="tight")


# ### 8. plot tissue specificity of lncRNAs vs mRNAs

# In[64]:


all_ncRNA = expr_gene[expr_gene["csf"] != "protein_coding"]
all_mRNA = expr_gene[expr_gene["csf"] == "protein_coding"]


# In[65]:


all_ncRNA_mean = all_ncRNA[["hESC_mean", "endo_mean", "meso_mean"]]
all_mRNA_mean = all_mRNA[["hESC_mean", "endo_mean", "meso_mean"]]

all_ncRNA_mean = all_ncRNA_mean + 1
all_mRNA_mean = all_mRNA_mean + 1

all_ncRNA_mean = np.log10(all_ncRNA_mean)
all_mRNA_mean = np.log10(all_mRNA_mean)

all_ncRNA_mean["gene_name"] = all_ncRNA["gene_name"]
all_mRNA_mean["gene_name"] = all_mRNA["gene_name"]


# In[66]:


all_ncRNA_mean_array = all_ncRNA_mean.drop("gene_name", axis=1).as_matrix()
ncRNA_max = np.max(all_ncRNA_mean_array, axis=1)
tmp = all_ncRNA_mean_array.T / ncRNA_max
tmp = 1 - tmp.T
ncRNA_specs = np.sum(tmp, axis=1)/3
print(len(ncRNA_specs))
print(len(all_ncRNA))


# In[67]:


all_mRNA_mean_array = all_mRNA_mean.drop("gene_name", axis=1).as_matrix()
mRNA_max = np.max(all_mRNA_mean_array, axis=1)
tmp = all_mRNA_mean_array.T / mRNA_max
tmp = 1 - tmp.T
mRNA_specs = np.sum(tmp, axis=1)/3
print(len(mRNA_specs))
print(len(all_mRNA))


# In[68]:


# # randomly sample for cdf plotting
# ncRNA_specs_samp = np.random.choice(ncRNA_specs, size=50)
# mRNA_specs_samp = np.random.choice(mRNA_specs, size=50)


# In[69]:


# remove transcripts with all 0 expression values (these have tissue spec of NA)
mRNA_specs_nonan = mRNA_specs[~np.isnan(mRNA_specs)]
ncRNA_specs_nonan = ncRNA_specs[~np.isnan(ncRNA_specs)]

print(len(mRNA_specs_nonan))
print(len(ncRNA_specs_nonan))


# In[70]:


all_ncRNA_mean["tissue_spec"] = ncRNA_specs
all_ncRNA_mean["type"] = "lncRNA"

all_mRNA_mean["tissue_spec"] = mRNA_specs
all_mRNA_mean["type"] = "mRNA"

tissue_spec = all_ncRNA_mean.append(all_mRNA_mean)
tissue_spec.sample(5)


# In[71]:


# compare lncRNAs to TFs
tfs = pd.read_table(tfs_f)
print(len(tfs.gene_name.unique()))
tfs.head()


# In[72]:


tfs_spec = tissue_spec[tissue_spec["gene_name"].isin(tfs["gene_name"])]["tissue_spec"]
len(tfs_spec)


# In[73]:


tf_specs_nonan = tfs_spec[~np.isnan(tfs_spec)]
len(tf_specs_nonan)


# In[74]:


fig, axarr = plt.subplots(figsize=(1.25,1.75), nrows=3, ncols=1, sharex=True, sharey=True)

ax = axarr[0]
sns.distplot(ncRNA_specs_nonan, label="lncRNAs", color="firebrick", 
             hist=False, kde_kws=dict(cumulative=False), ax=ax)
ax.get_legend().remove()
ax.set_xlabel("")
ax.set_ylabel("density")

ax = axarr[1]
sns.distplot(tf_specs_nonan, label="transcription factors", color=sns.color_palette("Set2")[1],
             hist=False, kde_kws=dict(cumulative=False), ax=ax)
ax.get_legend().remove()
ax.set_xlabel("")
ax.set_ylabel("density")

ax = axarr[2]
sns.distplot(mRNA_specs_nonan, label="protein-coding genes", 
             hist=False, kde_kws=dict(cumulative=False), ax=ax)
ax.get_legend().remove()
ax.set_xlabel("gene tissue specificity score\n(0=ubiquitous, 1=specific)")
ax.set_ylabel("density")

fig.subplots_adjust(hspace=0.3)
fig.savefig("Fig2F.pdf", bbox_inches="tight", dpi="figure")


# ### 9. GO plots

# In[75]:


endo_go_f = "../../misc/02__GO_enrichments/endo_enriched.txt"
endo_go = pd.read_table(endo_go_f, skiprows=7)
endo_go.head()


# In[76]:


endo_go["neg_p"] = -np.log10(endo_go["upload_1 (P-value)"])
endo_go["process"] = endo_go["GO biological process complete"].str.split(' \\(', expand=True)[0]
endo_go.head()


# In[77]:


meso_go_f = "../../misc/02__GO_enrichments/meso_enriched.txt"
meso_go = pd.read_table(meso_go_f, skiprows=7)
meso_go.head()


# In[78]:


meso_go["neg_p"] = -np.log10(meso_go["upload_1 (P-value)"])
meso_go["process"] = meso_go["GO biological process complete"].str.split(' \\(', expand=True)[0]
meso_go.head()


# In[79]:


fig, axarr = plt.subplots(figsize=(2, 3.5), ncols=1, nrows=2, sharex=False)
labels = ["endoderm", "mesoderm"]
for i, df in enumerate([endo_go, meso_go]):
    label = labels[i]
    ax = axarr[i]
    
    sub = df.sort_values(by="upload_1 (fold Enrichment)", ascending=False).head(10)
    sub["ytick"] = list(range(len(sub)))[::-1]
    
    for i, row in sub.iterrows():
        xmax = row["upload_1 (fold Enrichment)"]
        y = row.ytick
        ax.plot([0, xmax], [y, y], color="gray", linewidth=0.5)
    
    sns.scatterplot(data=sub, x="upload_1 (fold Enrichment)", y="ytick", size="neg_p", ax=ax, legend=False,
                    color=sns.color_palette("Set2")[2], sizes=(30, 70), zorder=10)
    
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["process"][::-1])
    ax.set_ylabel("")
    ax.set_xlim((0, 7))
    
    ax.set_xlabel("enrichment in %s" % label)

fig.subplots_adjust(hspace=0.4)
fig.savefig("Fig2C.pdf", dpi="figure", bbox_inches="tight")


# ### 8. find differentially expressed transcripts

# In[80]:


diff_hESC_endo = pd.read_table(diff_hESC_endo_f, sep="\t")
diff_hESC_meso = pd.read_table(diff_hESC_meso_f, sep="\t")


# In[81]:


diff_hESC_endo.fillna("sleuth NA", inplace=True)
diff_hESC_meso.fillna("sleuth NA", inplace=True)


# In[82]:


diff = diff_hESC_endo.merge(diff_hESC_meso, on="target_id", how="outer", suffixes=("_hESC_endo", "_hESC_meso"))
diff.head()


# In[83]:


diff[diff["target_id"] == "DIGIT"]


# In[84]:


diff["hESC_vs_endo"] = diff.apply(is_significant, colname="qval_hESC_endo", alpha=0.05, axis=1)
diff["hESC_vs_meso"] = diff.apply(is_significant, colname="qval_hESC_meso", alpha=0.05, axis=1)
diff.sample(5)


# In[85]:


expr_diff = expr.merge(diff, left_on="transcript_id", right_on="target_id", how="inner")
expr_diff["endo_pseudo"] = expr_diff["endo_mean"] + 1
expr_diff["hESC_pseudo"] = expr_diff["hESC_mean"] + 1
expr_diff["meso_pseudo"] = expr_diff["meso_mean"] + 1
expr_diff["endo_hESC_log2fc"] = np.log2(expr_diff["endo_pseudo"]/expr_diff["hESC_pseudo"])
expr_diff["meso_hESC_log2fc"] = np.log2(expr_diff["meso_pseudo"]/expr_diff["hESC_pseudo"])
expr_diff["endo_hESC_log2fc_abs"] = np.abs(expr_diff["endo_hESC_log2fc"])
expr_diff["meso_hESC_log2fc_abs"] = np.abs(expr_diff["meso_hESC_log2fc"])
expr_diff.sample(5)


# In[86]:


sig_hESC_endo = expr_diff[(expr_diff["hESC_vs_endo"] == "significant")]
sig_hESC_meso = expr_diff[(expr_diff["hESC_vs_meso"] == "significant")]

print("# of significantly differentially expressed transcripts b/w hESC and endo: %s" % (len(sig_hESC_endo)))
print("# of significantly differentially expressed transcripts b/w hESC and meso: %s" % (len(sig_hESC_meso)))


# In[87]:


print("# of significantly differentially expressed genes with b/w hESC and endo: %s" % (len(sig_hESC_endo.gene_name.unique())))
print("# of significantly differentially expressed genes with b/w hESC and meso: %s" % (len(sig_hESC_meso.gene_name.unique())))


# In[88]:


sig_hESC_endo_ncRNA = sig_hESC_endo[sig_hESC_endo["gene_name"].isin(expr_ncRNA["gene_name"])]
sig_hESC_meso_ncRNA = sig_hESC_meso[sig_hESC_meso["gene_name"].isin(expr_ncRNA["gene_name"])]


# In[89]:


print("# of significantly differentially expressed ncRNA transcripts b/w hESC and endo: %s" % (len(sig_hESC_endo_ncRNA)))
print("# of significantly differentially expressed ncRNA transcripts b/w hESC and meso: %s" % (len(sig_hESC_meso_ncRNA)))


# In[90]:


print("# of significantly differentially expressed ncRNA transcripts b/w hESC and endo: %s" % (len(sig_hESC_endo_ncRNA.gene_name.unique())))
print("# of significantly differentially expressed ncRNA transcripts b/w hESC and meso: %s" % (len(sig_hESC_meso_ncRNA.gene_name.unique())))


# In[91]:


sig_2fc_hESC_endo = expr_diff[(expr_diff["hESC_vs_endo"] == "significant") & 
                              (expr_diff["endo_hESC_log2fc_abs"] > 1)]
sig_2fc_hESC_meso = expr_diff[(expr_diff["hESC_vs_meso"] == "significant") & 
                              (expr_diff["meso_hESC_log2fc_abs"] > 1)]

print("# of significantly differentially expressed transcripts with >2 foldchange b/w hESC and endo: %s" % (len(sig_2fc_hESC_endo)))
print("# of significantly differentially expressed transcripts with >2 foldchange b/w hESC and meso: %s" % (len(sig_2fc_hESC_meso)))


# In[92]:


print("# of significantly differentially expressed genes with >2 foldchange b/w hESC and endo: %s" % (len(sig_2fc_hESC_endo.gene_name.unique())))
print("# of significantly differentially expressed genes with >2 foldchange b/w hESC and meso: %s" % (len(sig_2fc_hESC_meso.gene_name.unique())))


# In[93]:


sig_2fc_hESC_endo_ncRNA = sig_2fc_hESC_endo[sig_2fc_hESC_endo["gene_name"].isin(expr_ncRNA["gene_name"])]
sig_2fc_hESC_meso_ncRNA = sig_2fc_hESC_meso[sig_2fc_hESC_meso["gene_name"].isin(expr_ncRNA["gene_name"])]


# In[94]:


print("# of significantly differentially expressed ncRNA transcripts with >2 foldchange b/w hESC and endo: %s" % (len(sig_2fc_hESC_endo_ncRNA)))
print("# of significantly differentially expressed ncRNA transcripts with >2 foldchange b/w hESC and meso: %s" % (len(sig_2fc_hESC_meso_ncRNA)))


# In[95]:


print("# of significantly differentially expressed ncRNA transcripts with >2 foldchange b/w hESC and endo: %s" % (len(sig_2fc_hESC_endo_ncRNA.gene_name.unique())))
print("# of significantly differentially expressed ncRNA transcripts with >2 foldchange b/w hESC and meso: %s" % (len(sig_2fc_hESC_meso_ncRNA.gene_name.unique())))


# ## 9. volcano plots

# In[96]:


diff_hESC_endo = expr_diff[expr_diff["qval_hESC_endo"] != "sleuth NA"]
diff_hESC_meso = expr_diff[expr_diff["qval_hESC_meso"] != "sleuth NA"]
diff_hESC_endo["qval_log10_hESC_endo"] = -np.log10(diff_hESC_endo["qval_hESC_endo"].astype(float))
diff_hESC_meso["qval_log10_hESC_meso"] = -np.log10(diff_hESC_meso["qval_hESC_meso"].astype(float))


# In[97]:


diff_hESC_endo["hESC_vs_endo_log"] = diff_hESC_endo.apply(mark_for_volcano, exp_col="endo_hESC_log2fc_abs", sig_col="hESC_vs_endo", axis=1)
diff_hESC_meso["hESC_vs_meso_log"] = diff_hESC_meso.apply(mark_for_volcano, exp_col="meso_hESC_log2fc_abs", sig_col="hESC_vs_meso", axis=1)


# In[98]:


diff_hESC_endo_ncRNA = diff_hESC_endo[diff_hESC_endo["gene_name"].isin(expr_ncRNA["gene_name"])]
diff_hESC_meso_ncRNA = diff_hESC_meso[diff_hESC_meso["gene_name"].isin(expr_ncRNA["gene_name"])]


# In[99]:


diff_hESC_endo_ncRNA.hESC_vs_endo_log.value_counts()


# In[100]:


diff_hESC_meso_ncRNA.hESC_vs_meso_log.value_counts()


# In[101]:


len(diff_hESC_endo_ncRNA[diff_hESC_endo_ncRNA["qval_hESC_endo"] < 0.05])


# In[102]:


len(diff_hESC_endo_ncRNA[diff_hESC_endo_ncRNA["qval_hESC_endo"] < 0.05]["gene_name"].unique())


# In[103]:


fig = plt.figure(figsize=(1.5, 1.75))
g = sns.regplot(x="endo_hESC_log2fc", y="qval_log10_hESC_endo", data=diff_hESC_endo_ncRNA, fit_reg=False, 
               color="firebrick", scatter_kws={"s": 8, "edgecolors": "white", "linewidths": 0.5})

plt.xlabel("log2(endoderm/hESC)")
plt.ylabel("negative log10 q value")
plt.ylim((-0.1, 4))
plt.xlim((-8.5, 8.5))
plt.axhline(y=-np.log10(0.05), linestyle="dashed", color="black", linewidth=1)
#plt.title("volcano plot for ncRNAs in endoderm vs. hESCs\n(n=%s)" % (len(diff_hESC_endo_ncRNA)))
plt.savefig("Fig2E_1.pdf", bbox_inches="tight", dpi="figure")


# In[104]:


fig = plt.figure(figsize=(1.5, 1.75))
g = sns.regplot(x="meso_hESC_log2fc", y="qval_log10_hESC_meso", data=diff_hESC_meso_ncRNA, fit_reg=False, 
               color="firebrick", scatter_kws={"s": 8, "edgecolors": "white", "linewidths": 0.5})

plt.xlabel("log2(mesoderm/hESC)")
plt.ylabel("negative log10 q value")
plt.ylim((-0.1, 2.75))
#plt.xlim((-8.5, 8.5))
plt.axhline(y=-np.log10(0.05), linestyle="dashed", color="black", linewidth=1)
#plt.title("volcano plot for ncRNAs in endoderm vs. hESCs\n(n=%s)" % (len(diff_hESC_endo_ncRNA)))
plt.savefig("Fig2E_2.pdf", bbox_inches="tight", dpi="figure")


# In[105]:


len(diff_hESC_meso_ncRNA[diff_hESC_meso_ncRNA["qval_hESC_meso"] < 0.05])


# In[106]:


len(diff_hESC_meso_ncRNA[diff_hESC_meso_ncRNA["qval_hESC_meso"] < 0.05]["gene_name"].unique())


# In[107]:


meso_tx = list(diff_hESC_meso_ncRNA[diff_hESC_meso_ncRNA["qval_hESC_meso"] < 0.05]["transcript_id"])
endo_tx = list(diff_hESC_endo_ncRNA[diff_hESC_endo_ncRNA["qval_hESC_endo"] < 0.05]["transcript_id"])
tot_tx = meso_tx + endo_tx
tot_tx = set(tot_tx)
len(tot_tx)


# In[108]:


len(set(meso_tx).intersection(set(endo_tx)))


# In[109]:


meso_g = list(diff_hESC_meso_ncRNA[diff_hESC_meso_ncRNA["qval_hESC_meso"] < 0.05]["gene_name"])
endo_g = list(diff_hESC_endo_ncRNA[diff_hESC_endo_ncRNA["qval_hESC_endo"] < 0.05]["gene_name"])
tot_g = meso_g + endo_g
tot_g = set(tot_g)
len(tot_g)


# In[110]:


len(set(meso_g).intersection(set(endo_g)))


# ## 12. write final gene list file

# In[111]:


final = expr[["transcript_id", "gene_id", "gene_name", "csf", "cleaner_gene_biotype", 
              "cleaner_transcript_biotype", "hESC_rep1",
              "hESC_rep2", "endo_rep1", "endo_rep2", "meso_rep1", "meso_rep2",
              "hESC_mean", "endo_mean", "meso_mean", "overall_mean", "threshold"]]
final = final.merge(expr_diff[["transcript_id", "gene_id", "gene_name", "csf", 
                               "cleaner_gene_biotype", "cleaner_transcript_biotype", "qval_hESC_endo", 
                               "qval_hESC_meso", "endo_hESC_log2fc", 
                               "meso_hESC_log2fc"]], 
                    on=["transcript_id", "gene_id", "gene_name", "csf", "cleaner_gene_biotype",
                        "cleaner_transcript_biotype"],
                    how="left")
final = final[final["transcript_id"].isin(expr_filt["transcript_id"])]
final.fillna("filter not met", inplace=True)
final.head()


# In[112]:


len(final)


# In[113]:


final.columns


# In[114]:


f = "../../data/00__rna_seq/01__processed_results/rna_seq_results.tsv"


# In[115]:


final.to_csv(f, sep="\t", index=False)


# ## 13. write supplemental table -- lncRNA RNA-seq results

# In[116]:


supp = final[final["csf"] == "lncRNA_good_csf"]
len(supp)


# In[117]:


supp["abs_l2fc"] = np.abs(supp["endo_hESC_log2fc"])
supp = supp.sort_values(by="abs_l2fc", ascending=False)


# In[118]:


supp = supp[["gene_name", "gene_id", "transcript_id", "cleaner_transcript_biotype",
             "hESC_rep1", "hESC_rep2", "endo_rep1", "endo_rep2", "meso_rep1", "meso_rep2", "threshold",
             "endo_hESC_log2fc", "qval_hESC_endo", "meso_hESC_log2fc", "qval_hESC_meso"]]


# In[119]:


supp.columns = ["gene_name", "gene_id", "transcript_id", "biotype", "hESC_rep1_tpm", "hESC_rep2_tpm",
                "endo_rep1_tpm", "endo_rep2_tpm", "meso_rep1_tpm", "meso_rep2_tpm", "meets_threshold",
                "endo_hESC_log2fc", "endo_hESC_padj", "meso_hESC_log2fc", "meso_hESC_padj"]
supp.head()


# In[120]:


f = "../../data/00__rna_seq/01__processed_results/SuppTable.RNA_seq.txt"
supp.to_csv(f, sep="\t", index=False)

