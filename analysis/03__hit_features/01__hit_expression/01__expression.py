#!/usr/bin/env python
# coding: utf-8

# # 01__expression
# 
# in this notebook, i join enrichment data with expression data and look into how the two are related.
# 
# figures in this notebook:
# - Fig 5H: volcano plot showing log2 foldchange in expression from RNA-seq with hits from our screen highlighted

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
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
sys.path.append("../../../utils")
from plotting_utils import *
from enrich_utils import *

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
mpl.rcParams['figure.autolayout'] = False


# In[2]:


sns.set(**PAPER_PRESET)
fontsize = PAPER_FONTSIZE


# ## variables

# In[3]:


rna_seq_f = "../../../data/00__rna_seq/01__processed_results/rna_seq_results.tsv"


# In[4]:


index_f = "../../../data/01__design/02__final_sgRNAs/crispri_picked_sgRNAs.not_deduped.txt"


# In[5]:


data_f = "../../../data/02__screen/02__enrichment_data/enrichment_values.txt"


# ## 1. import data

# In[6]:


rna_seq = pd.read_table(rna_seq_f, sep="\t")
rna_seq.head()


# In[7]:


data = pd.read_table(data_f, sep="\t")
data.head()


# In[8]:


index = pd.read_table(index_f)
print(len(index))
print(len(index.tss_id.unique()))


# ## 2. merge enrichment data (which is grouped by TSS group, which can target multiple transcripts) with transcript-level information in index (in order to merge w/ RNA-seq)
# since some TSS groups target multiple transcripts, split these up so that we can easily join with the RNA-seq data on transcript_id

# In[9]:


index_sub = index[["tss_id", "gene_name", "gene_id", "transcript_name", "transcript_id"]].drop_duplicates()
print(len(index_sub))
print(len(index_sub.tss_id.unique()))
index_sub.head()


# In[10]:


data_split = tidy_split(data, "group_id", sep=",", keep=False)
data_split["group_id"] = data_split.apply(clean_split_col, col="group_id", axis=1)
data_split.head(10)


# In[11]:


data.columns


# In[12]:


data_sub = data_split[["group_id", "ctrl_status", "endo_ctrl_val", "BFP+_score__rep1",
                       "BFP+_score_rank__rep1", "BFP+_score__rep2",
                       "BFP+_score_rank__rep2", "BFP+_score__mean", "BFP+_score_rank__mean", "ctrl_status_fixed", 
                       "pval__rep1", "pval__rep2", "combined_pval", "padj", "neg_log_padj"]].drop_duplicates()
print(len(data_sub))
data_sub.sample(5)


# In[13]:


data_sub = data_sub[data_sub["ctrl_status"] != "scramble"]
len(data_sub)


# In[14]:


data_clean = data_sub.merge(index_sub, left_on="group_id", right_on="tss_id", how="left")
print(len(data_clean))
data_clean.head()


# In[15]:


data_clean[pd.isnull(data_clean["transcript_id"])]


# ## 3. merge w/ RNA-seq data using transcript_id

# In[16]:


data_w_seq = data_clean.merge(rna_seq, on=["gene_name", "gene_id", "transcript_id"], 
                              how="left").sort_values(by="BFP+_score_rank__mean")
data_w_seq.drop(["meso_mean", "overall_mean", "qval_hESC_meso", "meso_hESC_log2fc"], axis=1, inplace=True)
data_w_seq.head(20)


# In[17]:


data_w_seq[data_w_seq["gene_name"] == "DIGIT"]


# In[18]:


nulls = data_w_seq[(pd.isnull(data_w_seq["transcript_id"])) & (data_w_seq["ctrl_status"] != "scramble")]
len(nulls)


# In[19]:


data_w_seq[data_w_seq["gene_name"] == "RP11-474D1.4"][["group_id", "gene_name", "transcript_name", "cleaner_transcript_biotype", "cleaner_gene_biotype"]]


# ## 4. plot expression change vs. enrichment score

# In[20]:


experimental = data_w_seq[data_w_seq["ctrl_status_fixed"] == "experimental"]
control = data_w_seq[data_w_seq["ctrl_status_fixed"] == "control"]

sox17 = control[control["transcript_name"] == "SOX17-001"]
foxa2 = control[control["transcript_name"] == "FOXA2-002"]
gata6 = control[control["transcript_name"] == "GATA6-001"]
eomes1 = control[control["transcript_name"] == "EOMES-004"]
eomes2 = control[control["transcript_name"] == "EOMES-001"]
gsc = control[control["transcript_name"] == "GSC-001"]


# In[21]:


control.sort_values(by="BFP+_score__mean", ascending=False)[["transcript_name", "gene_name"]].head(10)


# In[22]:


fig = plt.figure(figsize=(4,1.5))

plt.axhline(y=0, linestyle="dashed", color="black", linewidth=1)
plt.scatter(experimental["BFP+_score__mean"], experimental["endo_hESC_log2fc"], s=10,
            color="darkgray", alpha=0.9)
plt.scatter(control["BFP+_score__mean"], control["endo_hESC_log2fc"], s=10,
            color="green", alpha=1)

# plot controls
plt.scatter(sox17["BFP+_score__mean"], sox17["endo_hESC_log2fc"], s=10,
         color="green", alpha=1, linewidths=1, edgecolors="black")
plt.annotate(s="SOX17", xy=(sox17["BFP+_score__mean"], sox17["endo_hESC_log2fc"]),
             xytext=(-15, -10), textcoords="offset points", fontsize=7)

plt.scatter(foxa2["BFP+_score__mean"], foxa2["endo_hESC_log2fc"], s=10,
         color="green", alpha=1, linewidths=1, edgecolors="black")
plt.annotate(s="FOXA2", xy=(foxa2["BFP+_score__mean"], foxa2["endo_hESC_log2fc"]),
             xytext=(2, -5), textcoords="offset points", fontsize=7)

plt.scatter(gsc["BFP+_score__mean"], gsc["endo_hESC_log2fc"], s=10,
         color="green", alpha=1, linewidths=1, edgecolors="black")
plt.annotate(s="GSC", xy=(gsc["BFP+_score__mean"], gsc["endo_hESC_log2fc"]),
             xytext=(2, -5), textcoords="offset points", fontsize=7)

plt.scatter(gata6["BFP+_score__mean"], gata6["endo_hESC_log2fc"], s=10,
         color="green", alpha=1, linewidths=1, edgecolors="black")
plt.annotate(s="GATA6", xy=(gata6["BFP+_score__mean"], gata6["endo_hESC_log2fc"]),
             xytext=(-10, 6), textcoords="offset points", fontsize=7)

plt.scatter(eomes1["BFP+_score__mean"], eomes1["endo_hESC_log2fc"], s=10,
         color="green", alpha=1, linewidths=1, edgecolors="black")
plt.scatter(eomes2["BFP+_score__mean"], eomes2["endo_hESC_log2fc"], s=10,
         color="green", alpha=1, linewidths=1, edgecolors="black")
plt.annotate(s="EOMES", xy=(eomes1["BFP+_score__mean"], eomes1["endo_hESC_log2fc"]),
             xytext=(-15, -10), textcoords="offset points", fontsize=7)

plt.xlabel("screen transcript enrichment score")
plt.ylabel("log2(endo tpm/hESC tpm)")

plt.xscale('symlog')
 
# #plt.xlim((-0.05, 1.7))
#fig.savefig("FigS5D.pdf", dpi="figure", bbox_inches="tight")


# ## 5. plot expression, tissue-specificity & expression change for hits vs. non hits

# In[23]:


data_w_seq["is_hit"] = data_w_seq.apply(is_hit, axis=1)
data_w_seq.is_hit.value_counts()


# In[24]:


hits = data_w_seq[data_w_seq["is_hit"] == "stringent hit"]
print(len(hits.group_id.unique()))
hits.head()


# In[45]:


hits[["gene_name", "ctrl_status_fixed", "cleaner_gene_biotype", "BFP+_score__mean"]].drop_duplicates().sort_values(by="BFP+_score__mean", ascending=False)


# In[26]:


hits[hits["ctrl_status_fixed"] == "control"]


# In[27]:


print(len(hits[hits["ctrl_status_fixed"] == "control"]))
hits[hits["ctrl_status_fixed"] == "control"].sort_values(by="group_id")


# In[28]:


print(len(hits[~hits["endo_ctrl_val"]]))


# In[29]:


len(hits[hits["ctrl_status_fixed"] == "control"]["gene_id"].unique())


# In[30]:


len(hits[~hits["endo_ctrl_val"]]["gene_id"].unique())


# In[31]:


data_w_seq["endo_hESC_abslog2fc"] = np.abs(data_w_seq["endo_hESC_log2fc"])


# In[32]:


order = ["lncRNA_good_csf", "protein_coding"]
hue_order = ["no hit", "stringent hit"]
pal = {"no hit": "gray", "lenient hit": sns.color_palette("Set2")[2], "stringent hit": "black"}

fig, axarr = plt.subplots(figsize=(1.75, 1.85), nrows=1, ncols=2, sharey=True)

for i, csf in enumerate(order):
    ax = axarr[i]
    sub = data_w_seq[data_w_seq["csf"] == csf]
    sns.boxplot(data=sub, x="is_hit", y="endo_hESC_abslog2fc",
                flierprops = dict(marker='o', markersize=3), order=hue_order, palette=pal, ax=ax)
    mimic_r_boxplot(ax)
    ax.set_xticklabels(hue_order, rotation=45, ha="right", va="top")
    ax.set_xlabel("")
    ax.set_ylabel("| log2 (endo/hESC) |")
    ax.set_title(csf)
    if i != 0:
        ax.set_ylabel("")
    ax.set_yscale("log")
    
    # calc p-vals b/w dists
    dist1 = np.asarray(sub[sub["is_hit"] == "no hit"]["endo_hESC_abslog2fc"])
    dist2 = np.asarray(sub[sub["is_hit"] == "stringent hit"]["endo_hESC_abslog2fc"])

    dist1 = dist1[~np.isnan(dist1)]
    dist2 = dist2[~np.isnan(dist2)]

    u, pval = stats.mannwhitneyu(dist1, dist2, alternative="less", use_continuity=False)
    print(pval)
    
    # annotate pval
    if i == 0:
        annotate_pval(ax, 0.2, 0.8, 1, 0, 1, pval, fontsize)
    else:
        annotate_pval(ax, 0.2, 0.8, 8, 0, 7.8, pval, fontsize)

plt.subplots_adjust(wspace=0.2)
#fig.savefig("Fig5G.pdf", dpi="figure", bbox_inches="tight")


# In[33]:


fig, axarr = plt.subplots(figsize=(1.75, 1.85), nrows=1, ncols=2, sharey=True)

for i, csf in enumerate(order):
    ax = axarr[i]
    sub = data_w_seq[data_w_seq["csf"] == csf]
    sns.boxplot(data=sub, x="is_hit", y="endo_mean",
                flierprops = dict(marker='o', markersize=3), order=hue_order, palette=pal, ax=ax)
    mimic_r_boxplot(ax)
    ax.set_xticklabels(hue_order, rotation=45, ha="right", va="top")
    ax.set_xlabel("")
    ax.set_ylabel("mean endo expression")
    ax.set_title(csf)
    if i != 0:
        ax.set_ylabel("")
    ax.set_yscale("log")
    
    # calc p-vals b/w dists
    dist1 = np.asarray(sub[sub["is_hit"] == "no hit"]["endo_mean"])
    dist2 = np.asarray(sub[sub["is_hit"] == "stringent hit"]["endo_mean"])

    dist1 = dist1[~np.isnan(dist1)]
    dist2 = dist2[~np.isnan(dist2)]

    u, pval = stats.mannwhitneyu(dist1, dist2, alternative="less", use_continuity=False)
    print(pval)
    
    # annotate pval
    if i == 0:
        annotate_pval(ax, 0.2, 0.8, 2, 0, 2, pval, fontsize)
    else:
        annotate_pval(ax, 0.2, 0.8, 170, 0, 170, pval, fontsize)

plt.subplots_adjust(wspace=0.2)
#fig.savefig("Fig5H.pdf", dpi="figure", bbox_inches="tight")


# ## 6. plot expression change v. enrichment score for stringent hits only

# In[34]:


hits = data_w_seq[data_w_seq["is_hit"] == "stringent hit"]
experimental = hits[hits["ctrl_status_fixed"] == "experimental"]
control = hits[hits["ctrl_status_fixed"] == "control"]
control["gene_name"]


# In[35]:


fig, ax = plt.subplots(figsize=(1.5,1.85), nrows=1, ncols=1)

ax.axhline(y=0, linestyle="dashed", color="black", linewidth=1, zorder=1)
ax.scatter(experimental["BFP+_score__mean"], experimental["endo_hESC_log2fc"], s=20,
            color="darkgray", alpha=1, linewidths=1, edgecolors="slategray", zorder=10)
ax.scatter(control["BFP+_score__mean"], control["endo_hESC_log2fc"], s=20,
            color="green", alpha=1, linewidths=1, edgecolors="black", zorder=11)
ax.set_xlabel("transcript enrichment score")
ax.set_ylabel("log2 (endo/hESC)")
ax.set_xscale('symlog')
#fig.savefig("Fig5I.pdf", dpi="figure", bbox_inches="tight")


# ## 7. mark hits in all RNA-seq data

# In[36]:


no_na = data_w_seq[~pd.isnull(data_w_seq["qval_hESC_endo"])]
no_na = no_na[~no_na["qval_hESC_endo"].str.contains("NA")]
no_na["qval_log10_hESC_endo"] = -np.log10(no_na["qval_hESC_endo"].astype(float))
len(no_na)


# In[37]:


fig = plt.figure(figsize=(1.75, 1.5))

ncRNA = no_na[no_na["ctrl_status_fixed"] == "experimental"]
mRNA = no_na[no_na["ctrl_status_fixed"] == "control"]

ncRNA_hits = ncRNA[ncRNA["is_hit"] == "stringent hit"]
ctrl_hits = mRNA[mRNA["is_hit"] == "stringent hit"]

ax = sns.regplot(x="endo_hESC_log2fc", y="qval_log10_hESC_endo", data=no_na, fit_reg=False, 
                 color="darkgray", scatter_kws={"s": 8, "edgecolors": "white", "linewidths": 0.5})
sns.regplot(x="endo_hESC_log2fc", y="qval_log10_hESC_endo", data=ctrl_hits, fit_reg=False, 
            color=sns.color_palette()[2], scatter_kws={"s": 10, "edgecolors": "black", "linewidths": 0.5}, ax=ax)
sns.regplot(x="endo_hESC_log2fc", y="qval_log10_hESC_endo", data=ncRNA_hits, fit_reg=False, 
            color="black", scatter_kws={"s": 8, "edgecolors": "black", "linewidths": 0.5}, ax=ax)


plt.xlabel("log2(endoderm/hESC)")
plt.ylabel("negative log10 q value")
# plt.ylim((-0.1, 4))
# plt.xlim((-8.5, 8.5))
plt.axhline(y=-np.log10(0.05), linestyle="dashed", color="black", linewidth=1)
#plt.title("volcano plot for ncRNAs in endoderm vs. hESCs\n(n=%s)" % (len(diff_hESC_endo_ncRNA)))
plt.savefig("Fig5J.pdf", bbox_inches="tight", dpi="figure")


# ## 6. write file

# In[38]:


f = "../../../data/02__screen/02__enrichment_data/enrichment_values.with_rna_seq.txt"


# In[39]:


data_w_seq.columns


# In[40]:


data_w_seq = data_w_seq[["group_id", "ctrl_status", "endo_ctrl_val", "gene_name", "gene_id", "transcript_name",
                         "transcript_id", "cleaner_transcript_biotype", "cleaner_gene_biotype", "BFP+_score__rep1", 
                         "BFP+_score_rank__rep1", "pval__rep1", 
                         "BFP+_score__rep2", "BFP+_score_rank__rep2", "pval__rep2", "BFP+_score__mean", 
                         "BFP+_score_rank__mean", "combined_pval", "padj", "neg_log_padj", "is_hit",
                         "csf", "hESC_mean", "endo_mean", "qval_hESC_endo", "endo_hESC_log2fc",
                         "endo_hESC_abslog2fc"]]
data_w_seq.head()


# In[41]:


data_w_seq = data_w_seq.sort_values(by="BFP+_score__mean", ascending=False)
data_w_seq.to_csv(f, sep="\t", index=False)


# In[ ]:




