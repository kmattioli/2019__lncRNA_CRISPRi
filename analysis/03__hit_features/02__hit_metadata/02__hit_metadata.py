
# coding: utf-8

# # 04__features
# 
# in this notebook, i examine how various genomic features relate to hit status in our CRISPRi screen
# 
# figure in this notebook:
# - Fig 5A-5F: boxplot showing various genomic features for hits vs. non-hits in our CRISPR screen

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

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
mpl.rcParams['figure.autolayout'] = False


# In[2]:


sns.set(**PAPER_PRESET)
fontsize = PAPER_FONTSIZE


# ## variables

# In[3]:


index_f = "../../../data/01__design/02__final_sgRNAs/crispri_picked_sgRNAs.not_deduped.txt"


# In[4]:


data_f = "../../../data/02__screen/02__enrichment_data/enrichment_values.with_rna_seq.txt"


# In[5]:


# all metadata files
meta_dir = "../../../misc/00__gene_metadata"
closest_enh_f = "%s/tss_coords.closest_f5_enh.with_DIGIT.txt" % meta_dir
closest_enh_genebody_f = "%s/transcript_coords.closest_f5_enh.with_DIGIT.txt" % meta_dir
trans_len_f = "%s/transcript_length.with_DIGIT.txt" % meta_dir
trans_len_RNA_f = "%s/transcript_length_RNA.with_DIGIT.txt" % meta_dir
n_exons_f = "%s/n_exons_per_transcript.with_DIGIT.txt" % meta_dir
phastcons_f = "%s/tss_coords.100buff.phastCons.summary.with_DIGIT.txt" % meta_dir
phylop100_f = "%s/tss_coords.100buff.phyloP100way.summary.with_DIGIT.txt" % meta_dir
phylop46_f = "%s/tss_coords.100buff.phyloP46way_placental.summary.with_DIGIT.txt" % meta_dir


# ## 1. import data

# In[6]:


index = pd.read_table(index_f)
index.head()


# In[7]:


data = pd.read_table(data_f)
data.head()


# In[8]:


closest_enh = pd.read_table(closest_enh_f, sep="\t", header=None)
closest_enh.columns = ["chr", "start", "end", "transcript_id", "score", "strand", "closest_enh_id", "enh_distance"]
closest_enh.head()


# In[9]:


closest_enh_genebody = pd.read_table(closest_enh_genebody_f, sep="\t", header=None)
closest_enh_genebody.columns = ["chr", "start", "end", "transcript_id", "score", "strand", "closest_enh_id", "enh_distance"]
closest_enh_genebody.head()


# In[10]:


trans_len = pd.read_table(trans_len_f, sep="\t", header=None)
trans_len.columns = ["transcript_id", "transcript_length"]
trans_len.head()


# In[11]:


trans_len_RNA = pd.read_table(trans_len_RNA_f, sep="\t", header=None)
trans_len_RNA.columns = ["transcript_id", "transcript_length_RNA"]
trans_len_RNA.head()


# In[12]:


n_exons = pd.read_table(n_exons_f, sep="\t", header=None)
n_exons.columns = ["n_exons", "gene_id", "transcript_id"]
n_exons.head()


# In[13]:


phastcons = pd.read_table(phastcons_f, sep="\t")
phylop100 = pd.read_table(phylop100_f, sep="\t")
phylop46 = pd.read_table(phylop46_f, sep="\t")
phastcons.head()


# ## 2. join metadata files

# In[14]:


closest_enh = closest_enh[["transcript_id", "closest_enh_id", "enh_distance"]].drop_duplicates()
closest_enh.columns = ["transcript_id", "tss_closest_enh_id", "tss_enh_distance"]
closest_enh_genebody = closest_enh_genebody[["transcript_id", "closest_enh_id", "enh_distance"]].drop_duplicates()
closest_enh_genebody.columns = ["transcript_id", "gb_closest_enh_id", "gb_enh_distance"]
trans_len = trans_len.drop_duplicates()
trans_len_RNA = trans_len_RNA.drop_duplicates()
n_exons = n_exons[["transcript_id", "n_exons"]].drop_duplicates()
phastcons = phastcons[["name", "mean"]].drop_duplicates()
phastcons.columns = ["transcript_id", "phastcons_mean"]
phylop100 = phylop100[["name", "mean"]].drop_duplicates()
phylop100.columns = ["transcript_id", "phylop100_mean"]
phylop46 = phylop46[["name", "mean"]].drop_duplicates()
phylop46.columns = ["transcript_id", "phylop46_mean"]


# In[15]:


print(len(closest_enh))
print(len(closest_enh_genebody))
print(len(trans_len))
print(len(trans_len_RNA))
print(len(n_exons))
print(len(phastcons))
print(len(phylop100))
print(len(phylop46))


# In[16]:


closest_enh_genebody = closest_enh_genebody.drop_duplicates(subset="transcript_id")
len(closest_enh_genebody)


# In[17]:


meta = closest_enh.merge(closest_enh_genebody, on="transcript_id", how="left").merge(trans_len, on="transcript_id", how="left").merge(trans_len_RNA, on="transcript_id", how="left")
len(meta)


# In[18]:


meta = meta.merge(n_exons, on="transcript_id", how="left")
len(meta)


# In[19]:


meta = meta.merge(phastcons, on="transcript_id", how="left").merge(phylop100, 
                                                                   on="transcript_id", 
                                                                   how="left").merge(phylop46, on="transcript_id",
                                                                                     how="left")


# In[20]:


# some transcripts have duplicate closest enhancers when there are ties, so drop these (keep first arbitrarily)
meta.drop_duplicates(subset="transcript_id", inplace=True)
print(len(meta))
meta.head()


# In[21]:


# see if transcript is conserved
id_map = data[["transcript_id", "transcript_name", "gene_id", "gene_name"]].drop_duplicates()
id_map["short_transcript_id"] = id_map["transcript_id"].str.split("_", expand=True)[0]
id_map.head()


# In[22]:


meta[meta["transcript_id"] == "DIGIT"]


# ## 3. merge metadata info with enrichment/hit info

# In[23]:


len(data)


# In[24]:


data.is_hit.value_counts()


# In[25]:


data = data.merge(meta, on="transcript_id", how="left")
print(len(data))
data.head()


# In[26]:


data.is_hit.value_counts()


# ## 4. series of plots to examine each feature & relationship to hit status

# In[27]:


order = ["lncRNA_good_csf", "protein_coding"]
hue_order = ["no hit", "stringent hit"]
pal = {"no hit": "gray", "lenient hit": sns.color_palette("Set2")[2], "stringent hit": "black"}


# In[28]:


hue_order2 = ["no hit", "hit"]
pal2 = {"no hit": "gray", "hit": "black"}


# ### distance to closest enhancer -- gene body

# In[29]:


fig, axarr = plt.subplots(figsize=(1.75, 1.85), nrows=1, ncols=2, sharey=True)

for i, csf in enumerate(order):
    ax = axarr[i]
    sub = data[data["csf"] == csf]
    sns.boxplot(data=sub, x="is_hit", y="gb_enh_distance",
                flierprops = dict(marker='o', markersize=3), order=hue_order, palette=pal, ax=ax)
    mimic_r_boxplot(ax)
    ax.set_xticklabels(hue_order, rotation=45, ha="right", va="top")
    ax.set_xlabel("")
    ax.set_ylabel("distance to closest enhancer")
    ax.set_title(csf)
    if i != 0:
        ax.set_ylabel("")
    ax.set_yscale("log")
    
    # calc p-vals b/w dists
    dist1 = np.asarray(sub[sub["is_hit"] == "no hit"]["gb_enh_distance"])
    dist2 = np.asarray(sub[sub["is_hit"] == "stringent hit"]["gb_enh_distance"])

    dist1 = dist1[~np.isnan(dist1)]
    dist2 = dist2[~np.isnan(dist2)]

    u, pval = stats.mannwhitneyu(dist1, dist2, alternative="greater", use_continuity=False)
    print(pval)
    
    # annotate pval
    if i == 0:
        annotate_pval(ax, 0.2, 0.8, 100000, 0, 100000, pval, fontsize)
    else:
        annotate_pval(ax, 0.2, 0.8, 45000, 0, 45000, pval, fontsize)

plt.subplots_adjust(wspace=0.1)
fig.savefig("Fig5B.pdf", dpi="figure", bbox_inches="tight")


# ### locus length

# In[30]:


fig, axarr = plt.subplots(figsize=(1.75, 1.85), nrows=1, ncols=2, sharey=True)

for i, csf in enumerate(order):
    ax = axarr[i]
    sub = data[data["csf"] == csf]
    sns.boxplot(data=sub, x="is_hit", y="transcript_length",
                flierprops = dict(marker='o', markersize=3), order=hue_order, palette=pal, ax=ax)
    mimic_r_boxplot(ax)
    ax.set_xticklabels(hue_order, rotation=45, ha="right", va="top")
    ax.set_xlabel("")
    ax.set_ylabel("locus length")
    ax.set_title(csf)
    if i != 0:
        ax.set_ylabel("")
    ax.set_yscale("log")
    
    # calc p-vals b/w dists
    dist1 = np.asarray(sub[sub["is_hit"] == "no hit"]["transcript_length"])
    dist2 = np.asarray(sub[sub["is_hit"] == "stringent hit"]["transcript_length"])

    dist1 = dist1[~np.isnan(dist1)]
    dist2 = dist2[~np.isnan(dist2)]

    u, pval1 = stats.mannwhitneyu(dist1, dist2, alternative="less", use_continuity=False)
    print(pval1)

    u, pval2 = stats.mannwhitneyu(dist1, dist2, alternative="greater", use_continuity=False)
    print(pval2)
    
    # annotate pval
    if i == 0:
        annotate_pval(ax, 0.2, 0.8, 80000, 0, 80000, pval1, fontsize)
    else:
        annotate_pval(ax, 0.2, 0.8, 100000, 0, 100000, pval2, fontsize)

plt.subplots_adjust(wspace=0.1)
fig.savefig("Fig5D.pdf", dpi="figure", bbox_inches="tight")


# ### transcript length

# In[31]:


fig, axarr = plt.subplots(figsize=(1.75, 1.85), nrows=1, ncols=2, sharey=True)

for i, csf in enumerate(order):
    ax = axarr[i]
    sub = data[data["csf"] == csf]
    sns.boxplot(data=sub, x="is_hit", y="transcript_length_RNA",
                flierprops = dict(marker='o', markersize=3), order=hue_order, palette=pal, ax=ax)
    mimic_r_boxplot(ax)
    ax.set_xticklabels(hue_order, rotation=45, ha="right", va="top")
    ax.set_xlabel("")
    ax.set_ylabel("RNA transcript length")
    ax.set_title(csf)
    if i != 0:
        ax.set_ylabel("")
    ax.set_yscale("log")
    
    # calc p-vals b/w dists
    dist1 = np.asarray(sub[sub["is_hit"] == "no hit"]["transcript_length_RNA"])
    dist2 = np.asarray(sub[sub["is_hit"] == "stringent hit"]["transcript_length_RNA"])

    dist1 = dist1[~np.isnan(dist1)]
    dist2 = dist2[~np.isnan(dist2)]

    u, pval1 = stats.mannwhitneyu(dist1, dist2, alternative="less", use_continuity=False)
    print(pval1)

    u, pval2 = stats.mannwhitneyu(dist1, dist2, alternative="greater", use_continuity=False)
    print(pval2)
    
    # annotate pval
    if i == 0:
        annotate_pval(ax, 0.2, 0.8, 1500, 0, 1500, pval1, fontsize)
    else:
        annotate_pval(ax, 0.2, 0.8, 5000, 0, 5000, pval1, fontsize)

plt.subplots_adjust(wspace=0.1)
fig.savefig("Fig5E.pdf", dpi="figure", bbox_inches="tight")


# ### number of exons

# In[32]:


fig, axarr = plt.subplots(figsize=(1.75, 1.85), nrows=1, ncols=2, sharey=True)

for i, csf in enumerate(order):
    ax = axarr[i]
    sub = data[data["csf"] == csf]
    sns.boxplot(data=sub, x="is_hit", y="n_exons",
                flierprops = dict(marker='o', markersize=3), order=hue_order, palette=pal, ax=ax)
    mimic_r_boxplot(ax)
    ax.set_xticklabels(hue_order, rotation=45, ha="right", va="top")
    ax.set_xlabel("")
    ax.set_ylabel("# of exons")
    ax.set_title(csf)
    if i != 0:
        ax.set_ylabel("")
    ax.set_yscale("log")
    
    # calc p-vals b/w dists
    dist1 = np.asarray(sub[sub["is_hit"] == "no hit"]["n_exons"])
    dist2 = np.asarray(sub[sub["is_hit"] == "stringent hit"]["n_exons"])

    dist1 = dist1[~np.isnan(dist1)]
    dist2 = dist2[~np.isnan(dist2)]
    
    print(np.median(dist1))
    print(np.median(dist2))

    u, pval1 = stats.mannwhitneyu(dist1, dist2, alternative="less", use_continuity=False)
    print(pval1)
    
    u, pval2 = stats.mannwhitneyu(dist1, dist2, alternative="greater", use_continuity=False)
    print(pval2)
    
    # annotate pval
    if i == 0:
        annotate_pval(ax, 0.2, 0.8, 5, 0, 5, pval1, fontsize)
    else:
        annotate_pval(ax, 0.2, 0.8, 12, 0, 12, pval2, fontsize)

plt.subplots_adjust(wspace=0.1)
fig.savefig("Fig5F.pdf", dpi="figure", bbox_inches="tight")


# ### conservation (phastcons)

# In[33]:


fig, axarr = plt.subplots(figsize=(1.75, 1.85), nrows=1, ncols=2, sharey=True)

for i, csf in enumerate(order):
    ax = axarr[i]
    sub = data[data["csf"] == csf]
    sns.boxplot(data=sub, x="is_hit", y="phastcons_mean",
                flierprops = dict(marker='o', markersize=3), order=hue_order, palette=pal, ax=ax)
    mimic_r_boxplot(ax)
    ax.set_xticklabels(hue_order, rotation=45, ha="right", va="top")
    ax.set_xlabel("")
    ax.set_ylabel("conservation +/- 100bp from TSS")
    ax.set_title(csf)
    if i != 0:
        ax.set_ylabel("")
#     ax.set_yscale("log")
    
    # calc p-vals b/w dists
    dist1 = np.asarray(sub[sub["is_hit"] == "no hit"]["phastcons_mean"])
    dist2 = np.asarray(sub[sub["is_hit"] == "stringent hit"]["phastcons_mean"])

    dist1 = dist1[~np.isnan(dist1)]
    dist2 = dist2[~np.isnan(dist2)]

    u, pval = stats.mannwhitneyu(dist1, dist2, alternative="less", use_continuity=False)
    print(pval)
    
    # annotate pval
    if i == 0:
        annotate_pval(ax, 0.2, 0.8, 0.3, 0, 0.3, pval, fontsize)
    else:
        annotate_pval(ax, 0.2, 0.8, 0.8, 0, 0.8, pval, fontsize)

plt.subplots_adjust(wspace=0.1)
fig.savefig("Fig5A.pdf", dpi="figure", bbox_inches="tight")


# ## write file

# In[34]:


hits = data[data["is_hit"] == "stringent hit"]
hits.csf.value_counts()


# In[35]:


len(meta)


# In[36]:


meta.to_csv("../../../data/02__screen/02__enrichment_data/transcript_features.txt", sep="\t", index=False)

