#!/usr/bin/env python
# coding: utf-8

# # 01__hit_expression
# 
# in this notebook, i join enrichment data with expression data and look into how the two are related. i also create a set of stringent non hits to serve as a comparison to screen hits.
# 
# figures in this notebook:
# - Fig 5G: volcano plot showing log2 foldchange in expression from RNA-seq with hits from our screen highlighted (looking only at stringent non hits)
# - Fig 7A: plot showing differential expression of the 6 hits predicted to have RNA mechanisms

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


data_f = "../../../data/02__screen/02__enrichment_data/SuppTable_S3.CRISPhieRmix_results.txt"


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


print(len(data))
data_split = tidy_split(data, "group_id", sep=",", keep=False)
print(len(data_split))
data_split["group_id"] = data_split["group_id"].str.replace('[', '')
data_split["group_id"] = data_split["group_id"].str.replace(']', '')
data_split["group_id"] = data_split["group_id"].str.replace(' ', '')
data_split.drop("transcript_name", axis=1, inplace=True)
data_split.head(10)


# In[11]:


data_sub = data_split[data_split["ctrl_status"] != "scramble"]
len(data_sub)


# In[12]:


data_clean = data_sub.merge(index_sub, left_on="group_id", 
                            right_on="tss_id", how="left")
print(len(data_clean))
data_clean.head()


# In[13]:


data_clean[pd.isnull(data_clean["transcript_id"])]


# ## 3. merge w/ RNA-seq data using transcript_id

# In[14]:


data_w_seq = data_clean.merge(rna_seq, on=["gene_name", "gene_id", "transcript_id"], 
                              how="left").sort_values(by="effect_size", ascending=False)
data_w_seq.drop(["meso_mean", "overall_mean", "qval_hESC_meso", "meso_hESC_log2fc"], axis=1, inplace=True)
data_w_seq.head(20)


# In[15]:


data_w_seq[data_w_seq["gene_name"] == "DIGIT"].iloc[0]


# In[16]:


nulls = data_w_seq[(pd.isnull(data_w_seq["transcript_id"])) & (data_w_seq["ctrl_status"] != "scramble")]
len(nulls)


# ## 4. plot expression change vs. enrichment score

# In[17]:


experimental = data_w_seq[data_w_seq["ctrl_status"] == "experimental"]
control = data_w_seq[data_w_seq["ctrl_status"] == "control"]

sox17 = control[control["transcript_name"] == "SOX17-001"]
foxa2 = control[control["transcript_name"] == "FOXA2-002"]
gata6 = control[control["transcript_name"] == "GATA6-001"]
eomes1 = control[control["transcript_name"] == "EOMES-004"]
eomes2 = control[control["transcript_name"] == "EOMES-001"]
gsc = control[control["transcript_name"] == "GSC-001"]


# In[18]:


experimental_nonan = experimental[~pd.isnull(experimental["CRISPhieRmix_FDR"])]
print(len(experimental))
print(len(experimental_nonan))


# In[19]:


control.sort_values(by="effect_size", ascending=False)[["transcript_name", "gene_name"]].head(10)


# In[20]:


fig = plt.figure(figsize=(4,1.5))

plt.axhline(y=0, linestyle="dashed", color="black", linewidth=1)
plt.scatter(experimental_nonan["effect_size"], experimental_nonan["endo_hESC_log2fc"], s=10,
            color="darkgray", alpha=0.9)
plt.scatter(control["effect_size"], control["endo_hESC_log2fc"], s=10,
            color="green", alpha=1)

# plot controls
plt.scatter(sox17["effect_size"], sox17["endo_hESC_log2fc"], s=10,
         color="green", alpha=1, linewidths=1, edgecolors="black")
plt.annotate(s="SOX17", xy=(sox17["effect_size"], sox17["endo_hESC_log2fc"]),
             xytext=(-15, -10), textcoords="offset points", fontsize=7)

plt.scatter(foxa2["effect_size"], foxa2["endo_hESC_log2fc"], s=10,
         color="green", alpha=1, linewidths=1, edgecolors="black")
plt.annotate(s="FOXA2", xy=(foxa2["effect_size"], foxa2["endo_hESC_log2fc"]),
             xytext=(2, -8), textcoords="offset points", fontsize=7)

plt.scatter(gsc["effect_size"], gsc["endo_hESC_log2fc"], s=10,
         color="green", alpha=1, linewidths=1, edgecolors="black")
plt.annotate(s="GSC", xy=(gsc["effect_size"], gsc["endo_hESC_log2fc"]),
             xytext=(2, -10), textcoords="offset points", fontsize=7)

plt.scatter(gata6["effect_size"], gata6["endo_hESC_log2fc"], s=10,
         color="green", alpha=1, linewidths=1, edgecolors="black")
plt.annotate(s="GATA6", xy=(gata6["effect_size"], gata6["endo_hESC_log2fc"]),
             xytext=(-10, 6), textcoords="offset points", fontsize=7)

plt.scatter(eomes1["effect_size"], eomes1["endo_hESC_log2fc"], s=10,
         color="green", alpha=1, linewidths=1, edgecolors="black")
plt.scatter(eomes2["effect_size"], eomes2["endo_hESC_log2fc"], s=10,
         color="green", alpha=1, linewidths=1, edgecolors="black")
plt.annotate(s="EOMES", xy=(eomes1["effect_size"], eomes1["endo_hESC_log2fc"]),
             xytext=(-15, -10), textcoords="offset points", fontsize=7)

plt.xlabel("screen transcript enrichment score")
plt.ylabel("log2(endo tpm/hESC tpm)")

#plt.xscale('symlog')
 
# #plt.xlim((-0.05, 1.7))
#fig.savefig("FigS5D.pdf", dpi="figure", bbox_inches="tight")


# ## 5. plot expression, tissue-specificity & expression change for hits vs. non hits

# In[21]:


def is_crisphie_hit(row):
    if pd.isnull(row.CRISPhieRmix_FDR):
        return "not considered"
    elif row.CRISPhieRmix_FDR < 0.1:
        return "hit"
    else:
        if row.CRISPhieRmix_FDR > 0.9:
            if row.n_sgRNA >= 9:
                return "stringent no hit"
            else:
                return "no hit"
        else:
            return "no hit"


# In[22]:


data_w_seq["is_hit"] = data_w_seq.apply(is_crisphie_hit, axis=1)
data_w_seq.is_hit.value_counts()


# In[23]:


data_w_seq[data_w_seq["is_hit"] == "stringent no hit"].ctrl_status.value_counts()


# In[24]:


data_w_seq[(data_w_seq["is_hit"] == "stringent no hit") & (data_w_seq["ctrl_status"] == "control")]


# In[25]:


hits = data_w_seq[data_w_seq["is_hit"] == "hit"]
print(len(hits.group_id.unique()))
hits.head()


# In[26]:


hits[["gene_name", "ctrl_status", "cleaner_transcript_biotype", "effect_size"]].drop_duplicates().sort_values(by="effect_size", ascending=False)


# In[27]:


hits[["gene_name", "ctrl_status", "cleaner_transcript_biotype", "effect_size"]].drop_duplicates().cleaner_transcript_biotype.value_counts()


# In[28]:


data_w_seq["endo_hESC_abslog2fc"] = np.abs(data_w_seq["endo_hESC_log2fc"])


# ## 6. plot expression change v. enrichment score for stringent hits only

# In[29]:


data_w_seq["neg_log_FDR"] = -np.log10(data_w_seq["CRISPhieRmix_FDR"]+1e-12)


# In[30]:


hits = data_w_seq[data_w_seq["is_hit"] == "hit"]
experimental = hits[hits["ctrl_status"] == "experimental"]
control = hits[hits["ctrl_status"] == "control"]
mrna = hits[hits["ctrl_status"] == "mRNA"]
control["gene_name"]


# In[31]:


nopromover = experimental[experimental["cleaner_transcript_biotype"] != "promoter_overlap"]
len(nopromover)


# In[32]:


experimental.sort_values(by="endo_hESC_abslog2fc", ascending=False)[["gene_name", "transcript_name", 
                                                                     "cleaner_transcript_biotype", "hESC_mean",
                                                                      "endo_mean", "endo_hESC_log2fc",
                                                                     "qval_hESC_endo", "CRISPhieRmix_FDR"]].head(10)


# ## 7. mark hits in all RNA-seq data

# In[33]:


no_na = data_w_seq[~pd.isnull(data_w_seq["qval_hESC_endo"])]
no_na = no_na[~no_na["qval_hESC_endo"].str.contains("NA")]
no_na["qval_log10_hESC_endo"] = -np.log10(no_na["qval_hESC_endo"].astype(float))
len(no_na)


# In[34]:


no_na = no_na[~pd.isnull(no_na["CRISPhieRmix_FDR"])]
len(no_na)


# In[35]:


fig = plt.figure(figsize=(1.75, 1.5))

ncRNA = no_na[no_na["ctrl_status"] == "experimental"]
mRNA = no_na[no_na["ctrl_status"] == "control"]
de = no_na[no_na["ctrl_status"] == "mRNA"]

ncRNA_hits = ncRNA[ncRNA["is_hit"] == "hit"]
ctrl_hits = mRNA[mRNA["is_hit"] == "hit"]

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
#plt.savefig("Fig5J.pdf", bbox_inches="tight", dpi="figure")


# In[36]:


fig = plt.figure(figsize=(1.25, 1.75))

no_hits = no_na[no_na["is_hit"] == "stringent no hit"]

ncRNA_hits = ncRNA[ncRNA["is_hit"] == "hit"]
ctrl_hits = mRNA[mRNA["is_hit"] == "hit"]
de_hits = de[de["is_hit"] == "hit"]

ax = sns.regplot(x="endo_hESC_log2fc", y="qval_log10_hESC_endo", data=no_hits, fit_reg=False, 
                 color="darkgray", scatter_kws={"s": 10, "edgecolors": "darkgray", "linewidths": 0.5})
sns.regplot(x="endo_hESC_log2fc", y="qval_log10_hESC_endo", data=ctrl_hits, fit_reg=False, 
            color=sns.color_palette()[2], scatter_kws={"s": 10, "edgecolors": "black", "linewidths": 0.5}, ax=ax)
sns.regplot(x="endo_hESC_log2fc", y="qval_log10_hESC_endo", data=ncRNA_hits, fit_reg=False, 
            color="black", scatter_kws={"s": 12, "edgecolors": "white", "linewidths": 0.5}, ax=ax)
sns.regplot(x="endo_hESC_log2fc", y="qval_log10_hESC_endo", data=de_hits, fit_reg=False, 
            color="white", scatter_kws={"s": 12, "edgecolors": "black", "linewidths": 0.5}, ax=ax)


plt.xlabel("log2(endoderm/hESC)")
plt.ylabel("negative log10 q value")
# plt.ylim((-0.1, 4))
plt.xlim((-10, 10))
plt.axhline(y=-np.log10(0.05), linestyle="dashed", color="black", linewidth=1)
fig.savefig("Fig5G.pdf", bbox_inches="tight", dpi="figure")


# In[37]:


# fisher's exact (all biotypes)
tmp = no_na[no_na["is_hit"].isin(["stringent no hit", "hit"])]
tmp_node = tmp[tmp["qval_log10_hESC_endo"] <= -np.log10(0.05)]
tmp_de = tmp[tmp["qval_log10_hESC_endo"] > -np.log10(0.05)]

de_hit = len(tmp_de[tmp_de["is_hit"] == "hit"])
de_nohit = len(tmp_de[tmp_de["is_hit"] == "stringent no hit"])
node_hit = len(tmp_node[tmp_node["is_hit"] == "hit"])
node_nohit = len(tmp_node[tmp_node["is_hit"] == "stringent no hit"])

stats.fisher_exact([[de_hit, de_nohit], [node_hit, node_nohit]])


# In[38]:


# proportion of lncRNA hits w/in classes
tmp_node_lnc = tmp_node[tmp_node["ctrl_status"] == "experimental"]
tmp_de_lnc = tmp_de[tmp_de["ctrl_status"] == "experimental"]

node_lnc_hit = len(tmp_node_lnc[tmp_node_lnc["is_hit"] == "hit"])
node_lnc_nohit = len(tmp_node_lnc[tmp_node_lnc["is_hit"] == "stringent no hit"])
de_lnc_hit = len(tmp_de_lnc[tmp_de_lnc["is_hit"] == "hit"])
de_lnc_nohit = len(tmp_de_lnc[tmp_de_lnc["is_hit"] == "stringent no hit"])

p_hit_lnc_de = de_lnc_hit/(de_lnc_hit+de_lnc_nohit)
p_hit_lnc_node = node_lnc_hit/(node_lnc_hit+node_lnc_nohit)

print("%% of hits within differentially expressed lncRNAs: %s" % p_hit_lnc_de)
print("%% of hits within non-differentially expressed lncRNAs: %s" % p_hit_lnc_node)


# ## 6. write file

# In[39]:


f = "../../../data/02__screen/02__enrichment_data/enrichment_values.with_rna_seq.UPDATED.txt"


# In[40]:


data_w_seq.columns


# In[41]:


data_w_seq = data_w_seq[["group_id", "ctrl_status", "gene_name", "gene_id", "transcript_name",
                         "transcript_id", "cleaner_transcript_biotype", "cleaner_gene_biotype", "CRISPhieRmix_FDR", 
                         "is_hit", "effect_size",
                         "csf", "hESC_mean", "endo_mean", "qval_hESC_endo", "endo_hESC_log2fc",
                         "endo_hESC_abslog2fc"]]
data_w_seq.head()


# In[42]:


data_w_seq = data_w_seq.sort_values(by="effect_size", ascending=False)
data_w_seq.to_csv(f, sep="\t", index=False)


# ## 7. investigate expression of the 6 hits in cluster 1
# FOXD3-AS1, LAMTOR5-AS1, VLDLR-AS1, HOXC-AS1, MEG3, LINC00623

# In[43]:


data_w_seq.columns


# In[44]:


tmp = data_w_seq[data_w_seq["gene_name"].isin(["FOXD3-AS1", "LAMTOR5-AS1", "VLDLR-AS1", "HOXC-AS1", "MEG3",
                                                "LINC00623"])]
tmp = tmp[tmp["is_hit"] == "hit"]
tmp.sort_values(by=["endo_hESC_abslog2fc", "gene_name"], ascending=False)


# In[45]:


order = ["FOXD3-AS1-004", "VLDLR-AS1-004", "MEG3-018", "LAMTOR5-AS1-022", "LAMTOR5-AS1-006", "LAMTOR5-AS1-023",
         "LAMTOR5-AS1-014", "LAMTOR5-AS1-031", "LAMTOR5-AS1-020", "LINC00623-013", "HOXC-AS1-002"]

fig = plt.figure(figsize=(2.2, 2))
ax = sns.barplot(data=tmp, x="transcript_name", y="endo_hESC_log2fc", order=order, color="dimgray")
ax.set_xlabel("")
ax.set_xticklabels(order, rotation=40, ha="right", va="top")
ax.set_ylabel("differential expression\nlog2(endoderm/hESC)")
ax.axhline(y=0, color="black", linestyle="dashed")
ax.set_ylim((-2, 1))

fig.savefig("Fig7A.pdf", dpi="figure", bbox_inches="tight")


# In[ ]:




