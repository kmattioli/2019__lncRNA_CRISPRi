#!/usr/bin/env python
# coding: utf-8

# # 04__results
# 
# in this notebook, i aggregate results from CRISPhieRmix and plot.
# 
# figures in this notebook:
# - Fig 3E: plot showing transcript-level enrichments vs. CRISPhieRmix FDR
# 
# supplemental tables in this notebook:
# - Table S3: enrichment scores & FDRs for all targeted TSSs

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

from ast import literal_eval
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


# ## functions

# ## variables

# In[3]:


data_filt_f = "../../../data/02__screen/02__enrichment_data/data_filt.tmp"


# In[4]:


crisphie_f = "../../../data/02__screen/02__enrichment_data/CRISPhieRmix.txt"


# ## 1. import data

# In[5]:


data_filt = pd.read_table(data_filt_f, sep="\t")
data_filt.head()


# In[6]:


crisphie = pd.read_table(crisphie_f, sep="\t")
len(crisphie)


# ## 2. parse results from CRISPhieRmix

# In[7]:


# if using gene level to group
# crisphie["gene_name"] = crisphie["groups"].str.split("__", expand=True)[0]
# crisphie["ctrl_status_fixed"] = crisphie["groups"].str.split("__", expand=True)[1]

# if using transcript level
crisphie["transcript_name"] = crisphie["groups"].str.split(";;;", expand=True)[0]
crisphie["group_id"] = crisphie["groups"].str.split(";;;", expand=True)[1]
crisphie["ctrl_status"] = crisphie["groups"].str.split(";;;", expand=True)[2]
crisphie["transcript_biotype_status"] = crisphie["groups"].str.split(";;;", expand=True)[3]
crisphie["n_sgRNA"] = crisphie["groups"].str.split(";;;", expand=True)[4].astype(int)
crisphie.head()


# In[8]:


crisphie[crisphie["groups"].str.contains("B.1-001")]


# ## 3. calculate effect size -- estimate as top 3 sgRNA median

# In[9]:


# need to remove ' from data_filt to merge with the crisphiermix table
# looks like R removes these by default
data_filt["gene_name"] = data_filt["gene_name"].str.replace("'", '')
data_filt["transcript_name"] = data_filt["transcript_name"].str.replace("'", '')
data_filt[data_filt["gene_name"].str.contains("AC007128.1")][["gene_name", "l2fc"]]


# In[10]:


data_filt = data_filt.sort_values(by=["group_id", "ctrl_status", "l2fc"], ascending=False)
eff_size = data_filt.groupby(["group_id", "ctrl_status"]).head(3)
eff_size = eff_size.groupby(["group_id", "ctrl_status"])["l2fc"].agg("median").reset_index()
eff_size.head()


# ## 4. merge effect size w/ CRISPhieRmix FDR

# In[11]:


crisphie = crisphie.merge(eff_size, on=["group_id", "ctrl_status"], how="left")
print(len(crisphie))
crisphie.sort_values(by="FDR").head()


# ## 5. plot results

# In[12]:


crisphie["neg_log_FDR"] = -np.log10(crisphie["FDR"]+1e-12)
sig = crisphie[crisphie["FDR"] < 0.1]
not_sig = crisphie[crisphie["FDR"] >= 0.1]
ctrl = crisphie[crisphie["ctrl_status"] == "control"]
exp = crisphie[crisphie["ctrl_status"] == "experimental"]
mrna = crisphie[crisphie["ctrl_status"] == "mRNA"]
scr = crisphie[crisphie["ctrl_status"] == "scramble"]


# In[13]:


sig.ctrl_status.value_counts()


# In[14]:


sig.transcript_biotype_status.value_counts()


# In[21]:


uniq_txs = list(exp[exp["FDR"] < 0.1]["transcript_name"].unique())
print(len(uniq_txs))

genes = []
for tx in uniq_txs:
    if tx.startswith("["):
        gene = tx.split(",")[0]
        gene = gene[1:-4]
        genes.append(gene)
    else:
        gene = tx[0:-4]
        genes.append(gene)

uniq_loci = set(genes)
print(len(uniq_loci))


# In[15]:


nopromoverlap = sig[(sig["transcript_biotype_status"] != "promoter_overlap") & (sig["ctrl_status"] == "experimental")]
len(nopromoverlap)


# In[16]:


nopromoverlap.sort_values(by="FDR")[["transcript_name", "transcript_biotype_status"]]


# In[17]:


promoverlap = sig[(sig["transcript_biotype_status"] == "promoter_overlap") & (sig["ctrl_status"] == "experimental")]
promoverlap[["transcript_name", "transcript_biotype_status", "FDR", "l2fc"]]


# In[18]:


pal = {"control": sns.color_palette()[2], "experimental": "black", "scramble": "gray"}
sns.palplot(pal.values())


# In[19]:


fig = plt.figure(figsize=(2,2))

ax = plt.gca()
ax.scatter(exp["l2fc"], exp["neg_log_FDR"], color="slategray", s=15, alpha=0.5)
ax.scatter(nopromoverlap["l2fc"], nopromoverlap["neg_log_FDR"], color="slategray", edgecolors="black",
           linewidths=0.5, s=15, alpha=1)
ax.scatter(mrna["l2fc"], mrna["neg_log_FDR"], color="white", s=15, alpha=1, edgecolors="black", linewidths=0.5)
ax.scatter(ctrl["l2fc"], ctrl["neg_log_FDR"], color=pal["control"], s=20, alpha=1, edgecolors="black",
           linewidths=0.5)
ax.scatter(scr["l2fc"], scr["neg_log_FDR"], color="gray", s=15, alpha=0.5)
ax.axhline(y=1, linestyle="dashed", color="black")
ax.set_xlabel("transcript enrichment score")
ax.set_ylabel("-log10(CRISPhieRmix FDR)")
ax.set_xscale('symlog')

# annotate #s
n_sig = len(sig)
n_not_sig = len(not_sig)
ax.text(0.05, 0.95, "FDR < 0.1\n(n=%s)" % (n_sig), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)

fig.savefig("Fig3E.pdf", dpi="figure", bbox_inches="tight")


# In[20]:


crisphie[crisphie["transcript_name"].str.contains("FOXD3")]


# In[21]:


crisphie[crisphie["transcript_name"].str.contains("DIGIT")]


# ## 9. write final file(s)

# In[22]:


def is_hit(row):
    if row["FDR"] < 0.1:
        return "hit"
    else:
        return "no hit"
    
crisphie["hit_status"] = crisphie.apply(is_hit, axis=1)


# In[23]:


supp = crisphie[["group_id", "transcript_name", "ctrl_status", "transcript_biotype_status", "FDR", "l2fc", 
                 "hit_status", "n_sgRNA"]]
supp.columns = ["group_id", "transcript_name", "ctrl_status", "transcript_biotype_status", "CRISPhieRmix_FDR", 
                "effect_size", "hit_status", "n_sgRNA"]
supp = supp.sort_values(by="CRISPhieRmix_FDR")
print(len(supp))
supp.head()


# In[24]:


supp.hit_status.value_counts()


# In[25]:


supp.sort_values(by="CRISPhieRmix_FDR", ascending=True).head(10)


# In[26]:


f = "../../../data/02__screen/02__enrichment_data/SuppTable_S3.CRISPhieRmix_results.txt"
supp.to_csv(f, sep="\t", index=False)


# In[ ]:



