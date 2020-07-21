#!/usr/bin/env python
# coding: utf-8

# # 02__cleaner_index
# 
# in this notebook, i clean up my original index file to add some additional columns and update some information (e.g. biotype classifications).
# 
# figures in this notebook:
# - Fig S4F_1: count of biotypes in original library design

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
from enrich_utils import *

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
mpl.rcParams['figure.autolayout'] = False


# In[2]:


sns.set(**PAPER_PRESET)
fontsize = PAPER_FONTSIZE


# ## variables

# In[3]:


index_f = "../../../data/01__design/02__final_sgRNAs/crispri_with_primers.txt"


# In[4]:


# rna seq file has biotype information
rna_seq_f = "../../../data/00__rna_seq/01__processed_results/rna_seq_results.tsv"


# In[5]:


endo_ctrls_validated_f = "../../../misc/04__pos_ctrls/endo_ctrls_validated.updated.txt"


# ## 1. import data

# In[6]:


index = pd.read_table(index_f, sep="\t")
print(len(index))
index.head()


# In[7]:


rna_seq = pd.read_table(rna_seq_f, sep="\t")
print(len(rna_seq))
rna_seq.head()


# In[8]:


endo_ctrls_validated = pd.read_table(endo_ctrls_validated_f, sep="\t", header=None)
endo_ctrls_validated.head()


# ## 2. un de-dupe index file such that each transcript has its own row
# since group ids may target multiple transcripts, but biotypes are assigned at transcript level

# In[9]:


# fix index ID columns that can have duplicates in them due to previous aggregation
index["gene_id"] = index.apply(fix_id_dupes, column="gene_id", axis=1)
index["gene_name"] = index.apply(fix_id_dupes, column="gene_name", axis=1)
index["transcript_id"] = index.apply(fix_id_dupes, column="transcript_id", axis=1)
index["transcript_name"] = index.apply(fix_id_dupes, column="transcript_name", axis=1)
index = index.drop_duplicates()
len(index.transcript_id.unique())


# In[10]:


index_sub = index[["tss_id_hg38", "gene_name", "gene_id", "transcript_name", "transcript_id"]].drop_duplicates()
print(len(index_sub))
print(len(index_sub.tss_id_hg38.unique()))
index_sub.head()


# In[11]:


print(len(index_sub))
index_sub_split = tidy_split(index_sub, "transcript_id", sep=",", keep=False)
print(len(index_sub_split))
index_sub_split["transcript_id"] = index_sub_split["transcript_id"].str.replace('[', '')
index_sub_split["transcript_id"] = index_sub_split["transcript_id"].str.replace(']', '')
index_sub_split["transcript_id"] = index_sub_split["transcript_id"].str.replace(' ', '')
index_sub_split["transcript_id"] = index_sub_split["transcript_id"].str.replace("'", '')
index_sub_split.drop(["gene_name", "gene_id", "transcript_name"], axis=1, inplace=True)
index_sub_split.head(10)


# ## 3. join un de-duped index w/ rna seq file

# In[12]:


index_sub_split = index_sub_split.merge(rna_seq, on="transcript_id", how="left")
print(len(index_sub_split))
index_sub_split.head()


# ## 4. aggregate back to tss level: collapse csf column, biotype columns

# In[13]:


index_sub_tmp = index_sub_split.groupby("tss_id_hg38")[["csf", 
                                                        "cleaner_gene_biotype", 
                                                        "cleaner_transcript_biotype"]].agg(set).reset_index()
index_sub_tmp.head()


# In[14]:


def dedup_status(row, col):
    vals = row[col]
    if len(vals) == 1:
        return list(vals)[0]
    else:
        return "multi-targeting"
    
index_sub_tmp["csf_status"] = index_sub_tmp.apply(dedup_status, axis=1, col="csf")
index_sub_tmp["gene_biotype_status"] = index_sub_tmp.apply(dedup_status, axis=1, col="cleaner_gene_biotype")
index_sub_tmp["transcript_biotype_status"] = index_sub_tmp.apply(dedup_status, axis=1, col="cleaner_transcript_biotype")


# In[15]:


len(index_sub_tmp)


# ## 5. join back to original file

# In[16]:


index = index.merge(index_sub_tmp[["tss_id_hg38", "csf_status", "gene_biotype_status",
                                   "transcript_biotype_status"]], on="tss_id_hg38", how="left")
len(index)


# ## 6. fix control status column

# In[17]:


val_ctrls = list(endo_ctrls_validated[0])
len(val_ctrls)


# In[18]:


def fix_ctrl_status(row):
    if row.gene_name in val_ctrls:
        return "control"
    else:
        if row.gene_name == "scramble":
            return "scramble"
        elif row.gene_biotype_status == "protein_coding":
            return "mRNA"
        else:
            return "experimental"


# In[19]:


index["ctrl_status_fixed"] = index.apply(fix_ctrl_status, axis=1)


# ## 7. output counts

# In[20]:


print(" === COUNT OF GUIDES === ")
print("")
index.ctrl_status_fixed.value_counts()


# In[21]:


print(" === COUNT OF TSSS ===")
print("")
index.drop_duplicates(subset="tss_id_hg38").ctrl_status_fixed.value_counts()


# ## 8. plot biotypes

# In[22]:


index_lncrna = index[index["ctrl_status_fixed"] == "experimental"]
len(index_lncrna)


# In[23]:


index_lncrna = index_lncrna[["tss_id_hg38", "transcript_biotype_status"]].drop_duplicates()
len(index_lncrna)


# In[24]:


fig = plt.figure(figsize=(1, 1.75))

order = ["intergenic", "promoter_overlap", "transcript_overlap", "gene_nearby", "multi-targeting"]
ax = sns.countplot(data=index_lncrna, y="transcript_biotype_status", order=order, color=sns.color_palette()[0])

for p in ax.patches:
    w = p.get_width()
    y = p.get_y()
    h = p.get_height()
    
    ax.text(w + 100, y + h/2, int(w), ha="left", va="center", fontsize=fontsize) 
    
plt.xlim((0,9000))
plt.ylabel("")
plt.title("biotypes in\nsgRNA library")
fig.savefig("FigS4F_1.pdf", dpi="figure", bbox_inches="tight")


# ## 9. write cleaner index file

# In[25]:


len(index)


# In[26]:


index.head()


# In[27]:


index_f = "../../../data/01__design/02__final_sgRNAs/crispri.clean_index.txt"
index.to_csv(index_f, sep="\t", index=False)


# In[ ]:




