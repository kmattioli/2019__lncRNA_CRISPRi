#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import gseapy
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
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
from classify_utils import *

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
mpl.rcParams['figure.autolayout'] = False


# In[2]:


sns.set(**PAPER_PRESET)
fontsize = PAPER_FONTSIZE


# ## variables

# In[3]:


expr_dir = "../../../misc/07__ENCODE_expr"


# In[4]:


gene_sub = "w_gene"
tx_sub = "tx_only"


# In[5]:


# seq_counts_f = "../../../data/00__rna_seq/00__sleuth_results/sleuth_abundances_raw_counts.TRANSCRIPTS_WITH_GENE_ISOFORM.txt"


# In[6]:


gene_map_f = "../../../misc/00__gene_metadata/gencode.v25lift37.GENE_ID_TRANSCRIPT_ID_MAP.with_DIGIT.fixed.txt"


# In[7]:


rna_seq_f = "../../../data/00__rna_seq/01__processed_results/rna_seq_results.tsv"


# ## 1. import data

# In[8]:


files = [x[2] for x in os.walk(expr_dir)]
samps = set([x.split(".")[0] for x in files[0] if "kallisto" in x])
samps


# In[9]:


for i, samp in enumerate(samps):
    print(samp)
    tx_path = "%s/%s.%s.kallisto.tsv" % (expr_dir, samp, tx_sub)
    gene_path = "%s/%s.%s.kallisto.tsv" % (expr_dir, samp, gene_sub)
    if i == 0:
        tx_df = pd.read_table(tx_path, sep="\t")[["target_id", "tpm"]]
        tx_df.columns = ["target_id", samp]
        gene_df = pd.read_table(gene_path, sep="\t")[["target_id", "tpm"]]
        gene_df.columns = ["target_id", samp]
    else:
        tx_tmp = pd.read_table(tx_path, sep="\t")[["target_id", "tpm"]]
        tx_tmp.columns = ["target_id", samp]
        gene_tmp = pd.read_table(gene_path, sep="\t")[["target_id", "tpm"]]
        gene_tmp.columns = ["target_id", samp]
        
        tx_df = tx_df.merge(tx_tmp, on="target_id")
        gene_df = gene_df.merge(gene_tmp, on="target_id")

tx_df.head()


# In[10]:


# seq_counts = pd.read_table(seq_counts_f).reset_index()
# seq_counts.head()


# In[11]:


gene_map = pd.read_table(gene_map_f, header=None)
gene_map.columns = ["gene_id", "transcript_id"]
gene_map.head()


# In[12]:


rna_seq = pd.read_table(rna_seq_f)
print(len(rna_seq))
rna_seq.head()


# ## 2. sep gene & transcripts

# In[13]:


gene_counts = gene_df[gene_df["target_id"].str.startswith("ENSG")]
print(len(gene_counts))
gene_counts.sample(5)


# In[14]:


gene_counts = gene_counts.append(gene_df[gene_df["target_id"].str.contains("DIGIT::")])
print(len(gene_counts))


# In[15]:


transcript_counts = gene_df[gene_df["target_id"].str.startswith("ENST")]
print(len(transcript_counts))
transcript_counts.sample(5)


# In[16]:


transcript_counts = transcript_counts.append(gene_df[gene_df["target_id"] == "DIGIT"])
print(len(transcript_counts))


# ## 4. make 1 dataframe with gene/transcript counts mapped to gene ids

# In[17]:


gene_counts["gene_id"] = gene_counts["target_id"].str.split("::", expand=True)[0]
gene_counts["type"] = "gene"
gene_counts.head()


# In[18]:


transcript_counts["transcript_id"] = transcript_counts["target_id"]
transcript_counts["type"] = "transcript"
transcript_counts.head()


# In[19]:


# cols = ["index", "type", "hESC_rep1", "hESC_rep2", "endo_rep1", "endo_rep2", "meso_rep1", "meso_rep2"]


# In[20]:


tot_counts = gene_counts.append(transcript_counts)
tot_counts["id"] = tot_counts["target_id"].str.split("::", expand=True)[0]
tot_counts.drop(["gene_id", "transcript_id"], axis=1, inplace=True)
tot_counts.sample(5)


# In[21]:


tot_counts = tot_counts.merge(gene_map, left_on="id", right_on="transcript_id", how="left")
tot_counts.sample(5)


# In[22]:


def fill_gene_id(row):
    if pd.isnull(row["gene_id"]):
        return row["id"]
    else:
        return row["gene_id"]
    
tot_counts["gene_id"] = tot_counts.apply(fill_gene_id, axis=1)
tot_counts.sample(5)


# In[23]:


cols = ["id", "gene_id", "type"]
cols.extend(samps)

tot_counts = tot_counts[cols]
tot_counts = tot_counts.sort_values(by="gene_id")
tot_counts.head()


# ## 5. sum counts per gene (incl. gene isoform and all transcripts)

# In[24]:


sum_counts = tot_counts.groupby("gene_id")[list(samps)].agg("sum").reset_index()
sum_counts.head()


# ## 6. calculate splicing efficiency

# In[25]:


len(gene_counts)


# In[26]:


len(sum_counts)


# In[27]:


data = gene_counts.merge(sum_counts, on="gene_id", suffixes=("_gene", "_tot"))
data.head()


# In[28]:


for samp in samps:
    data["%s_eff" % samp] = 1-((data["%s_gene" % samp]+1)/(data["%s_tot" % samp]+1))
data.head()


# In[29]:


cells = set([x.split("_")[0] for x in samps])
cells


# In[30]:


for cell in cells:
    sub_cols = ["%s_eff" % x for x in samps if cell in x]
    print("cell: %s, samps: %s" % (cell, sub_cols))
    data["%s_eff_mean" % cell] = data[sub_cols].mean(axis=1)
    sub_cols = ["%s_tot" % x for x in samps if cell in x]
    data["%s_exp_mean" % cell] = data[sub_cols].mean(axis=1)
    
    # also record the ratio of expr & splicing efficiency (eff * expr)
    data["%s_eff_ratio" % cell] = data["%s_eff_mean" % cell] * data["%s_exp_mean" % cell]
data.head()


# In[31]:


data["gene_id"] = data.target_id.str.split("::", expand=True)[0]

clean_cols = ["gene_id"]
clean_cols.extend([x for x in data.columns if "_mean" in x or "_ratio" in x])
data = data[clean_cols]
data.head()


# ## 7. merge w/ biotype info

# In[32]:


biotypes = rna_seq[["gene_id", "gene_name", "csf", "cleaner_gene_biotype"]].drop_duplicates()
print(len(biotypes))
biotypes.head()


# In[33]:


data = data.merge(biotypes, on="gene_id", how="left")
print(len(data))
data.sample(5)


# ## 9. some QC plots

# In[34]:


data.cleaner_gene_biotype.value_counts()


# In[35]:


order = ["protein_coding", "intergenic", "promoter_overlap", "transcript_overlap", "gene_nearby"]
pal = {"protein_coding": sns.color_palette("deep")[0], "intergenic": "firebrick", "promoter_overlap": "firebrick",
       "transcript_overlap": "firebrick", "gene_nearby": "firebrick"}


# In[36]:


sorted(list(cells))


# In[37]:


fig, axarr = plt.subplots(figsize=(6, 4), nrows=4, ncols=3, sharex=True)

c = 0
for i in range(4):
    for j in range(3):
        col = sorted(list(cells))[c]

        sub = data[data["%s_exp_mean" % col] >= 0.1]
        ax = axarr[i, j]

        sns.boxplot(data=sub, x="cleaner_gene_biotype", y="%s_eff_mean" % col, order=order, palette=pal,
                    flierprops=dict(marker='o', markersize=3), ax=ax)
        mimic_r_boxplot(ax)

        ax.set_xlabel("")
        ax.set_ylabel(col)
        ax.set_xticklabels(["protein-coding", "intergenic", "promoter overlap", "transcript overlap", "gene nearby"],
                           rotation=50, ha="right", va="top")
        c += 1

plt.subplots_adjust(wspace=0.4)
plt.text(0.05, 0.5, 'splicing efficiency in:',
         horizontalalignment='right',
         verticalalignment='center',
         rotation='vertical',
         transform=plt.gcf().transFigure)

fig.savefig("splicing_eff_boxplot.pdf", dpi="figure", bbox_inches="tight")


# In[38]:


eff_cols = [x for x in data.columns if "eff" in x and "ratio" not in x]
exp_cols = [x for x in data.columns if "exp" in x]
ratio_cols = [x for x in data.columns if "ratio" in x]

data["max_eff"] = data[eff_cols].max(axis=1)
data["max_exp"] = data[exp_cols].max(axis=1)
data["max_ratio"] = data[ratio_cols].max(axis=1)
data.head()


# ## 10. write file

# In[39]:


data_f = "../../../misc/08__model_features/gene_splicing_efficiency.with_DIGIT.txt"
data.to_csv(data_f, index=False, sep="\t")


# In[ ]:




