#!/usr/bin/env python
# coding: utf-8

# # 05__validation
# 
# in this notebook, i compare our screen results to individual validations performed on 22 sgRNAs.
# 
# figures in this notebook:
# - Fig 4A: scatter plot showing the comparison between screen results and individual validation results
# - Fig 4B: RT-qPCR results for a subset of mRNAs
# - Fig 4C: RT-qPCR results for a subset of lncRNAs
# 
# tables in this notebook:
# - Table S4: validation results + screen results for these validated sgRNAs

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
from os import walk
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


data_f = "../../../data/02__screen/02__enrichment_data/SuppTable_S2.sgRNA_results.txt"


# In[4]:


crisphie_f = "../../../data/02__screen/02__enrichment_data/SuppTable_S3.CRISPhieRmix_results.txt"


# In[5]:


validation_data_f = "../../../data/02__screen/03__validation_data/validation_data.xlsx"


# In[6]:


qpcr_dir = "../../../data/02__screen/03__validation_data/rt_qpcr_data"


# ## 1. import data

# In[7]:


data = pd.read_table(data_f, sep="\t")
#data["sgRNA_rank"] = data["sgRNA_rank"].astype(str)
data.head()


# In[8]:


val_data = pd.read_excel(validation_data_f)
print(len(val_data))
val_data.head()


# In[9]:


crisphie = pd.read_table(crisphie_f, sep="\t")
print(len(crisphie))
crisphie.head()


# In[10]:


files = []
for (dirpath, dirnames, filenames) in walk(qpcr_dir):
    files.extend(filenames)
    break
print(len(files))
files


# ## 2. merge validation data w/ screen data

# In[11]:


data_sub = data[["sgRNA", "sgRNA_l2fc", "tss_id"]]


# In[12]:


val_data = val_data.merge(data_sub, left_on="guide_sequence", right_on="sgRNA", how="left")


# In[13]:


val_data = val_data.merge(crisphie[["group_id", "transcript_biotype_status", "hit_status"]], 
                          left_on="tss_id", right_on="group_id", how="left")
val_data.drop(["sgRNA", "group_id"], axis=1, inplace=True)
val_data


# ## 3. calculate validation data "enrichment" (guide enrichment normalized by scrambled enrichment)

# In[14]:


val_data["guide_undiff_enrich"] = val_data["guide_undiff"]/val_data["guide_diff"]
val_data["scram_undiff_enrich"] = val_data["scram_undiff"]/val_data["scram_diff"]
val_data["val_score"] = np.log2(val_data["guide_undiff_enrich"]/val_data["scram_undiff_enrich"])
val_data.sort_values(by="sgRNA_l2fc", ascending=False)


# In[15]:


# remove the 2 sgRNAs that weren't in the screen
val_data = val_data[~pd.isnull(val_data["hit_status"])]
val_data


# ## 4. plot correlation -- all validated sgRNAs

# In[16]:


fig = plt.figure(figsize=(2, 2))
ax = sns.regplot(data=val_data, x="sgRNA_l2fc", y="val_score", color="black")

hits = val_data[val_data["hit_status"] == "hit"]
nonhits = val_data[val_data["hit_status"] == "no hit"]

pc_hits = hits[hits["transcript_biotype_status"] == "protein_coding"]
pc_nonhits = nonhits[nonhits["transcript_biotype_status"] == "protein_coding"]

lnc_hits = hits[hits["transcript_biotype_status"] != "protein_coding"]
lnc_nonhits = nonhits[nonhits["transcript_biotype_status"] != "protein_coding"]

ax.scatter(pc_hits["sgRNA_l2fc"], pc_hits["val_score"], color=sns.color_palette()[2], zorder=11,
           edgecolors="black", linewidth=0.5)
ax.scatter(pc_nonhits["sgRNA_l2fc"], pc_nonhits["val_score"], color="white", 
           edgecolors=sns.color_palette()[2], linewidth=0.5, zorder=11)

ax.scatter(lnc_hits["sgRNA_l2fc"], lnc_hits["val_score"], color="black", zorder=11)
ax.scatter(lnc_nonhits["sgRNA_l2fc"], lnc_nonhits["val_score"], color="white", 
           edgecolors="black", linewidth=0.5, zorder=11)

ax.set_xlabel("screen sgRNA enrichment score")
ax.set_ylabel("validation sgRNA enrichment score")

no_nan = val_data[~pd.isnull(val_data["sgRNA_l2fc"])]
r, p = stats.spearmanr(no_nan["sgRNA_l2fc"], no_nan["val_score"])
ax.text(0.05, 0.95, "r = %s\np = %s\nn= %s" % ((round(r, 2), round(p, 4), len(no_nan))), 
        ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)

fig.savefig("Fig4A.pdf", dpi="figure", bbox_inches="tight")


# In[17]:


val_data[val_data["gene_name"] == "RP11-222K16.2"][["guide_num", "guide_sequence", "sgRNA_l2fc", "val_score"]]


# ## 5. load RT-qPCR data

# In[18]:


def row_type(row):
    if "scrm" in row["variable"]:
        return "scramble"
    else:
        return "sgRNA"


# In[19]:


qpcr_dict = {}

for f in files:
    n = f.split("__")[0]
    sgrna = f.split("__")[2].split(".")[0]
    df = pd.read_table("%s/%s" % (qpcr_dir, f))
    df.columns = ["gene", "scrm_val", "scrm_yerr1", "scrm_yerr2", "sgrna_val", "sgrna_yerr1", "sgrna_yerr2"]
    melt = pd.melt(df, id_vars=["gene"])
    melt["type"] = melt.apply(row_type, axis=1)
    qpcr_dict["%s__%s" % (n.upper(), sgrna)] = melt


# In[20]:


qpcr_dict["SOX17__1"]


# ## 6. make RT-qPCR plots
# for SOX17, EOMES, RP11-120D5.1, RP11-222K16.2 (figure 4)

# In[21]:


genes = ["SOX17", "EOMES", "RP11-120D5.1", "RP11-222K16.2"]
plot_df = pd.DataFrame()

for key in qpcr_dict:
    gene = key.split("__")[0]
    if gene in genes:
        df = qpcr_dict[key]
        df["info"] = key
        plot_df = plot_df.append(df)


# In[22]:


vals = plot_df[plot_df["variable"].str.contains("val")]
yerrs = plot_df[plot_df["variable"].str.contains("yerr1")]
print(len(vals))
print(len(yerrs))


# In[23]:


order = ["SOX17__1", "SOX17__4", "EOMES__1", "EOMES__10"]
pal = {"scrm_val": "darkgray", "sgrna_val": sns.color_palette()[2]}


# In[24]:


fig = plt.figure(figsize=(2.5, 2))

ax = sns.barplot(data=vals, x="info", y="value", hue="variable", order=order, palette=pal)
ax.set_xlabel("")
ax.set_ylabel("fold change")
ax.get_legend().remove()
ax.set_ylim((0, 1.5))
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# plot errors
xs = [-0.2, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 3.2]
c = 0
for i, x in enumerate(xs):
    if i % 2 == 0:
        name = order[c]
        c += 1
    subvals = vals[vals["info"] == name]
    subyerrs = yerrs[yerrs["info"] == name]
    if i % 2 == 0: #scrambled sgRNA
        val_ = subvals[subvals["type"] == "scramble"]["value"].iloc[0]
        yerr_ = subyerrs[subyerrs["type"] == "scramble"]["value"].iloc[0]
    else:
        val_ = subvals[subvals["type"] == "sgRNA"]["value"].iloc[0]
        yerr_ = subyerrs[subyerrs["type"] == "sgRNA"]["value"].iloc[0]
    ax.plot([x, x], [val_ - yerr_, val_ + yerr_], color="black")
    ax.plot([x-0.1, x+0.1], [val_ - yerr_, val_ - yerr_], color="black")
    ax.plot([x-0.1, x+0.1], [val_ + yerr_, val_ + yerr_], color="black")
    
fig.savefig("Fig4B.pdf", dpi="figure", bbox_inches="tight")


# In[25]:


order = ["RP11-120D5.1__1", "RP11-120D5.1__2", "RP11-222K16.2__3", "RP11-222K16.2__9"]
pal = {"scrm_val": "darkgray", "sgrna_val": "dimgray"}


# In[26]:


fig = plt.figure(figsize=(2.5, 2))

ax = sns.barplot(data=vals, x="info", y="value", hue="variable", order=order, palette=pal)
ax.set_xlabel("")
ax.set_ylabel("fold change")
ax.get_legend().remove()
ax.set_ylim((0, 1.5))
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# plot errors
xs = [-0.2, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 3.2]
c = 0
for i, x in enumerate(xs):
    if i % 2 == 0:
        name = order[c]
        c += 1
    subvals = vals[vals["info"] == name]
    subyerrs = yerrs[yerrs["info"] == name]
    if i % 2 == 0: #scrambled sgRNA
        val_ = subvals[subvals["type"] == "scramble"]["value"].iloc[0]
        yerr_ = subyerrs[subyerrs["type"] == "scramble"]["value"].iloc[0]
    else:
        val_ = subvals[subvals["type"] == "sgRNA"]["value"].iloc[0]
        yerr_ = subyerrs[subyerrs["type"] == "sgRNA"]["value"].iloc[0]
    ax.plot([x, x], [val_ - yerr_, val_ + yerr_], color="black")
    ax.plot([x-0.1, x+0.1], [val_ - yerr_, val_ - yerr_], color="black")
    ax.plot([x-0.1, x+0.1], [val_ + yerr_, val_ + yerr_], color="black")
    
fig.savefig("Fig4C.pdf", dpi="figure", bbox_inches="tight")


# ## 7. join all data, incl rt-qpcr

# In[27]:


tmp1 = plot_df[plot_df["variable"].isin(["scrm_val", "scrm_yerr1", 
                                         "sgrna_val", "sgrna_yerr1"])][["info", "variable", "value"]]
tmp1 = tmp1.pivot(index="info", columns="variable").reset_index()
tmp1


# In[28]:


genes = [x for x in val_data.gene_name.unique() if x not in ["SOX17", "EOMES", "RP11-120D5.1", "RP11-222K16.2"]]
rem_plot_df = pd.DataFrame()

for key in qpcr_dict:
    gene = key.split("__")[0]
    if gene in genes:
        df = qpcr_dict[key]
        df["info"] = key
        rem_plot_df = rem_plot_df.append(df)
rem_plot_df.head()


# In[29]:


tmp2 = rem_plot_df[rem_plot_df["variable"].isin(["scrm_val", "scrm_yerr1", 
                                                 "sgrna_val", "sgrna_yerr1"])][["info", "variable", "value"]]
tmp2 = tmp2.pivot(index="info", columns="variable").reset_index()
tmp2


# In[30]:


qpcr_df = tmp1.append(tmp2)
qpcr_df["gene_name"] = qpcr_df["info"].str.split("__", expand=True)[0]
qpcr_df["guide_num"] = qpcr_df["info"].str.split("__", expand=True)[1].astype(int)


# In[31]:


val_data = val_data.merge(qpcr_df, how="left", on=["gene_name", "guide_num"])
val_data.head()


# ## 5. write updated file

# In[32]:


val_data.columns


# In[33]:


val_data = val_data.iloc[:, [0, 1, 2, 8, 9, 3, 4, 5, 6, 10, 16, 13, 20, 21, 18, 19]]
val_data.columns = ["gene_name", "sgRNA_num", "sgRNA", "FACS_date", "RTqPCR_date", "sgRNA_FACS_pUndiff",
                    "scrambled_FACS_pUndiff", "sgRNA_FACS_pDiff", "scrambled_FACS_pDiff", "sgRNA_screen_l2fc",
                    "sgRNA_validation_score", "screen_hit_status", "sgRNA_RTqPCR_mean", "sgRNA_RTqPCR_error",
                    "scrambled_RTqPCR_mean", "scrambled_RTqPCR_error"]
val_data.head()


# In[34]:


# update sgRNA numbers so they reflect what's in the corr plot
gene_order = ["SOX17", "EOMES", "RP11-120D5.1", "RP11-222K16.2", "FOXA2", "GATA6", 'ACVR2B-AS1', 'DIGIT', 
              'KB-1440D3.14', 'LINC00479', 'PVT1', 'RP11-23F23.2', 'RP3-508I15.9']
gene_order_df = pd.DataFrame(gene_order).reset_index()
gene_order_df.columns = ["rank", "gene_name"]
gene_order_df


# In[35]:


val_data = val_data.merge(gene_order_df, on="gene_name")
val_data = val_data.sort_values(by=["rank", "sgRNA_num"], ascending=True)


# In[36]:


val_data.reset_index(inplace=True)
val_data.reset_index(inplace=True)
val_data["sgRNA_num"] = val_data["level_0"] + 1
val_data.drop(["level_0", "index", "rank"], axis=1, inplace=True)
val_data


# In[37]:


val_data.to_csv("../../../data/02__screen/03__validation_data/SuppTable_S4.validation_results.txt", sep="\t",
               index=False)


# In[ ]:




