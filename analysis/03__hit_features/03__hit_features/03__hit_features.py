#!/usr/bin/env python
# coding: utf-8

# # 03__hit_features
# 
# in this notebook, i plot boxplots for features aggregated between stringent non hits and hits in our screen.
# 
# figures in this notebook:
# - Fig 5A-F: boxplots of most significant features
# - Fig S6: boxplots of remaining features

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import re
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
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


# In[3]:


np.random.seed(2019)


# ## variables

# In[4]:


data_f = "../../../data/03__features/SuppTable_S5.locus_features.txt"


# ## 1. import data

# In[5]:


data = pd.read_table(data_f, sep="\t")
data.head()


# ## 2. mann whitney tests for each variable

# In[6]:


all_feature_cols = ['max_eff', 'max_exp', 'gc', 'n_tss', 'n_enh', 'enh_tss_dist', 'enh_tran_dist', 
                    'DE_enh_tss_dist', 'DE_enh_tran_dist', 'prom_cons',
                    'exon_cons', 'dna_len', 'rna_len', 'n_exons', 'hESC_mean', 'endo_mean', 'endo_hESC_abslog2fc',
                    'closest_endo_snp_distance']


# In[7]:


pal = {"stringent no hit": "gray", "hit": "black"}


# In[8]:


data.is_hit.value_counts()


# In[9]:


data[data["minimal_biotype"] == "lncRNA"]["is_hit"].value_counts()


# In[10]:


data[data["minimal_biotype"] == "mRNA"]["is_hit"].value_counts()


# In[11]:


data[(data["is_hit"] == "hit") & (data["minimal_biotype"] == "mRNA")][["gene_name", "minimal_biotype", "n_tss"]]


# In[12]:


ylabels = ["max splicing efficiency", "max expression", "GC content", "# TSSs", "# nearby enhancers",
           "distance to closest enhancer\n(from TSS)", "distance to closest enhancer", 
           "distance to closest DE enhancer\n(from TSS)", "distance to closest DE enhancer",
           "TSS conservation", "exon conservation", "DNA locus length",
           "RNA transcript length", "# exons", "hESC expression", "endoderm expression", "log2(endo/hESC expression)",
           "distance to closest\n endoderm cancer SNP"]
rounds = [1, 1, 1, 0, 0, "kb", "kb", "kb", "kb", 4, 3, "kb", "bp", 0, 1, 1, 1, "kb"]

logs = [False, True, False, False, False, True, True, True, True, False, False, True, True, True, True, True, 
        False, True]
all_ys = {"lncRNA": [0.8, 18, 0.57, 1.3, 90, 100000, 100000, 1000000, 1000000, 0.15, 0.19, 50000, 3000, 
                     5, 7, 7, 1.8, 5000000],
          "mRNA": [1.02, 250, 0.54, 3.8, 90, 40000, 40000, 40000, 40000, 0.9, 1.05, 100000, 7000, 20, 
                   100, 250, 7.7, 4000000]}
plotnames = ["Fig5E", "Fig5F", "FigS6H", "Fig5C", "FigS6J", "FigS6G", None, "FigS6K", None, "FigS6I", 
             "FigS6B", "FigS6F", "FigS6E", "FigS6C", "FigS6D", "Fig5D", "FigS6A", "Fig5B"]
all_pvals = {"lncRNA": {}, "mRNA": {}}
for i, col in enumerate(all_feature_cols):
    print("===")
    print(col)
    log = logs[i]
    ylabel = ylabels[i]
    plotname = plotnames[i]
    for biotype in ["lncRNA", "mRNA"]:
        print(biotype)
        y = all_ys[biotype][i]

        fig = plt.figure(figsize=(1.25, 1.75))
        sub = data[data["minimal_biotype"] == biotype]
        ax = sns.boxplot(data=sub, x="is_hit", y=col,
                         flierprops = dict(marker='o', markersize=3), order=["stringent no hit", "hit"], 
                         palette=pal)
        mimic_r_boxplot(ax)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(["non-hit\n(stringent)", "hit"])

        # calc p-vals b/w dists
        dist1 = np.asarray(sub[sub["is_hit"] == "stringent no hit"][col])
        dist2 = np.asarray(sub[sub["is_hit"] == "hit"][col])

        dist1 = dist1[~np.isnan(dist1)]
        dist2 = dist2[~np.isnan(dist2)]

        u, pval = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
        print(pval)
        med1 = np.median(dist1)
        med2 = np.median(dist2)
        print(med1)
        
        r = rounds[i]
        if r in [1, 2, 3, 4]:
            txt1 = round(med1, r)
            txt2 = round(med2, r)
        elif r == 0:
            txt1 = int(med1)
            txt2 = int(med2)
        elif r == "bp":
            txt1 = "%s bp" % int(med1)
            txt2 = "%s bp" % int(med2)
        else:
            kb1 = med1/1000
            kb2 = med2/1000
            txt1 = "%s kb" % int(kb1)
            txt2 = "%s kb" % int(kb2)
            
        ax.text(0, med1, txt1, ha="center", va="center", fontweight="bold",
                bbox=dict(facecolor="lightgray", pad=2, edgecolor="gray"))
        ax.text(1, med2, txt2, ha="center", va="center", fontweight="bold",
                bbox=dict(facecolor="gray", pad=2, edgecolor="black"))
        
        
        tmp = all_pvals[biotype]
        tmp[col] = pval
            
        annotate_pval(ax, 0.2, 0.8, y, 0, y, pval, fontsize)

        if log:
            ax.set_yscale("symlog")
            
        if col == "n_tss":
            ax.set_ylim((-0.5, 3.2))
        if col == "prom_cons" or col == "exon_cons":
            ax.set_ylim((-0.1, 1.1))

        plt.show()
        if biotype == "lncRNA":
            if plotname != None:
                fig.savefig("%s.pdf" % plotname, dpi="figure", bbox_inches="tight")


# In[13]:


lncrna_pvals = pd.DataFrame.from_dict(all_pvals["lncRNA"], orient="index")
lncrna_pvals = lncrna_pvals.drop(["enh_tran_dist", "DE_enh_tran_dist"])
lncrna_pvals.columns = ["pval"]
lncrna_pvals["padj"] = multicomp.multipletests(lncrna_pvals["pval"], method="fdr_bh")[1]
lncrna_pvals.sort_values(by="pval")


# In[14]:


mrna_pvals = pd.DataFrame.from_dict(all_pvals["mRNA"], orient="index")
mrna_pvals.columns = ["pval"]
mrna_pvals.sort_values(by="pval")


# ## take a look at hits

# In[15]:


data.columns


# In[16]:


tmp = data[(data["is_hit"] == "hit") & (data["minimal_biotype"] == "lncRNA")][["gene_name", "cleaner_gene_biotype",
                                                                               "max_eff", "max_exp", "enh_tss_dist",
                                                                               "n_tss", "n_enh",
                                                                               "prom_cons", "exon_cons", "hESC_mean",
                                                                               "endo_mean", "endo_hESC_abslog2fc",
                                                                               "closest_endo_snp_distance",
                                                                               "DE_enh_tss_dist"]]
tmp.sort_values(by="endo_hESC_abslog2fc", ascending=False).head(10)


# In[ ]:




