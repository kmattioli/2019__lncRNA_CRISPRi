#!/usr/bin/env python
# coding: utf-8

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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from imblearn.over_sampling import SMOTE


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


data_f = "../../../data/03__features/SuppTable_ScreenGenes.txt"


# ## 1. import data

# In[5]:


data = pd.read_table(data_f, sep="\t")
data.head()


# ## 2. mann whitney tests for each variable

# In[6]:


all_feature_cols = ['max_eff', 'max_exp', 'gc', 'n_tss', 'n_enh', 'enh_dist', 'prom_cons',
                    'exon_cons', 'dna_len', 'rna_len', 'n_exons', 'hESC_mean', 'endo_mean', 'endo_hESC_abslog2fc',
                    'closest_endo_snp_distance']


# In[7]:


pal = {"no hit": "gray", "hit": "black"}


# In[8]:


data[data["minimal_biotype"] == "lncRNA"]["min_hit"].value_counts()


# In[9]:


data[data["minimal_biotype"] == "mRNA"]["min_hit"].value_counts()


# In[10]:


logs = [False, True, False, False, False, True, False, False, True, True, True, True, True, False, True]
all_ys = {"lncRNA": [0.85, 20, 0.6, 2.5, 100, 100000, 0.075, 0.15, 50000, 3000, 10, 7, 7, 1.8, 5000000],
          "mRNA": [1.02, 250, 0.54, 3.8, 90, 40000, 0.9, 1.05, 100000, 7000, 20, 100, 250, 7.7, 4000000]}
ylabels = ["max splicing efficiency", "max expression", "GC content", "# TSSs", "# nearby enhancers",
           "distance to closest enhancer", "TSS conservation", "exon conservation", "DNA locus length",
           "RNA transcript length", "# exons", "hESC expression", "endoderm expression", "log2(endo/hESC expression)",
           "distance to closest\n endoderm cancer SNP"]
plotnames = ["Fig5C", "Fig5B", "FigS6A", "Fig5D", "FigS6B", "Fig5E", "FigS6C", "FigS6D", "FigS6E", "FigS6F",
             "FigS6G", "FigS6H", "Fig5F", "Fig5G", "FigS6H"]
all_pvals = {"lncRNA": {}, "mRNA": {}}
for i, col in enumerate(all_feature_cols):
    print("===")
    print(col)
    log = logs[i]
    ylabel = ylabels[i]
    for biotype in ["lncRNA", "mRNA"]:
        print(biotype)
        y = all_ys[biotype][i]

        fig = plt.figure(figsize=(0.75, 1.5))
        sub = data[data["minimal_biotype"] == biotype]
        ax = sns.boxplot(data=sub, x="min_hit", y=col,
                         flierprops = dict(marker='o', markersize=3), order=["no hit", "hit"], 
                         palette=pal)
        mimic_r_boxplot(ax)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)

        # calc p-vals b/w dists
        dist1 = np.asarray(sub[sub["min_hit"] == "no hit"][col])
        dist2 = np.asarray(sub[sub["min_hit"] == "hit"][col])

        dist1 = dist1[~np.isnan(dist1)]
        dist2 = dist2[~np.isnan(dist2)]

        u, pval = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
        print(pval)
        
        tmp = all_pvals[biotype]
        tmp[col] = pval
            
        if pval < 0.05:
            if log:
                text_y = 0.78 * y
            else:
                text_y = 0.975 * y
        else:
            text_y = y
            
        annotate_pval(ax, 0.2, 0.8, y, 0, text_y, pval, fontsize)

        if log:
            ax.set_yscale("symlog")

        plt.show()
        if biotype == "lncRNA":
            fig.savefig("%s.pdf" % plotnames[i], dpi="figure", bbox_inches="tight")


# In[11]:


0.258/2


# In[12]:


lncrna_pvals = pd.DataFrame.from_dict(all_pvals["lncRNA"], orient="index")
lncrna_pvals.columns = ["pval"]
lncrna_pvals["padj"] = multicomp.multipletests(lncrna_pvals["pval"], method="fdr_bh")[1]
lncrna_pvals.sort_values(by="pval")


# In[13]:


mrna_pvals = pd.DataFrame.from_dict(all_pvals["mRNA"], orient="index")
mrna_pvals.columns = ["pval"]
mrna_pvals.sort_values(by="pval")

