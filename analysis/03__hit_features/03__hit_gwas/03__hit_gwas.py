
# coding: utf-8

# # 03__hit_gwas
# 
# in this notebook, i examine whether our hits have GWAS-associated SNPs in close proximity
# 
# figures in this notebook:
# - Fig 5C: barplot showing % of hits vs. non-hits with an endo-cancer associated SNP within 5kb

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


# all gwas files
gwas_dir = "../../../misc/06__gwas"
closest_gwas_f = "%s/tss_coords.closest_gwas_snp.with_DIGIT.bed" % gwas_dir
closest_cancer_f = "%s/tss_coords.closest_all_cancer_snp.with_DIGIT.bed" % gwas_dir
closest_endo_f = "%s/tss_coords.closest_endo_cancer_snp.with_DIGIT.bed" % gwas_dir
closest_gwas_gb_f = "%s/transcript_coords.closest_gwas_snp.with_DIGIT.bed" % gwas_dir
closest_cancer_gb_f = "%s/transcript_coords.closest_all_cancer_snp.with_DIGIT.bed" % gwas_dir
closest_endo_gb_f = "%s/transcript_coords.closest_endo_cancer_snp.with_DIGIT.bed" % gwas_dir


# ## 1. import data

# In[6]:


index = pd.read_table(index_f)
index.head()


# In[7]:


data = pd.read_table(data_f)
data.head()


# In[8]:


closest_gwas = pd.read_table(closest_gwas_f, sep="\t", header=None)
closest_gwas.columns = ["chr", "start", "end", "transcript_id", "score", "strand", "snp_chr", "snp_start", "snp_end",
                        "snp_id", "disease", "distance"]
closest_gwas.head()


# In[9]:


closest_cancer = pd.read_table(closest_cancer_f, sep="\t", header=None)
closest_cancer.columns = ["chr", "start", "end", "transcript_id", "score", "strand", "snp_chr", "snp_start", "snp_end",
                        "snp_id", "disease", "distance"]
closest_cancer.head()


# In[10]:


closest_endo = pd.read_table(closest_endo_f, sep="\t", header=None)
closest_endo.columns = ["chr", "start", "end", "transcript_id", "score", "strand", "snp_chr", "snp_start", "snp_end",
                        "snp_id", "disease", "distance"]
closest_endo.head()


# In[11]:


closest_gwas_gb = pd.read_table(closest_gwas_gb_f, sep="\t", header=None)
closest_gwas_gb.columns = ["chr", "start", "end", "transcript_id", "snp_chr", "snp_start", "snp_end",
                        "snp_id", "disease", "distance"]
closest_gwas_gb.head()


# In[12]:


closest_cancer_gb = pd.read_table(closest_cancer_gb_f, sep="\t", header=None)
closest_cancer_gb.columns = ["chr", "start", "end", "transcript_id", "snp_chr", "snp_start", "snp_end",
                        "snp_id", "disease", "distance"]
closest_cancer_gb.head()


# In[13]:


closest_endo_gb = pd.read_table(closest_endo_gb_f, sep="\t", header=None)
closest_endo_gb.columns = ["chr", "start", "end", "transcript_id", "snp_chr", "snp_start", "snp_end",
                        "snp_id", "disease", "distance"]
closest_endo_gb.head()


# ## 2. join metadata files

# In[14]:


closest_gwas = closest_gwas[["transcript_id", "snp_id", "disease", "distance"]].drop_duplicates()
closest_cancer = closest_cancer[["transcript_id", "snp_id", "disease", "distance"]].drop_duplicates()
closest_endo = closest_endo[["transcript_id", "snp_id", "disease", "distance"]].drop_duplicates()


# In[15]:


print(len(closest_gwas))
print(len(closest_cancer))
print(len(closest_endo))


# In[16]:


meta = closest_gwas.merge(closest_cancer, on="transcript_id", how="left").merge(closest_endo, on="transcript_id", how="left")
len(meta)


# In[17]:


meta.columns = ["transcript_id", "closest_all_snp", "closest_all_snp_disease", "closest_all_snp_dist",
                "closest_cancer_snp", "closest_cancer_snp_disease", "closest_cancer_snp_dist", 
                "closest_endo_snp", "closest_endo_snp_disease", "closest_endo_snp_dist"]
meta.head()


# In[18]:


# some transcripts have duplicate closest snps when there are ties, so drop these (keep first arbitrarily)
meta.drop_duplicates(subset="transcript_id", inplace=True)
print(len(meta))
meta.head()


# In[19]:


closest_gwas_gb = closest_gwas_gb[["transcript_id", "snp_id", "disease", "distance"]].drop_duplicates()
closest_cancer_gb = closest_cancer_gb[["transcript_id", "snp_id", "disease", "distance"]].drop_duplicates()
closest_endo_gb = closest_endo_gb[["transcript_id", "snp_id", "disease", "distance"]].drop_duplicates()


# In[20]:


print(len(closest_gwas_gb))
print(len(closest_cancer_gb))
print(len(closest_endo_gb))


# In[21]:


meta = meta.merge(closest_gwas_gb, on="transcript_id", how="left").merge(closest_cancer_gb, on="transcript_id", how="left").merge(closest_endo_gb, on="transcript_id", how="left")
len(meta)


# In[22]:


meta.columns = ["transcript_id", "tss_closest_all_snp", "tss_closest_all_snp_disease", "tss_closest_all_snp_dist",
                "tss_closest_cancer_snp", "tss_closest_cancer_snp_disease", "tss_closest_cancer_snp_dist", 
                "tss_closest_endo_snp", "tss_closest_endo_snp_disease", "tss_closest_endo_snp_dist",
                "gb_closest_all_snp", "gb_closest_all_snp_disease", "gb_closest_all_snp_dist",
                "gb_closest_cancer_snp", "gb_closest_cancer_snp_disease", "gb_closest_cancer_snp_dist",
                "gb_closest_endo_snp", "gb_closest_endo_snp_disease", "gb_closest_endo_snp_dist"]
meta.head()


# In[23]:


# some transcripts have duplicate closest snps when there are ties, so drop these (keep first arbitrarily)
meta.drop_duplicates(subset="transcript_id", inplace=True)
print(len(meta))
meta.head()


# In[24]:


meta[meta["transcript_id"] == "DIGIT"]


# ## 3. merge metadata w/ hit info

# In[25]:


data.is_hit.value_counts()


# In[26]:


data = data.merge(meta, on="transcript_id", how="left")
print(len(data))
data.head()


# In[27]:


data.is_hit.value_counts()


# ## 4. plots to examine gwas closeness

# In[28]:


order = ["lncRNA_good_csf", "protein_coding"]
hue_order = ["no hit", "stringent hit"]
pal = {"no hit": "gray", "lenient hit": sns.color_palette("Set2")[2], "stringent hit": "black"}


# In[29]:


hue_order2 = ["no hit", "hit"]
pal2 = {"no hit": "gray", "hit": "black"}


# In[30]:


def snp_within_x(row, col, x):
    if row[col] < x:
        return "yes"
    else:
        return "no"


# In[31]:


data["gwas_within_5_gb"] = data.apply(snp_within_x, col="gb_closest_all_snp_dist", x=5000, axis=1)
data["cancer_within_5_gb"] = data.apply(snp_within_x, col="gb_closest_cancer_snp_dist", x=5000, axis=1)
data["endo_within_5_gb"] = data.apply(snp_within_x, col="gb_closest_endo_snp_dist", x=5000, axis=1)


# In[33]:


x = 5

for t in ["endo"]:
    fig, axarr = plt.subplots(figsize=(1.75, 1.85), nrows=1, ncols=2, sharey=True)

    for i, csf in enumerate(order):
        ax = axarr[i]
        sub = data[data["csf"] == csf]
        sub = sub[sub["gb_closest_all_snp_dist"] != -1]

        tots = sub.groupby("is_hit")["gene_id"].agg("count").reset_index()
        within_x = sub.groupby(["is_hit", "%s_within_%s_gb" % (t, x)])["gene_id"].agg("count").reset_index()
        within_x = within_x[within_x["%s_within_%s_gb" % (t, x)] == "yes"]
        perc = tots.merge(within_x, on="is_hit", how="left")
        perc["perc"] = (perc["gene_id_y"]/perc["gene_id_x"])*100

        # fisher's exact
        perc["no_enh"] = perc["gene_id_x"] - perc["gene_id_y"]
        table = perc[perc["is_hit"].isin(hue_order)][["is_hit", "no_enh", "gene_id_y"]].set_index("is_hit")
        #print(table)
        table.fillna(0, inplace=True)
        o, p = stats.fisher_exact(table, alternative='greater')
        print(p)


        sns.barplot(data=perc, x="is_hit", y="perc", ax=ax, order=hue_order, palette=pal)

        ax.set_xticklabels(hue_order, rotation=45, ha="right", va="top")
        ax.set_xlabel("")
        ax.set_ylabel("%% with %s cancer SNP within %s kb" % (t, x))
        ax.set_title(csf)
        if i != 0:
            ax.set_ylabel("")
        
        if t == "gwas":
            ax.set_ylim((0, 70))
        elif t == "cancer":
            ax.set_ylim((0, 25))
        else:
            ax.set_ylim((0, 15))
            
        # annotate pval
        max_p = perc["perc"].max()
        annotate_pval(ax, 0.2, 0.8, max_p + 1, 0, max_p + 1, p, fontsize)

        fig.savefig("Fig5C.pdf", dpi="figure", bbox_inches="tight")


# ## write file

# In[34]:


len(meta)


# In[35]:


meta.to_csv("../../../data/02__screen/02__enrichment_data/transcript_features.txt", sep="\t", index=False)

