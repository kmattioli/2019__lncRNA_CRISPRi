#!/usr/bin/env python
# coding: utf-8

# # 01__tSNE
# 
# in this notebook, i plot the 11 genomic features aggregated for lncRNAs and mRNAs in 2D using tSNE, and then perform k-means clustering.
# 
# figures in this notebook:
# - Fig 6B-D: tSNE with various things highlighted
# - Fig S7: tSNE colored by feature
# 
# tables in this notebook:
# - Table S7: result of clustering analysis

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

from matplotlib.patches import Rectangle

from scipy.stats import spearmanr
from scipy.stats import pearsonr
from statsmodels.sandbox.stats import multicomp

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


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


# ## functions

# In[4]:


def min_biotype(row):
    if row.cleaner_gene_biotype == "protein_coding":
        return "mRNA"
    else:
        return "lncRNA"


# ## variables

# In[5]:


data_f = "../../../data/03__features/all_features.tmp"


# In[6]:


hits_f = "../../../data/03__features/SuppTable_S5.locus_features.txt"


# ## 1. import data

# In[7]:


data = pd.read_table(data_f)
data.head()


# In[8]:


data[data["gene_name"] == "DIGIT"]


# In[9]:


hits = pd.read_table(hits_f)
hits.head()


# In[10]:


len(hits)


# ## 2. filter data -- remove bad biotypes & rows with NA features

# In[11]:


data_filt = data[~pd.isnull(data["gene_name"])]
len(data_filt)


# In[12]:


data_filt_nonan = data_filt.dropna(axis=0)
len(data_filt_nonan)


# In[13]:


meta_cols = ["gene_id", "gene_name", "csf", "cleaner_gene_biotype", "minimal_biotype"]
all_feature_cols = ['IMR-90_eff_mean', 'IMR-90_exp_mean', 'IMR-90_eff_ratio', 'HUES64_eff_mean',
       'HUES64_exp_mean', 'HUES64_eff_ratio', 'NCI-H460_eff_mean', 'NCI-H460_exp_mean', 'NCI-H460_eff_ratio',
       'SK-N-SH_eff_mean', 'SK-N-SH_exp_mean', 'SK-N-SH_eff_ratio', 'K562_eff_mean',
       'K562_exp_mean', 'K562_eff_ratio', 'A549_eff_mean', 'A549_exp_mean', 'A549_eff_ratio', 'H1_eff_mean',
       'H1_exp_mean', 'H1_eff_ratio', 'MCF-7_eff_mean', 'MCF-7_exp_mean', 'MCF-7_eff_ratio' , 'HT1080_eff_mean',
       'HT1080_exp_mean', 'HT1080_eff_ratio', 'SK-N-DZ_eff_mean', 'SK-N-DZ_exp_mean', 'SK-N-DZ_eff_ratio',
       'SK-MEL-5_eff_mean', 'SK-MEL-5_exp_mean', 'SK-MEL-5_eff_ratio', 'GM12878_eff_mean',
       'GM12878_exp_mean', 'GM12878_eff_ratio', 'max_eff', 'max_exp', 'max_ratio', 'gc', 'n_tss', 'n_enh', 
       'enh_tss_dist', 'prom_cons', 'exon_cons', 'dna_len', 'rna_len', 'n_exons']


# In[14]:


log_cols = ["max_exp", "enh_tss_dist", "dna_len", "rna_len", "n_exons"]
for col in log_cols:
    data_filt_nonan["%s_log" % col] = np.log10(data_filt_nonan[col]+1)


# In[15]:


sub_feature_cols = ['max_eff', 'max_exp', 'gc', 'n_tss', 'n_enh', 'enh_tss_dist', 'prom_cons',
                    'exon_cons', 'dna_len', 'rna_len', 'n_exons']


# ## 3. calculate t-SNE embeddings

# In[16]:


# Separating out the features
x = data_filt_nonan.loc[:, sub_feature_cols].values

# Standardizing the features
x = StandardScaler().fit_transform(x)
len(x)


# In[17]:


x.shape


# In[18]:


x_embedded = TSNE(n_components=2).fit_transform(x)
x_embedded


# ## 4. cluster the data

# In[19]:


alg = KMeans(n_clusters=2)
alg.fit_predict(x)
labels = alg.labels_


# In[20]:


print(len(labels))
labels


# In[21]:


unique, counts = np.unique(labels, return_counts=True)
print(np.asarray((unique, counts)).T)


# In[22]:


labels = [x+1 for x in labels]
len(labels)


# In[23]:


unique, counts = np.unique(labels, return_counts=True)
print(np.asarray((unique, counts)).T)


# In[24]:


len(unique)


# In[25]:


df = pd.DataFrame(data = x_embedded, columns = ["f1", "f2"])
df = pd.concat([df, data_filt_nonan[meta_cols].reset_index()], axis = 1)
df = pd.concat([df, data_filt_nonan[sub_feature_cols].reset_index()], axis = 1)
df["cluster"] = labels
df.head()


# ## 5. visualize data using t-SNE

# In[26]:


c1 = sns.color_palette("magma", n_colors=2, desat=0.5)[0]
c2 = sns.color_palette("plasma", n_colors=12, desat=0.5)[10]


# In[27]:


lut = dict(zip(list(range(1, len(unique)+1)), [c2, c1]))
row_colors = df.cluster.map(lut)
df["color"] = row_colors
df.head()


# In[28]:


# sample mRNAs and lncRNAs for plotting purposes
mrnas = df[df["minimal_biotype"] == "mRNA"].sample(1000)
lncrnas = df[df["minimal_biotype"].isin(["lncRNA", "featured lncRNA"])].sample(1000)
#feat = df[df["minimal_biotype"] == "featured lncRNA"]
spec = df[df["gene_name"].isin(["XIST", "MALAT1", "NEAT1"])]

all_plot = mrnas.append(lncrnas).append(spec)
len(all_plot)


# In[29]:


spec


# In[30]:


fig, ax = plt.subplots(figsize=(1.5, 1.5), nrows=1, ncols=1)

for c in unique:
    print(c)
    sub = all_plot[all_plot["cluster"] == c]
    ax.scatter(sub["f1"], sub["f2"], color=sub["color"].iloc[0], s=5, linewidths=0, alpha=0.8, label="cluster %s" % c)

plt.legend(loc=2, bbox_to_anchor=(1.05, 1))

ax.set_xlabel("t-SNE dimension 1")
ax.set_ylabel("t-SNE dimension 2")
fig.savefig("Fig6B.pdf", dpi="figure", bbox_inches="tight")


# In[31]:


fig, ax = plt.subplots(figsize=(1.5, 1.5), nrows=1, ncols=1)

ax.scatter(mrnas["f1"], mrnas["f2"], color=sns.color_palette("deep")[2], edgecolors="darkgreen", linewidths=0,
           s=5, alpha=0.8)
ax.scatter(lncrnas["f1"], lncrnas["f2"], color="silver", edgecolors="dimgray", linewidths=0, s=5, alpha=0.8)
ax.scatter(spec["f1"], spec["f2"], color="dimgray", edgecolors="black", 
           linewidths=0.75, s=17, alpha=1)

ax.set_xlabel("t-SNE dimension 1")
ax.set_ylabel("t-SNE dimension 2")

plt.show()
fig.savefig("Fig6C.pdf", dpi="figure", bbox_inches="tight")


# In[32]:


titles = ["       max splicing efficiency       ", "       log10 max expression       ", 
          "                GC content                ", 
          "          # TSSs w/in 100bp          ",  "        # enhancers w/in 1Mb        ", 
          "  log10 bp to closest enhancer  ", 
          "       promoter conservation       ", "            exon conservation         ",
          "      log10 DNA locus length       ", "   log10 RNA transcript length   ",
          "       log10 number of exons       "]
logs = [False, True, False, False, False, True, False, False, True, True, True]

fig = plt.subplots(figsize=(6, 6), squeeze=False, sharex=True, sharey=True)

c = 0
for i in range(4):
    for j in range(3):
        ax = plt.subplot2grid((4, 3), (i, j))
        if c < len(sub_feature_cols):
            col = sub_feature_cols[c]
            log = logs[c]

            if log:
                sc = ax.scatter(all_plot["f1"], all_plot["f2"], c=np.log10(all_plot[col]+1), s=2, alpha=0.6,
                                cmap="viridis")
            else:
                sc = ax.scatter(all_plot["f1"], all_plot["f2"], c=all_plot[col], s=2, alpha=0.6, cmap="viridis")
            ax.set_title(titles[c], loc="left",
                         **{"bbox": dict(facecolor='lightgray', edgecolor=None, linewidth=None, 
                                         pad=0.6), "va": "center"})
            
            c += 1
            
            plt.colorbar(sc, ax=ax)
        
        else:
            v = mpl.cm.get_cmap('viridis', 12)
            ax.scatter(mrnas["f1"], mrnas["f2"], color=v(0), s=2, alpha=0.6)
            ax.scatter(lncrnas["f1"], lncrnas["f2"], color=v(11), s=2, alpha=0.6)
            ax.scatter(spec["f1"], spec["f2"], color="white", alpha=1, s=30, 
                        linewidths=1, edgecolors="black")
            ax.set_title("                   biotype                   ", loc="left", color="white",
                         **{"bbox": dict(facecolor='black', edgecolor=None, linewidth=None, 
                                         pad=0.6), "va": "center"})
            plt.colorbar(sc, ax=ax)
        
        if j == 0:
            ax.set_ylabel("t-SNE dimension 2")
        if i == 3:
            ax.set_xlabel("t-SNE dimension 1")

plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.savefig("FigS7.pdf", dpi="figure", bbox_inches="tight")


# In[33]:


fig, ax = plt.subplots(figsize=(1.75, 1.5), nrows=1, ncols=1)

sc = ax.scatter(all_plot["f1"], all_plot["f2"], c=all_plot["max_eff"], s=5, linewidths=0, alpha=0.8, cmap="viridis")
plt.colorbar(sc, ax=ax)

ax.set_xlabel("t-SNE dimension 1")
ax.set_ylabel("t-SNE dimension 2")

plt.show()
#fig.savefig("tsne_splicing_eff.pdf", dpi="figure", bbox_inches="tight")


# ## 6. limit to genes in screen

# In[34]:


hits.head()


# In[35]:


hits.is_hit.value_counts()


# In[36]:


print(len(df))
df_screen = df.merge(hits[["gene_id", "is_hit"]], on="gene_id")
print(len(df_screen))
df_screen.head()


# In[37]:


mrnas_screen = df_screen[df_screen["minimal_biotype"] == "mRNA"]
lncrnas_screen = df_screen[df_screen["minimal_biotype"] != "mRNA"]

mrnas_hits = mrnas_screen[mrnas_screen["is_hit"] == "hit"]
lncrnas_hits = lncrnas_screen[lncrnas_screen["is_hit"] == "hit"]


# In[38]:


fig, ax = plt.subplots(figsize=(1.5, 1.5), nrows=1, ncols=1)

ax.scatter(lncrnas_screen["f1"], lncrnas_screen["f2"], color="silver", s=2, linewidths=0, alpha=0.8)
ax.scatter(mrnas_hits["f1"], mrnas_hits["f2"], color=sns.color_palette("deep")[2], edgecolors="black", s=15, alpha=1)
ax.scatter(lncrnas_hits["f1"], lncrnas_hits["f2"], color="dimgray", edgecolors="black", s=15, alpha=1)

ax.set_xlabel("t-SNE dimension 1")
ax.set_ylabel("t-SNE dimension 2")

plt.show()


# In[39]:


fig, ax = plt.subplots(figsize=(1.5, 1.5), nrows=1, ncols=1)

for c in unique[::-1]:
    print(c)
    sub = lncrnas_screen[lncrnas_screen["cluster"] == c]
    ax.scatter(sub["f1"], sub["f2"], color=sub["color"].iloc[0], s=2, linewidths=0, 
               alpha=0.7, label="cluster %s" % c)
    
    sub = lncrnas_hits[lncrnas_hits["cluster"] == c]
    ax.scatter(sub["f1"], sub["f2"], color=sub["color"].iloc[0], edgecolors="black", 
               linewidths=0.75, s=17, alpha=1, zorder=100)

plt.legend(loc=2, bbox_to_anchor=(1.05, 1))

ax.set_xlabel("t-SNE dimension 1")
ax.set_ylabel("t-SNE dimension 2")

plt.show()
fig.savefig("Fig6D.pdf", dpi="figure", bbox_inches="tight")


# In[40]:


dedup = lncrnas_hits[["f1", "f2", "minimal_biotype", "cleaner_gene_biotype", "gene_name", "cluster"]].drop_duplicates()
print(len(dedup))
dedup.sort_values(by="f1", ascending=False).head(10)


# In[41]:


tmp = dedup[dedup["cluster"] == 1]
tmp.sort_values(by="f1")


# ## 9. write files

# In[42]:


len(df)


# In[43]:


df.minimal_biotype.value_counts()


# In[44]:


sub_cols = ["gene_id", "gene_name", "minimal_biotype", "max_eff", "max_exp", "gc", "n_tss", "n_enh", "enh_tss_dist",
            "prom_cons", "exon_cons", "dna_len", "rna_len", "n_exons", "f1", "f2", "cluster"]
supp = df[sub_cols]
supp.columns = ["gene_id", "gene_name", "biotype", "max_splicing_eff", "max_expression", "gc", "n_tss", "n_enh",
                "closest_enh_dist", "tss_conservation", "exon_conservation", "dna_locus_len", "rna_trans_len",
                "n_exons", "tsne_f1", "tsne_f2", "assigned_cluster"]


# In[45]:


supp = supp.sort_values(by="gene_id")
supp.head()


# In[46]:


supp.to_csv("../../../data/04__clusters/SuppTable_S7.cluster_predictions.txt", sep="\t", index=False)


# In[47]:


df_screen.head()


# In[48]:


sub_cols = ["gene_id", "gene_name", "minimal_biotype", "cleaner_gene_biotype", "f1", "f2", "is_hit", "cluster"]
supp = df_screen[sub_cols]
supp.columns = ["gene_id", "gene_name", "biotype", "lncrna_class", "tsne_f1", "tsne_f2", "screen_hit", 
                "assigned_cluster"]


# In[49]:


supp[supp["gene_name"].isin(["RP11-1144P22.1", "CTD-2058B24.2", "DANCR", "DIGIT", "RP11-222K16.2",
                             "FOXD3-AS1", "RP11-479O16.1", "RP11-120D5.1"])]


# In[ ]:




