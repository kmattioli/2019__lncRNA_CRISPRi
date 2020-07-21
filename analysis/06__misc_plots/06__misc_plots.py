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

# import utils
sys.path.append("../../utils")
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


fig1f_f = "../../data/05__misc_plots/fig1f_rt_qpcr.txt"
fig1g_f = "../../data/05__misc_plots/fig1g_rt_qpcr.txt"
fig7c_f = "../../data/05__misc_plots/fig7c_rt_qpcr.txt"
fig7b_f = "../../data/05__misc_plots/fig7b_time_course.txt"
figs9a_f = "../../data/05__misc_plots/figs9a_time_course.txt"


# ## 1. import data

# In[5]:


fig1f = pd.read_table(fig1f_f, sep="\t")
fig1f.head()


# In[6]:


fig1g = pd.read_table(fig1g_f, sep="\t")
fig1g.head()


# In[7]:


fig7c = pd.read_table(fig7c_f, sep="\t")
fig7c.head()


# In[8]:


fig7b = pd.read_table(fig7b_f, sep="\t")
fig7b.head()


# In[9]:


figs9a = pd.read_table(figs9a_f, sep="\t")
figs9a.head()


# ## 2. make Fig 1F

# In[10]:


fig1f_m = pd.melt(fig1f, id_vars="gene_name")
fig1f_m.head()


# In[11]:


order = list(fig1f_m.gene_name.unique())
pal = {"scram_mean": "darkgray", "sgrna_mean": "black"}


# In[12]:


fig = plt.figure(figsize=(6, 2))

ax = sns.barplot(data=fig1f_m[fig1f_m["variable"].str.contains("mean")], 
                 x="gene_name", y="value", hue="variable", order=order, palette=pal)
ax.set_xlabel("")
ax.set_ylabel("fold change")
ax.get_legend().remove()
ax.set_ylim((0, 1.3))
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticklabels(order, rotation=40, ha="right", va="top")

# plot errors
x1 = [x-0.2 for x in range(len(order))]
x2 = [x+0.2 for x in range(len(order))]
xs = [[x, y] for x, y in zip(x1, x2)]
xs = [i for x in xs for i in x]
c = 0
for i, x in enumerate(xs):
    if i % 2 == 0:
        name = order[c]
        c += 1
    subvals = fig1f_m[(fig1f_m["gene_name"] == name) & (fig1f_m["variable"].str.contains("mean"))]
    subyerrs = fig1f_m[(fig1f_m["gene_name"] == name) & (fig1f_m["variable"].str.contains("error"))]
    if i % 2 == 0: #scrambled sgRNA
        val_ = subvals[subvals["variable"].str.contains("scram")]["value"].iloc[0]
        yerr_ = subyerrs[subyerrs["variable"].str.contains("scram")]["value"].iloc[0]
    else:
        val_ = subvals[~subvals["variable"].str.contains("scram")]["value"].iloc[0]
        yerr_ = subyerrs[~subyerrs["variable"].str.contains("scram")]["value"].iloc[0]
    ax.plot([x, x], [val_ - yerr_, val_ + yerr_], color="black")
    ax.plot([x-0.1, x+0.1], [val_ - yerr_, val_ - yerr_], color="black")
    ax.plot([x-0.1, x+0.1], [val_ + yerr_, val_ + yerr_], color="black")
    
fig.savefig("Fig1F.pdf", dpi="figure", bbox_inches="tight")


# ## 3. make Fig 1G

# In[13]:


fig1g_m = pd.melt(fig1g, id_vars="gene_name")
fig1g_m.head()


# In[14]:


order = list(fig1g_m.gene_name.unique())
pal = {"scram_mean": "darkgray", "sgrna_mean": "black"}


# In[15]:


order


# In[16]:


for num, gene in enumerate(order):
    sub = fig1g_m[fig1g_m["gene_name"] == gene]
    
    fig = plt.figure(figsize=(0.7, 2))
    
    ax = sns.barplot(data=sub[sub["variable"].str.contains("mean")], 
                 x="gene_name", y="value", hue="variable", palette=pal)
    ax.set_xlabel("")
    ax.set_ylabel("fold change")
    ax.get_legend().remove()
    ax.set_ylim((0, 1.3))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels([order[num]])

    # plot errors
    x1 = [x-0.2 for x in range(1)]
    x2 = [x+0.2 for x in range(1)]
    xs = [[x, y] for x, y in zip(x1, x2)]
    xs = [i for x in xs for i in x]
    c = 0
    for i, x in enumerate(xs):
        if i % 2 == 0:
            c += 1
        subvals = sub[(sub["gene_name"] == gene) & (sub["variable"].str.contains("mean"))]
        subyerrs = sub[(sub["gene_name"] == gene) & (sub["variable"].str.contains("error"))]
        if i % 2 == 0: #scrambled sgRNA
            val_ = subvals[subvals["variable"].str.contains("scram")]["value"].iloc[0]
            yerr_ = subyerrs[subyerrs["variable"].str.contains("scram")]["value"].iloc[0]
        else:
            val_ = subvals[~subvals["variable"].str.contains("scram")]["value"].iloc[0]
            yerr_ = subyerrs[~subyerrs["variable"].str.contains("scram")]["value"].iloc[0]
        ax.plot([x, x], [val_ - yerr_, val_ + yerr_], color="black")
        ax.plot([x-0.1, x+0.1], [val_ - yerr_, val_ - yerr_], color="black")
        ax.plot([x-0.1, x+0.1], [val_ + yerr_, val_ + yerr_], color="black")

    fig.savefig("Fig1G_%s.pdf" % str(num+1), dpi="figure", bbox_inches="tight")


# ## 3. make figure 7C

# In[17]:


fig7c_m = pd.melt(fig7c, id_vars="gene_name")
fig7c_m.head()


# In[18]:


pal = {"scram_mean": "darkgray", "foxd3_shrna_1_mean": sns.color_palette("Set2")[0], 
       "foxd3_shrna_2_mean": sns.color_palette("Set2")[1]}


# In[19]:


sub = fig7c_m[fig7c_m["gene_name"] == "FOXD3-AS1"]
    
fig = plt.figure(figsize=(0.8, 2))

ax = sns.barplot(data=sub[sub["variable"].str.contains("mean")], 
                 x="variable", y="value", palette=pal)
ax.set_xlabel("")
ax.set_ylabel("fold change")
ax.set_ylim((0, 1.3))
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticklabels(["scrambled shRNA", "FOXD3-AS1 shRNA #1", "FOXD3-AS1 shRNA #2"], rotation=40, ha="right", va="top")

# plot errors
xs = [0, 1, 2]

c = 0
for i, x in enumerate(xs):
    if x == 0: #scramble
        val_ = sub[sub["variable"] == "scram_mean"]["value"].iloc[0]
        yerr_ = sub[sub["variable"] == "scram_error"]["value"].iloc[0]
        x = x - 0.067
    elif x == 2: #shrna #2
        val_ = sub[sub["variable"] == "foxd3_shrna_2_mean"]["value"].iloc[0]
        yerr_ = sub[sub["variable"] == "foxd3_shrna_2_error"]["value"].iloc[0]
        x = x + 0.067
    else: #shrna #1
        val_ = sub[sub["variable"] == "foxd3_shrna_1_mean"]["value"].iloc[0]
        yerr_ = sub[sub["variable"] == "foxd3_shrna_1_error"]["value"].iloc[0]
    ax.plot([x, x], [val_ - yerr_, val_ + yerr_], color="black")
    ax.plot([x-0.1, x+0.1], [val_ - yerr_, val_ - yerr_], color="black")
    ax.plot([x-0.1, x+0.1], [val_ + yerr_, val_ + yerr_], color="black")


fig.savefig("Fig7C.pdf", dpi="figure", bbox_inches="tight")


# In[20]:


sub = fig7c_m[fig7c_m["gene_name"] != "FOXD3-AS1"]
order = list(sub.gene_name.unique())
    
fig = plt.figure(figsize=(7, 2))

ax = sns.barplot(data=sub[sub["variable"].str.contains("mean")], 
                 x="gene_name", y="value", hue="variable", palette=pal, order=order)
ax.set_xlabel("")
ax.set_ylabel("fold change")
ax.get_legend().remove()
ax.set_ylim((0, 20))
ax.set_xticklabels(order, rotation=40, va="top", ha="right")

# plot errors
x1 = [x-0.2 for x in range(len(order))]
x2 = [x for x in range(len(order))]
x3 = [x+0.2 for x in range(len(order))]
xs = [[x, y, z] for x, y, z in zip(x1, x2, x3)]
xs = [i for x in xs for i in x]

c = 0
for i, x in enumerate(xs):
    if x == -0.2 or ".8" in str(x): #scramble
        name = order[c]
        val_ = sub[(sub["gene_name"] == name) & (sub["variable"] == "scram_mean")]["value"].iloc[0]
        yerr_ = sub[(sub["gene_name"] == name) & (sub["variable"] == "scram_error")]["value"].iloc[0]
        x = x - 0.067
        c += 1
    elif ".2" in str(x): #shrna #2
        val_ = sub[(sub["gene_name"] == name) & (sub["variable"] == "foxd3_shrna_2_mean")]["value"].iloc[0]
        yerr_ = sub[(sub["gene_name"] == name) & (sub["variable"] == "foxd3_shrna_2_error")]["value"].iloc[0]
        x = x + 0.067
    else: #shrna #1
        val_ = sub[(sub["gene_name"] == name) & (sub["variable"] == "foxd3_shrna_1_mean")]["value"].iloc[0]
        yerr_ = sub[(sub["gene_name"] == name) & (sub["variable"] == "foxd3_shrna_1_error")]["value"].iloc[0]
    ax.plot([x, x], [val_ - yerr_, val_ + yerr_], color="black")
    ax.plot([x-0.1, x+0.1], [val_ - yerr_, val_ - yerr_], color="black")
    ax.plot([x-0.1, x+0.1], [val_ + yerr_, val_ + yerr_], color="black")


fig.savefig("Fig7E.pdf", dpi="figure", bbox_inches="tight")


# ## 4. make Figure 7B

# In[21]:


uniq_genes = list(set([x[:-5] for x in fig7b.columns if "_mean" in x]))
uniq_genes


# In[22]:


order = ["foxd3_as1", "foxd3", "oct4", "nanog", "sox2"]
labels = ["FOXD3-AS1", "FOXD3", "OCT4", "NANOG", "SOX2"]
colors = sns.xkcd_palette(["melon", "bluish green", "windows blue", "marigold", "orchid"])

fig = plt.figure(figsize=(2.5, 2))

for i, gene, label in zip(list(range(len(order))), order, labels):
    plt.errorbar(fig7b["day"], fig7b["%s_mean" % gene], yerr=fig7b["%s_stdev" % gene], 
                 label=label, color=colors[i])
    plt.plot(fig7b["day"], fig7b["%s_mean" % gene], '-', color=colors[i])
    plt.plot(fig7b["day"], fig7b["%s_mean" % gene], '.', color=colors[i])
    plt.legend(loc=2, bbox_to_anchor=(1.05, 0.75))

plt.ylabel("fold change")
fig.savefig("Fig7B.pdf", dpi="figure", bbox_inches="tight")


# ## 5. make Figure S9A

# In[23]:


uniq_genes = [x for x in figs9a.columns if "mean" in x]
print(len(uniq_genes))


# In[24]:


hits = ["DIGIT", "RP11-222K16.2", "RP11-541P9.3", "ACVR2B-AS1", "RP3-508I15.9", "KB-1440D3.14", "LLNLR-260G6.1",
        "RP11-121L10.2", "RP11-414H23.2", "RP1-148H17.1", "AC068831.3", "RP11-1C8.4", "LAMTOR5-AS1",
        "LINC01424", "RP11-421L21.3", "PITPNA-AS1", "FOXD3-AS1", "RP11-120D5.1"]


# In[25]:


fig = plt.figure(figsize=(2.5, 2))
colors = sns.color_palette("husl", n_colors=len(hits))

for i, hit in enumerate(hits):
    plt.errorbar(figs9a["day"], figs9a["%s_mean" % hit], yerr=figs9a["%s_stdev" % hit], 
                 label=hit, color=colors[i])
    plt.plot(figs9a["day"], figs9a["%s_mean" % hit], '-', color=colors[i])
    plt.plot(figs9a["day"], figs9a["%s_mean" % hit], '.', color=colors[i])
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.2))
    plt.yscale("log")
    plt.ylabel("fold change")
    plt.title("endoderm differentiation")
    
fig.savefig("FigS9A.pdf", dpi="figure", bbox_inches="tight")


# In[ ]:




