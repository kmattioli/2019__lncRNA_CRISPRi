
# coding: utf-8

# # 00__preprocess
# 
# in this notebook, i upload all individual library/primer sgRNA counts and sum together (after checking correlations) to make 1 dataframe with all sgRNA counts across populations (Day Zero, Endo++, Endo--).
# 
# figures in this notebook:
# - Fig S5B (heatmap showing biological replicate correlations of sgRNA counts)

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybedtools as pb
import re
import seaborn as sns
import sys

from scipy import stats
from scipy.stats import spearmanr

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

# In[3]:


def to_cpm(df):
    cpm = pd.DataFrame()
    cpm["sgRNA"] = df["sgRNA"]
    for col in df.columns:
        if col not in ["sgRNA"]:
            cpm[col] = df[col]/np.nansum(df[col])*1e6
    return cpm


# In[4]:


def pseudocount(df):
    pseudo = pd.DataFrame()
    pseudo["sgRNA"] = df["sgRNA"]
    for col in df.columns:
        if col not in ["sgRNA"]:
            pseudo[col] = df[col] + 1
    return pseudo


# ## variables

# In[5]:


# day zero
day0_lib1_rep1_a_f = "../../../data/02__screen/00__counts/CRISPRi__DayZero__Rep1__Lib1__SetA.sgRNA_counts.txt"
day0_lib1_rep1_b_f = "../../../data/02__screen/00__counts/CRISPRi__DayZero__Rep1__Lib1__SetB.sgRNA_counts.txt"
day0_lib2_rep1_a_f = "../../../data/02__screen/00__counts/CRISPRi__DayZero__Rep1__Lib2__SetA.sgRNA_counts.txt"
day0_lib2_rep1_b_f = "../../../data/02__screen/00__counts/CRISPRi__DayZero__Rep1__Lib2__SetB.sgRNA_counts.txt"
day0_lib3_rep1_a_f = "../../../data/02__screen/00__counts/CRISPRi__DayZero__Rep1__Lib3__SetA.sgRNA_counts.txt"
day0_lib3_rep1_b_f = "../../../data/02__screen/00__counts/CRISPRi__DayZero__Rep1__Lib3__SetB.sgRNA_counts.txt"

day0_lib1_rep2_a_f = "../../../data/02__screen/00__counts/CRISPRi__DayZero__Rep2__Lib1__SetA.sgRNA_counts.txt"
day0_lib1_rep2_b_f = "../../../data/02__screen/00__counts/CRISPRi__DayZero__Rep2__Lib1__SetB.sgRNA_counts.txt"
day0_lib2_rep2_a_f = "../../../data/02__screen/00__counts/CRISPRi__DayZero__Rep2__Lib2__SetA.sgRNA_counts.txt"
day0_lib2_rep2_b_f = "../../../data/02__screen/00__counts/CRISPRi__DayZero__Rep2__Lib2__SetB.sgRNA_counts.txt"
day0_lib3_rep2_a_f = "../../../data/02__screen/00__counts/CRISPRi__DayZero__Rep2__Lib3__SetA.sgRNA_counts.txt"


# In[6]:


# BFP+ endo++
bfppos_endopos_lib1_rep1_a_f = "../../../data/02__screen/00__counts/CRISPRi__EndoPos__Rep1__Lib1__SetA.sgRNA_counts.txt"
bfppos_endopos_lib1_rep1_b_f = "../../../data/02__screen/00__counts/CRISPRi__EndoPos__Rep1__Lib1__SetB.sgRNA_counts.txt"
bfppos_endopos_lib2_rep1_a_f = "../../../data/02__screen/00__counts/CRISPRi__EndoPos__Rep1__Lib2__SetA.sgRNA_counts.txt"
bfppos_endopos_lib2_rep1_b_f = "../../../data/02__screen/00__counts/CRISPRi__EndoPos__Rep1__Lib2__SetB.sgRNA_counts.txt"

bfppos_endopos_rep2_a_f = "../../../data/02__screen/00__counts/CRISPRi__EndoPos__Rep2__Lib1__SetA.sgRNA_counts.txt"
bfppos_endopos_rep2_b_f = "../../../data/02__screen/00__counts/CRISPRi__EndoPos__Rep2__Lib1__SetB.sgRNA_counts.txt"


# In[7]:


# BFP+ endo--
bfppos_endoneg_lib1_rep1_a_f = "../../../data/02__screen/00__counts/CRISPRi__EndoNeg__Rep1__Lib1__SetA.sgRNA_counts.txt"
bfppos_endoneg_lib2_rep1_a_f = "../../../data/02__screen/00__counts/CRISPRi__EndoNeg__Rep1__Lib2__SetA.sgRNA_counts.txt"


bfppos_endoneg_rep2_a_f = "../../../data/02__screen/00__counts/CRISPRi__EndoNeg__Rep2__Lib1__SetA.sgRNA_counts.txt"
bfppos_endoneg_rep2_b_f = "../../../data/02__screen/00__counts/CRISPRi__EndoNeg__Rep2__Lib1__SetB.sgRNA_counts.txt"


# In[8]:


index_f = "../../../data/01__design/02__final_sgRNAs/crispri_with_primers.txt"


# ## 1. load data

# In[9]:


day0_lib1_rep1_a = pd.read_table(day0_lib1_rep1_a_f, sep="\t")
day0_lib1_rep1_b = pd.read_table(day0_lib1_rep1_b_f, sep="\t")
day0_lib2_rep1_a = pd.read_table(day0_lib2_rep1_a_f, sep="\t")
day0_lib2_rep1_b = pd.read_table(day0_lib2_rep1_b_f, sep="\t")
day0_lib3_rep1_a = pd.read_table(day0_lib3_rep1_a_f, sep="\t")
day0_lib3_rep1_b = pd.read_table(day0_lib3_rep1_b_f, sep="\t")

day0_rep1 = day0_lib1_rep1_a.merge(day0_lib1_rep1_b, 
                                   on="sgRNA").merge(day0_lib2_rep1_a, 
                                                     on="sgRNA").merge(day0_lib2_rep1_b, 
                                                                       on="sgRNA").merge(day0_lib3_rep1_a,
                                                                                         on="sgRNA").merge(day0_lib3_rep1_b,
                                                                                                           on="sgRNA")

day0_rep1.columns = ["sgRNA", "lib1_rep1_a", "lib1_rep1_b", "lib2_rep1_a", "lib2_rep1_b", "lib3_rep1_a", "lib3_rep1_b"]
day0_rep1.sort_values(by="sgRNA").head()


# In[10]:


day0_lib1_rep2_a = pd.read_table(day0_lib1_rep2_a_f, sep="\t")
day0_lib1_rep2_b = pd.read_table(day0_lib1_rep2_b_f, sep="\t")
day0_lib2_rep2_a = pd.read_table(day0_lib2_rep2_a_f, sep="\t")
day0_lib2_rep2_b = pd.read_table(day0_lib2_rep2_b_f, sep="\t")
day0_lib3_rep2_a = pd.read_table(day0_lib3_rep2_a_f, sep="\t")

day0_rep2 = day0_lib1_rep2_a.merge(day0_lib1_rep2_b, 
                                   on="sgRNA").merge(day0_lib2_rep2_a, 
                                                     on="sgRNA").merge(day0_lib2_rep2_b, 
                                                                       on="sgRNA").merge(day0_lib3_rep2_a, 
                                                                                         on="sgRNA")

day0_rep2.columns = ["sgRNA", "lib1_rep2_a", "lib1_rep2_b", "lib2_rep2_a", "lib2_rep2_b", "lib3_rep2_a"]
day0_rep2.sort_values(by="sgRNA").head()


# In[11]:


bfppos_endopos_lib1_rep1_a = pd.read_table(bfppos_endopos_lib1_rep1_a_f, sep="\t")
bfppos_endopos_lib1_rep1_b = pd.read_table(bfppos_endopos_lib1_rep1_b_f, sep="\t")
bfppos_endopos_lib2_rep1_a = pd.read_table(bfppos_endopos_lib2_rep1_a_f, sep="\t")
bfppos_endopos_lib2_rep1_b = pd.read_table(bfppos_endopos_lib2_rep1_b_f, sep="\t")
bfppos_endopos_rep2_a = pd.read_table(bfppos_endopos_rep2_a_f, sep="\t")
bfppos_endopos_rep2_b = pd.read_table(bfppos_endopos_rep2_b_f, sep="\t")

bfppos_endopos_rep1 = bfppos_endopos_lib1_rep1_a.merge(bfppos_endopos_lib1_rep1_b, 
                                                       on="sgRNA").merge(bfppos_endopos_lib2_rep1_a,
                                                                         on="sgRNA").merge(bfppos_endopos_lib2_rep1_b,
                                                                                           on="sgRNA")
bfppos_endopos_rep2 = bfppos_endopos_rep2_a.merge(bfppos_endopos_rep2_b, on="sgRNA")

bfppos_endopos_rep1.columns = ["sgRNA", "lib1_rep1_a", "lib1_rep1_b", "lib2_rep1_a", "lib2_rep1_b"]
bfppos_endopos_rep2.columns = ["sgRNA", "lib1_rep2_a", "lib1_rep2_b"]
bfppos_endopos_rep1.sort_values(by="sgRNA").head()


# In[12]:


bfppos_endoneg_lib1_rep1_a = pd.read_table(bfppos_endoneg_lib1_rep1_a_f, sep="\t")
bfppos_endoneg_lib2_rep1_a = pd.read_table(bfppos_endoneg_lib2_rep1_a_f, sep="\t")
bfppos_endoneg_rep2_a = pd.read_table(bfppos_endoneg_rep2_a_f, sep="\t")
bfppos_endoneg_rep2_b = pd.read_table(bfppos_endoneg_rep2_b_f, sep="\t")

bfppos_endoneg_rep1 = bfppos_endoneg_lib1_rep1_a.merge(bfppos_endoneg_lib2_rep1_a, on="sgRNA")
bfppos_endoneg_rep2 = bfppos_endoneg_rep2_a.merge(bfppos_endoneg_rep2_b, on="sgRNA")

bfppos_endoneg_rep1.columns = ["sgRNA", "lib1_rep1_a", "lib2_rep1_a"]
bfppos_endoneg_rep2.columns = ["sgRNA", "lib1_rep2_a", "lib1_rep2_b"]
bfppos_endoneg_rep1.sort_values(by="sgRNA").head()


# In[13]:


index = pd.read_table(index_f, sep="\t")


# ## 2. compare technical replicates

# In[14]:


all_dfs = {"Day__Zero": [day0_rep1, day0_rep2], "BFP+__Endo++": [bfppos_endopos_rep1, bfppos_endopos_rep2], 
           "BFP+__Endo--": [bfppos_endoneg_rep1, bfppos_endoneg_rep2]}

for name in all_dfs:
    dfs = all_dfs[name]
    for i, df in enumerate(dfs):
        cols = [x for x in df.columns if x != "sgRNA"]
        
        # log transform data for plotting
        tmp = df.copy()
        tmp[cols] = np.log10(tmp[cols] + 1)
        
        # compare A & B seq from same library
        uniq_libs = list(set([x[:-2] for x in cols]))
        
        # edit cols for heatmap below
        new_cols = ["sgRNA"]
        cols = ["%s__%s" % (name, x) for x in cols]
        new_cols.extend(cols)
        df.columns = new_cols
        
        if "Endo--" in name and i == 0:
            continue
        
        for lib in uniq_libs:
            if "lib3" in lib:
                continue
                
            print("%s, rep %s, lib: %s" % (name, i+1, lib))
            col1 = "%s_a" % lib
            col2 = "%s_b" % lib
            
            # plot
            g = sns.jointplot(tmp[col1], tmp[col2], color="gray", size=2.2,
                              marginal_kws=dict(bins=15),
                              joint_kws=dict(s=5, rasterized=True))
            g.set_axis_labels("log10(%s (Set A) + 1) counts" % lib, "log10(%s (Set B) + 1) counts" % lib)
            
            # correlation
            r, p = spearmanr(tmp[col1], tmp[col2])
            g.ax_joint.annotate( "r = {:.2f}".format(r), ha="left", xy=(0.1, .90), xycoords=g.ax_joint.transAxes, 
                                fontsize=fontsize)

            #g.savefig("%s_%s_lib_corr_scatter.pdf" % (name, lib), dpi="figure", bbox_inches="tight")
            plt.show()


# In[15]:


bfppos_counts = day0_rep1.merge(day0_rep2, on="sgRNA").merge(bfppos_endopos_rep1, on="sgRNA").merge(bfppos_endopos_rep2, on="sgRNA").merge(bfppos_endoneg_rep1, on="sgRNA").merge(bfppos_endoneg_rep2, on="sgRNA")


# In[16]:


bfppos_counts.set_index("sgRNA", inplace=True)
bfppos_counts_corr = bfppos_counts.corr(method="spearman")


# In[17]:


cmap = sns.cubehelix_palette(as_cmap=True)
cg = sns.clustermap(bfppos_counts_corr, figsize=(3.75, 3.75), cmap=cmap, annot=False, vmin=0.6)
_ = plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.suptitle("spearman correlation of all replicates (incl technical)\ncounts of all barcodes")
#cg.savefig("BFP+_COUNTS__rep_and_lib_corr_heatmap.pdf", dpi="figure", bbox_inches="tight")


# ## 3. sum technical replicates

# ### day 0

# In[18]:


day0_rep1["lib1_rep1"] = day0_rep1["Day__Zero__lib1_rep1_a"] + day0_rep1["Day__Zero__lib1_rep1_b"]
day0_rep1.drop(["Day__Zero__lib1_rep1_a", "Day__Zero__lib1_rep1_b"], axis=1, inplace=True)

day0_rep1["lib2_rep1"] = day0_rep1["Day__Zero__lib2_rep1_a"] + day0_rep1["Day__Zero__lib2_rep1_b"]
day0_rep1.drop(["Day__Zero__lib2_rep1_a", "Day__Zero__lib2_rep1_b"], axis=1, inplace=True)

day0_rep1["lib3_rep1"] = day0_rep1["Day__Zero__lib3_rep1_a"] + day0_rep1["Day__Zero__lib3_rep1_b"]
day0_rep1.drop(["Day__Zero__lib3_rep1_a", "Day__Zero__lib3_rep1_b"], axis=1, inplace=True)

day0_rep1["Day_Zero__rep1"] = day0_rep1["lib1_rep1"] + day0_rep1["lib2_rep1"] + day0_rep1["lib3_rep1"]
day0_rep1.drop(["lib1_rep1", "lib2_rep1", "lib3_rep1"], axis=1, inplace=True)
day0_rep1.head()


# In[19]:


day0_rep2["lib1_rep2"] = day0_rep2["Day__Zero__lib1_rep2_a"] + day0_rep2["Day__Zero__lib1_rep2_b"]
day0_rep2.drop(["Day__Zero__lib1_rep2_a", "Day__Zero__lib1_rep2_b"], axis=1, inplace=True)

day0_rep2["lib2_rep2"] = day0_rep2["Day__Zero__lib2_rep2_a"] + day0_rep2["Day__Zero__lib2_rep2_b"]
day0_rep2.drop(["Day__Zero__lib2_rep2_a", "Day__Zero__lib2_rep2_b"], axis=1, inplace=True)

day0_rep2["Day_Zero__rep2"] = day0_rep2["lib1_rep2"] + day0_rep2["lib2_rep2"] + day0_rep2["Day__Zero__lib3_rep2_a"]
day0_rep2.drop(["lib1_rep2", "lib2_rep2", "Day__Zero__lib3_rep2_a"], axis=1, inplace=True)
day0_rep2.head()


# In[20]:


day0 = day0_rep1.merge(day0_rep2, on="sgRNA")
day0.head()


# ### BFP+ endo++

# In[21]:


bfppos_endopos_rep1["BFP+_Endo++__rep1"] = bfppos_endopos_rep1["BFP+__Endo++__lib1_rep1_a"] + bfppos_endopos_rep1["BFP+__Endo++__lib1_rep1_b"] + bfppos_endopos_rep1["BFP+__Endo++__lib2_rep1_a"] + bfppos_endopos_rep1["BFP+__Endo++__lib2_rep1_b"]
bfppos_endopos_rep1.drop(["BFP+__Endo++__lib1_rep1_a", "BFP+__Endo++__lib1_rep1_b", "BFP+__Endo++__lib2_rep1_a", "BFP+__Endo++__lib2_rep1_b"], axis=1, inplace=True)

bfppos_endopos_rep2["BFP+_Endo++__rep2"] = bfppos_endopos_rep2["BFP+__Endo++__lib1_rep2_a"] + bfppos_endopos_rep2["BFP+__Endo++__lib1_rep2_b"]
bfppos_endopos_rep2.drop(["BFP+__Endo++__lib1_rep2_a", "BFP+__Endo++__lib1_rep2_b"], axis=1, inplace=True)

bfppos_endopos = bfppos_endopos_rep1.merge(bfppos_endopos_rep2, on="sgRNA")
bfppos_endopos.head()


# ### BFP+ endo--

# In[22]:


bfppos_endoneg_rep1["BFP+_Endo--__rep1"] = bfppos_endoneg_rep1["BFP+__Endo--__lib1_rep1_a"] + bfppos_endoneg_rep1["BFP+__Endo--__lib2_rep1_a"]
bfppos_endoneg_rep1.drop(["BFP+__Endo--__lib1_rep1_a", "BFP+__Endo--__lib2_rep1_a"], axis=1, inplace=True)

bfppos_endoneg_rep2["BFP+_Endo--__rep2"] = bfppos_endoneg_rep2["BFP+__Endo--__lib1_rep2_a"] + bfppos_endoneg_rep2["BFP+__Endo--__lib1_rep2_b"]
bfppos_endoneg_rep2.drop(["BFP+__Endo--__lib1_rep2_a", "BFP+__Endo--__lib1_rep2_b"], axis=1, inplace=True)

bfppos_endoneg = bfppos_endoneg_rep1.merge(bfppos_endoneg_rep2, on="sgRNA")
bfppos_endoneg.head()


# ## 4. compare biological replicates

# In[23]:


all_dfs = {"Day_Zero": day0, "BFP+_Endo++": bfppos_endopos, "BFP+_Endo--": bfppos_endoneg}

for name in all_dfs:
    print(name)
    df = all_dfs[name]
    cols = [x for x in df.columns if x != "sgRNA"]
        
    # log transform data for plotting
    tmp = df.copy()
    tmp[cols] = np.log10(tmp[cols] + 1)
            
    # plot
    g = sns.jointplot(tmp[cols[0]], tmp[cols[1]], color="gray", size=2.2,
                      marginal_kws=dict(bins=15),
                      joint_kws=dict(s=5, rasterized=True))
    g.set_axis_labels("log10(rep_1 + 1) counts", "log10(rep_2 + 1) counts")

    # correlation
    r, p = spearmanr(tmp[cols[0]], tmp[cols[1]])
    g.ax_joint.annotate( "r = {:.2f}".format(r), ha="left", xy=(0.1, .90), xycoords=g.ax_joint.transAxes, 
                        fontsize=fontsize)

    #g.savefig("%s_rep_corr_scatter.pdf" % (name), dpi="figure", bbox_inches="tight")
    plt.show()


# In[24]:


bfppos_counts = day0.merge(bfppos_endopos, on="sgRNA").merge(bfppos_endoneg, on="sgRNA")


# In[25]:


tmp = bfppos_counts.set_index("sgRNA")
bfppos_counts_corr = tmp.corr(method="spearman")


# In[26]:


cg = sns.clustermap(bfppos_counts_corr, figsize=(2.5, 2.5), cmap=cmap, annot=True, vmin=0.6)
_ = plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.suptitle("spearman correlation of all replicates\ncounts of all barcodes")
cg.savefig("FigS5B.pdf", dpi="figure", bbox_inches="tight")


# ## 5. normalize for sequencing depth

# In[27]:


all_counts = pseudocount(bfppos_counts)
all_counts = to_cpm(all_counts)
all_counts.head()


# In[28]:


all_counts_norm = all_counts


# ## 6. join w/ index and write file

# In[29]:


data = index.merge(all_counts_norm, on="sgRNA")
len(data)


# In[30]:


data.sample(5)


# In[31]:


# write column with enrichment per guide (just foldchange between endo-- and endo++)
data["BFP+_enrichment__rep1"] = data["BFP+_Endo--__rep1"]/data["BFP+_Endo++__rep1"]
data["BFP+_enrichment__rep2"] = data["BFP+_Endo--__rep2"]/data["BFP+_Endo++__rep2"]

data["BFP+_enrichment__mean"] = data[["BFP+_enrichment__rep1", "BFP+_enrichment__rep2"]].mean(axis=1)
data.replace(np.inf, np.nan, inplace=True)

data = data.sort_values(by="BFP+_enrichment__mean", ascending=False)
data.head()


# In[32]:


f = "../../../data/02__screen/01__normalized_counts/normalized_sgRNA_counts.txt"
data.to_csv(f, sep="\t", index=False)

