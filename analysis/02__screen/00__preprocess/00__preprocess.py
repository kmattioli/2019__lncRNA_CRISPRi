
# coding: utf-8

# # 00__preprocess
# 
# in this notebook, i upload all individual library/primer sgRNA counts and sum together (after checking correlations) to make 1 dataframe with all sgRNA counts across populations (Day Zero, Endo++, Endo--). i also convert counts to cpms for downstream analyses
# 
# figures in this notebook:
# - Fig S5A (scatterplot histogram of replicate correspondence)

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import sys

from ast import literal_eval
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


index_f = "../../../data/01__design/02__final_sgRNAs/crispri.clean_index.txt"


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


# In[15]:


bfppos_counts = day0_rep1.merge(day0_rep2, on="sgRNA").merge(bfppos_endopos_rep1, on="sgRNA").merge(bfppos_endopos_rep2, on="sgRNA").merge(bfppos_endoneg_rep1, on="sgRNA").merge(bfppos_endoneg_rep2, on="sgRNA")


# In[16]:


bfppos_counts.set_index("sgRNA", inplace=True)
bfppos_counts.sum(axis=0)


# ## 3. sum technical replicates

# ### day 0

# In[17]:


day0_rep1["lib1_rep1"] = day0_rep1["Day__Zero__lib1_rep1_a"] + day0_rep1["Day__Zero__lib1_rep1_b"]
day0_rep1.drop(["Day__Zero__lib1_rep1_a", "Day__Zero__lib1_rep1_b"], axis=1, inplace=True)

day0_rep1["lib2_rep1"] = day0_rep1["Day__Zero__lib2_rep1_a"] + day0_rep1["Day__Zero__lib2_rep1_b"]
day0_rep1.drop(["Day__Zero__lib2_rep1_a", "Day__Zero__lib2_rep1_b"], axis=1, inplace=True)

day0_rep1["lib3_rep1"] = day0_rep1["Day__Zero__lib3_rep1_a"] + day0_rep1["Day__Zero__lib3_rep1_b"]
day0_rep1.drop(["Day__Zero__lib3_rep1_a", "Day__Zero__lib3_rep1_b"], axis=1, inplace=True)

day0_rep1["Day_Zero__rep1"] = day0_rep1["lib1_rep1"] + day0_rep1["lib2_rep1"] + day0_rep1["lib3_rep1"]

day0_rep1.drop(["lib1_rep1", "lib2_rep1", "lib3_rep1"], axis=1, inplace=True)
day0_rep1.head()


# In[18]:


day0_rep2["lib1_rep2"] = day0_rep2["Day__Zero__lib1_rep2_a"] + day0_rep2["Day__Zero__lib1_rep2_b"]
day0_rep2.drop(["Day__Zero__lib1_rep2_a", "Day__Zero__lib1_rep2_b"], axis=1, inplace=True)

day0_rep2["lib2_rep2"] = day0_rep2["Day__Zero__lib2_rep2_a"] + day0_rep2["Day__Zero__lib2_rep2_b"]
day0_rep2.drop(["Day__Zero__lib2_rep2_a", "Day__Zero__lib2_rep2_b"], axis=1, inplace=True)

day0_rep2["Day_Zero__rep2"] = day0_rep2["lib1_rep2"] + day0_rep2["lib2_rep2"] + day0_rep2["Day__Zero__lib3_rep2_a"]
day0_rep2.drop(["lib1_rep2", "lib2_rep2", "Day__Zero__lib3_rep2_a"], axis=1, inplace=True)
day0_rep2.head()


# In[19]:


day0 = day0_rep1.merge(day0_rep2, on="sgRNA")
day0.head()


# ### BFP+ endo++

# In[20]:


bfppos_endopos_rep1["BFP+_Endo++__rep1"] = bfppos_endopos_rep1["BFP+__Endo++__lib1_rep1_a"] + bfppos_endopos_rep1["BFP+__Endo++__lib1_rep1_b"] + bfppos_endopos_rep1["BFP+__Endo++__lib2_rep1_a"] + bfppos_endopos_rep1["BFP+__Endo++__lib2_rep1_b"]

bfppos_endopos_rep1.drop(["BFP+__Endo++__lib1_rep1_a", "BFP+__Endo++__lib1_rep1_b", "BFP+__Endo++__lib2_rep1_a", "BFP+__Endo++__lib2_rep1_b"], axis=1, inplace=True)

bfppos_endopos_rep2["BFP+_Endo++__rep2"] = bfppos_endopos_rep2["BFP+__Endo++__lib1_rep2_a"] + bfppos_endopos_rep2["BFP+__Endo++__lib1_rep2_b"]
bfppos_endopos_rep2.drop(["BFP+__Endo++__lib1_rep2_a", "BFP+__Endo++__lib1_rep2_b"], axis=1, inplace=True)

bfppos_endopos = bfppos_endopos_rep1.merge(bfppos_endopos_rep2, on="sgRNA")
bfppos_endopos.head()


# ### BFP+ endo--

# In[21]:


bfppos_endoneg_rep1["BFP+_Endo--__rep1"] = bfppos_endoneg_rep1["BFP+__Endo--__lib1_rep1_a"] + bfppos_endoneg_rep1["BFP+__Endo--__lib2_rep1_a"]
bfppos_endoneg_rep1.drop(["BFP+__Endo--__lib1_rep1_a", "BFP+__Endo--__lib2_rep1_a"], axis=1, inplace=True)

bfppos_endoneg_rep2["BFP+_Endo--__rep2"] = bfppos_endoneg_rep2["BFP+__Endo--__lib1_rep2_a"] + bfppos_endoneg_rep2["BFP+__Endo--__lib1_rep2_b"]
bfppos_endoneg_rep2.drop(["BFP+__Endo--__lib1_rep2_a", "BFP+__Endo--__lib1_rep2_b"], axis=1, inplace=True)

bfppos_endoneg = bfppos_endoneg_rep1.merge(bfppos_endoneg_rep2, on="sgRNA")
bfppos_endoneg.head()


# ## 4. compare overlap of most abundant sgRNAs across reps

# In[22]:


bfppos_counts = day0.merge(bfppos_endopos, on="sgRNA").merge(bfppos_endoneg, on="sgRNA")


# In[23]:


len(bfppos_counts)


# In[24]:


cols = ["Day_Zero__rep1", "Day_Zero__rep2", "BFP+_Endo++__rep1", "BFP+_Endo++__rep2", 
        "BFP+_Endo--__rep1", "BFP+_Endo--__rep2"]
arr = np.zeros((len(cols), len(cols)))
for i, repa in enumerate(cols):
    for j, repb in enumerate(cols):
        top_a = set(bfppos_counts.sort_values(by=repa, ascending=False).head(100)["sgRNA"])
        top_b = set(bfppos_counts.sort_values(by=repb, ascending=False).head(100)["sgRNA"])
        a_b = len(top_a.intersection(top_b))
        arr[i, j] = int(a_b)
df = pd.DataFrame(arr)
df.columns = ["Day Zero\n(rep 1)", "Day Zero\n(rep 2)", "Differentiated\n(rep 1)", "Differentiated\n(rep 2)",
              "Undifferentiated\n(rep 1)", "Undifferentiated\n(rep 2)"]
df.index = ["Day Zero\n(rep 1)", "Day Zero\n(rep 2)", "Differentiated\n(rep 1)", "Differentiated\n(rep 2)",
              "Undifferentiated\n(rep 1)", "Undifferentiated\n(rep 2)"]
df = df.astype(int)
df.head()


# In[25]:


cg = sns.clustermap(df, figsize=(2.75, 2.75), cmap="Greys", annot=True, **{"cbar": False}, fmt="g", vmax=70)
cg.cax.set_visible(False)
_ = plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
_ = plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=60, ha="right", va="top")
#cg.savefig("FigS5X.pdf", dpi="figure", bbox_inches="tight")


# ## 5. report read count sums across conditions

# In[26]:


bfppos_counts[["Day_Zero__rep1", "Day_Zero__rep2", "BFP+_Endo++__rep1", "BFP+_Endo++__rep2",
               "BFP+_Endo--__rep1", "BFP+_Endo--__rep2"]].sum(axis=0)


# ## 6. write counts file for DESeq2 input

# In[27]:


# write summed counts to file for DESeq2
bfppos_counts.columns = ["sgRNA", "DZ_Rep1", "DZ_Rep2", "Pos_Rep1", "Pos_Rep2", "Neg_Rep1", "Neg_Rep2"]
bfppos_counts.to_csv("../../../data/02__screen/00__counts/Biol_Reps.sgRNA_counts.txt", sep="\t", index=False)


# In[28]:


# filter out any sgRNAs w/ < 10 counts in either day zero replicate
filt_bfppos_counts = bfppos_counts[(bfppos_counts["DZ_Rep1"] >= 10) & ((bfppos_counts["DZ_Rep2"] >= 10))]
print(len(filt_bfppos_counts))
filt_bfppos_counts.to_csv("../../../data/02__screen/00__counts/Biol_Reps.sgRNA_counts.DZ_filt.txt", sep="\t", index=False)


# ## 7. normalize counts to cpm for downstream analyses

# In[29]:


# DZ_Rep1_rc = bfppos_counts["DZ_Rep1"].sum()
# DZ_Rep2_rc = bfppos_counts["DZ_Rep2"].sum()
# Pos_Rep1_rc = bfppos_counts["Pos_Rep1"].sum()
# Pos_Rep2_rc = bfppos_counts["Pos_Rep2"].sum()
# Neg_Rep1_rc= bfppos_counts["Neg_Rep1"].sum()
# Neg_Rep2_rc = bfppos_counts["Neg_Rep2"].sum()

# print("DZ Rep 1 RC: %s" % DZ_Rep1_rc)
# print("DZ Rep 2 RC: %s" % DZ_Rep2_rc)
# print("Pos Rep 1 RC: %s" % Pos_Rep1_rc)
# print("Pos Rep 2 RC: %s" % Pos_Rep2_rc)
# print("Neg Rep 1 RC: %s" % Neg_Rep1_rc)
# print("Neg Rep 2 RC: %s" % Neg_Rep2_rc)


# In[30]:


# DZ_Rep1_cellnum = 400e6
# DZ_Rep2_cellnum = 782e6
# Pos_Rep1_cellnum = 37.2e6
# Pos_Rep2_cellnum = 69.4e6
# Neg_Rep1_cellnum = 975000
# Neg_Rep2_cellnum = 1.5e6

# max_cell_num = np.max([DZ_Rep1_cellnum, DZ_Rep2_cellnum, Pos_Rep1_cellnum, Pos_Rep2_cellnum, Neg_Rep1_cellnum,
#                        Neg_Rep2_cellnum])

# DZ_Rep1_nf = max_cell_num / DZ_Rep1_cellnum
# DZ_Rep2_nf = max_cell_num / DZ_Rep2_cellnum
# Pos_Rep1_nf = max_cell_num / Pos_Rep1_cellnum
# Pos_Rep2_nf = max_cell_num / Pos_Rep2_cellnum
# Neg_Rep1_nf = max_cell_num / Neg_Rep1_cellnum
# Neg_Rep2_nf = max_cell_num / Neg_Rep2_cellnum

# print("DZ Rep 1 NF: %s" % DZ_Rep1_nf)
# print("DZ Rep 2 NF: %s" % DZ_Rep2_nf)
# print("Pos Rep 1 NF: %s" % Pos_Rep1_nf)
# print("Pos Rep 2 NF: %s" % Pos_Rep2_nf)
# print("Neg Rep 1 NF: %s" % Neg_Rep1_nf)
# print("Neg Rep 2 NF: %s" % Neg_Rep2_nf)


# In[31]:


# all_norm_counts = bfppos_counts.copy()
# all_norm_counts["DZ_Rep1"] = (all_norm_counts["DZ_Rep1"]/DZ_Rep1_rc/DZ_Rep1_nf)*1e6
# all_norm_counts["DZ_Rep2"] = (all_norm_counts["DZ_Rep2"]/DZ_Rep2_rc/DZ_Rep2_nf)*1e6
# all_norm_counts["Pos_Rep1"] = (all_norm_counts["Pos_Rep1"]/Pos_Rep1_rc/Pos_Rep1_nf)*1e6
# all_norm_counts["Pos_Rep2"] = (all_norm_counts["Pos_Rep2"]/Pos_Rep2_rc/Pos_Rep2_nf)*1e6
# all_norm_counts["Neg_Rep1"] = (all_norm_counts["Neg_Rep1"]/Neg_Rep1_rc/Neg_Rep1_nf)*1e6
# all_norm_counts["Neg_Rep2"] = (all_norm_counts["Neg_Rep2"]/Neg_Rep2_rc/Neg_Rep2_nf)*1e6
# all_norm_counts.max(axis=0)


# In[32]:


all_cpm_counts = pseudocount(bfppos_counts)
all_cpm_counts = to_cpm(all_cpm_counts)
all_cpm_counts.head()


# In[33]:


len(all_cpm_counts)


# In[34]:


tmp = all_cpm_counts.set_index("sgRNA")
print(len(tmp))
print(tmp.corr(method="spearman"))


# In[35]:


g = sns.jointplot(data=np.log10(tmp+1), x="DZ_Rep1", y="DZ_Rep2",
                  kind="hist", height=1.5, color="black", bins=20, marginal_kws={"bins": 20})
g.set_axis_labels("Day Zero (rep 1)\n(log10 cpm)", "Day Zero (rep 2)\n(log10 cpm)")
g.savefig("FigS5A_1.pdf", dpi="figure", bbox_inches="tight")


# In[36]:


g = sns.jointplot(data=np.log10(tmp+1), x="Pos_Rep1", y="Pos_Rep2",
                  kind="hist", height=1.5, color="black", bins=20, marginal_kws={"bins": 20})
g.set_axis_labels("Differentiated (rep 1)\n(log10 cpm)", "Differentiated (rep 2)\n(log10 cpm)")
g.savefig("FigS5A_2.pdf", dpi="figure", bbox_inches="tight")


# In[37]:


g = sns.jointplot(data=np.log10(tmp+1), x="Neg_Rep1", y="Neg_Rep2",
                  kind="hist", height=1.5, color="black", bins=20, marginal_kws={"bins": 20})
g.set_axis_labels("Undifferentiated (rep 1)\n(log10 cpm)", "Undifferentiated (rep 2)\n(log10 cpm)")
g.savefig("FigS5A_3.pdf", dpi="figure", bbox_inches="tight")


# ## 8. join w/ index & write full table

# In[39]:


data = bfppos_counts.merge(all_cpm_counts, on="sgRNA", suffixes=("__counts", "__cpm"))
print(len(data))
data.head()


# In[40]:


index.columns


# In[41]:


index_sub = index[["sgRNA", "tss_id_hg38", "transcript_id", "transcript_name", "gene_id", "gene_name",
                   "tss_type", "cage_id_hg19", "rank", "ctrl_status_fixed", "transcript_biotype_status",
                   "sgRNA_id"]]
len(index_sub)


# In[42]:


index_sub["refseq"] = index_sub["tss_id_hg38"].str.split(":", expand=True)[0]
index_sub["tss_start_hg38"] = index_sub["tss_id_hg38"].str.split(":", expand=True)[2]
index_sub["tss_strand_hg38"] = index_sub["tss_id_hg38"].str.split(":", expand=True)[1]


# In[43]:


f = "../../../misc/05__refseq/chr_to_refseq.txt"
refseq = pd.read_table(f, sep="\t", header=None)
refseq.columns = ["tss_chr_hg38", "refseq"]


# In[44]:


index_sub = index_sub.merge(refseq, on="refseq", how="left")
len(index_sub)


# In[45]:


def fix_starts(row):
    if row["ctrl_status_fixed"] == "scramble":
        return np.nan
    elif "," in row["tss_start_hg38"]:
        all_ids = literal_eval(row["tss_id_hg38"])
        s = []
        for i in all_ids:
            start = i.split(":")[2]
            s.append(start)
        return s
    else:
        return row["tss_start_hg38"]
    
index_sub["tss_start_hg38"] = index_sub.apply(fix_starts, axis=1)
index_sub[pd.isnull(index_sub["tss_chr_hg38"])]       


# In[46]:


def fix_null_chrs(row):
    if pd.isnull(row["tss_chr_hg38"]):
        if row["ctrl_status_fixed"] == "scramble":
            return "scramble"
        else:
            all_ids = literal_eval(row["tss_id_hg38"])
            idx = all_ids[0]
            idx_refseq = idx.split(":")[0]
            chrom = refseq[refseq["refseq"] == idx_refseq]["tss_chr_hg38"].iloc[0]
            return chrom
    else:
        return row["tss_chr_hg38"]

index_sub["tss_chr_hg38"] = index_sub.apply(fix_null_chrs, axis=1)
index_sub[pd.isnull(index_sub["tss_chr_hg38"])]    


# In[47]:


index_sub = index_sub[["sgRNA", "ctrl_status_fixed", "gene_id", "gene_name", "transcript_id", "transcript_name", 
                       "transcript_biotype_status", "tss_chr_hg38", "tss_start_hg38", "tss_strand_hg38", "tss_type",
                       "tss_id_hg38", "sgRNA_id"]]
index_sub.columns = ["sgRNA", "ctrl_status", "gene_id", "gene_name", "transcript_id", "transcript_name", 
                     "transcript_biotype_status", "tss_chr_hg38", "tss_start_hg38", "tss_strand_hg38", "tss_type",
                     "tss_id_hg38", "sgRNA_id"]
print(len(index_sub))
index_sub.head()


# In[48]:


data = index_sub.merge(data, on="sgRNA")
print(len(data))
data.head()


# In[49]:


data_f = "../../../data/02__screen/01__normalized_counts/Biol_Reps.sgRNA_counts.w_index.txt"
data.to_csv(data_f, sep="\t", index=False)

