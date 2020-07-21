#!/usr/bin/env python
# coding: utf-8

# # 01__guide_analysis
# 
# in this notebook, i count the number of sgRNAs in our library based on their off-target classification, as defined by the GPP sgRNA designer: https://portals.broadinstitute.org/gpp/public/software/sgrna-scoring-help
# 
# no figures or tables in this notebook

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

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
mpl.rcParams['figure.autolayout'] = False


# In[2]:


sns.set(**PAPER_PRESET)
fontsize = PAPER_FONTSIZE


# ## variables

# In[3]:


sgrnas_dir = "../../../data/01__design/01__sgRNAs_from_Broad"
main_grna_f = "%s/Mattioli_CRISPRi_designs_12132017.txt.gz" % sgrnas_dir


# In[4]:


lib_f = "../../../data/01__design/02__final_sgRNAs/crispri_with_primers.txt"


# In[6]:


data_f = "../../../data/02__screen/02__enrichment_data/SuppTable_S2.sgRNA_results.txt"


# In[7]:


val_f = "../../../data/02__screen/03__validation_data/validation_data.xlsx"


# ## 1. import data

# In[8]:


main_grnas = pd.read_table(main_grna_f)
len(main_grnas)


# In[9]:


# import other grna files that were run later
extras = ["1", "2", "3", "4", "digit"]
extra_crispri = pd.DataFrame()
for suf in extras:
    crispri_filename = "%s/crispri_%s.txt" % (sgrnas_dir, suf)
    crispri_tmp = pd.read_table(crispri_filename, sep="\t")
    extra_crispri = extra_crispri.append(crispri_tmp)


# In[10]:


cols = main_grnas.columns
sgrnas = main_grnas.append(extra_crispri)
sgrnas = sgrnas[cols]
len(sgrnas)


# In[11]:


lib = pd.read_table(lib_f)
len(lib)


# In[12]:


data = pd.read_table(data_f)
len(data)


# In[13]:


data.sgRNA_status.value_counts()


# In[14]:


filt = data[data["sgRNA_status"] == "filter met"]
len(filt)


# In[15]:


val = pd.read_excel(val_f)
val.head()


# ## 2. merge off-target info w/ guides in library

# In[16]:


sgrnas.columns


# In[17]:


lib = lib.merge(sgrnas[["sgRNA Sequence", "# Off-Target Tier I Match Bin I Matches", 
                        "# Off-Target Tier II Match Bin I Matches", "# Off-Target Tier III Match Bin I Matches",
                        "# Off-Target Tier I Match Bin II Matches", "# Off-Target Tier II Match Bin II Matches",
                        "# Off-Target Tier III Match Bin II Matches", "# Off-Target Tier I Match Bin III Matches",
                        "# Off-Target Tier II Match Bin III Matches", "# Off-Target Tier III Match Bin III Matches",
                        "Off-Target Rank", "Off-Target Rank Weight", "sgRNA 'Cut' Site TSS Offset"]], left_on="sgRNA", 
                right_on="sgRNA Sequence", how="left")
lib.sample(5)


# In[18]:


# remove scrambles
lib_ns = lib[lib["tss_id_hg38"] != "scramble"]
len(lib_ns)


# In[19]:


filt = filt.merge(sgrnas[["sgRNA Sequence", "# Off-Target Tier I Match Bin I Matches", 
                        "# Off-Target Tier II Match Bin I Matches", "# Off-Target Tier III Match Bin I Matches",
                        "# Off-Target Tier I Match Bin II Matches", "# Off-Target Tier II Match Bin II Matches",
                        "# Off-Target Tier III Match Bin II Matches", "# Off-Target Tier I Match Bin III Matches",
                        "# Off-Target Tier II Match Bin III Matches", "# Off-Target Tier III Match Bin III Matches",
                        "Off-Target Rank", "Off-Target Rank Weight", "sgRNA 'Cut' Site TSS Offset"]], left_on="sgRNA", 
                right_on="sgRNA Sequence", how="left")
filt.sample(5)


# In[20]:


# remove scrambles
filt_ns = filt[filt["control_status"] != "scramble"]
len(filt_ns)


# In[21]:


len(filt_ns.sgRNA.unique())


# ## 3. investigate off-target scores for guides

# In[22]:


tier_id = ["I", "II", "III"]
bin_id = ["I", "II"]

fig, axarr = plt.subplots(figsize=(5, 3), nrows=len(bin_id), ncols=len(tier_id), sharex=True, sharey=True)

for i in range(len(bin_id)):
    for j in range(len(tier_id)):
        ax = axarr[i, j]
        b = bin_id[i]
        t = tier_id[j]
        print("tier: %s, bin: %s" % (t, b))
        vals = list(lib_ns["# Off-Target Tier %s Match Bin %s Matches" % (t, b)])
        
        # some columns max out at 10 so make all cols max out at 10
        vals = [int(x) if str(x) != "MAX" else 10 for x in vals]
        print(np.min(vals))
        vals = [x if x < 9 else 10 for x in vals]
        print(np.max(vals))
        
        sns.distplot(vals, kde=False, ax=ax, bins=10)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_xticklabels(["0", "2", "4", "6", "8", "10+"])
        ax.set_title("Tier %s, Bin %s" % (t, b))
        
        if i == len(bin_id)-1:
            ax.set_xlabel("# predicted off-targets")
        if j == 0:
            ax.set_ylabel("# sgRNAs")

plt.tight_layout()


# In[23]:


lib.columns


# In[24]:


lib_ns_grp = lib_ns.drop(["tss_id_hg38", "transcript_id", "transcript_name", "gene_id", "gene_name", "tss_type",
                          "cage_id_hg19", "sgRNA_qual", "cut_offset", "rank", "sgRNA_id", "ctrl_status", "oligo",
                          "oligo_len", "sgRNA Sequence", "Off-Target Rank", "Off-Target Rank Weight"], axis=1)
lib_ns_grp = lib_ns_grp.drop_duplicates()
len(lib_ns_grp)


# In[25]:


print("TOTAL NON-SCRAMBLED GUIDES: %s" % len(lib_ns_grp))

n_tier1_bin1 = len(lib_ns_grp[lib_ns_grp["# Off-Target Tier I Match Bin I Matches"].astype(str) == '0'])
n_tier2_bin1 = len(lib_ns_grp[lib_ns_grp["# Off-Target Tier II Match Bin I Matches"].astype(str) == '0'])
n_tier3_bin1 = len(lib_ns_grp[lib_ns_grp["# Off-Target Tier III Match Bin I Matches"].astype(str) == '0'])

print("   %% w/ 0 bin 1 tier I off-targets: %s (%s total)" % (round(n_tier1_bin1/len(lib_ns_grp), 3), n_tier1_bin1))
print("   %% w/ 0 bin 1 tier II off-targets: %s (%s total)" % (round(n_tier2_bin1/len(lib_ns_grp), 3), n_tier2_bin1))
print("   %% w/ 0 bin 1 tier III off-targets: %s (%s total)" % (round(n_tier3_bin1/len(lib_ns_grp), 3), n_tier3_bin1))


# In[26]:


tier_id = ["I", "II", "III"]
bin_id = ["I", "II"]

fig, axarr = plt.subplots(figsize=(5, 3), nrows=len(bin_id), ncols=len(tier_id), sharex=True, sharey=True)

for i in range(len(bin_id)):
    for j in range(len(tier_id)):
        ax = axarr[i, j]
        b = bin_id[i]
        t = tier_id[j]
        print("tier: %s, bin: %s" % (t, b))
        vals = list(filt_ns["# Off-Target Tier %s Match Bin %s Matches" % (t, b)])
        
        # some columns max out at 10 so make all cols max out at 10
        vals = [int(x) if str(x) != "MAX" else 10 for x in vals]
        print(np.min(vals))
        vals = [x if x < 9 else 10 for x in vals]
        print(np.max(vals))
        
        sns.distplot(vals, kde=False, ax=ax, bins=10)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_xticklabels(["0", "2", "4", "6", "8", "10+"])
        ax.set_title("Tier %s, Bin %s" % (t, b))
        
        if i == len(bin_id)-1:
            ax.set_xlabel("# predicted off-targets")
        if j == 0:
            ax.set_ylabel("# sgRNAs")

plt.tight_layout()


# In[27]:


filt_ns.columns


# In[30]:


filt_ns_grp = filt_ns.drop(["control_status", "transcript_id", "gene_name", "DayZero_Rep1__counts", 
                            "DayZero_Rep2__counts", "Undiff_Rep1__counts", "Undiff_Rep2__counts", 
                            "Diff_Rep1__counts", "Diff_Rep2__counts", "sgRNA_status", 
                            "sgRNA_l2fc", "sgRNA_l2fcSE", 
                            "sgRNA Sequence", "Off-Target Rank", "Off-Target Rank Weight"], axis=1)
filt_ns_grp = filt_ns_grp.drop_duplicates()
len(filt_ns_grp)


# In[31]:


print("TOTAL NON-SCRAMBLED GUIDES: %s" % len(filt_ns_grp))

n_tier1_bin1 = len(filt_ns_grp[filt_ns_grp["# Off-Target Tier I Match Bin I Matches"].astype(str) == '0'])
n_tier2_bin1 = len(filt_ns_grp[filt_ns_grp["# Off-Target Tier II Match Bin I Matches"].astype(str) == '0'])
n_tier3_bin1 = len(filt_ns_grp[filt_ns_grp["# Off-Target Tier III Match Bin I Matches"].astype(str) == '0'])

print("   %% w/ 0 bin 1 tier I off-targets: %s (%s total)" % (round(n_tier1_bin1/len(filt_ns_grp), 3), n_tier1_bin1))
print("   %% w/ 0 bin 1 tier II off-targets: %s (%s total)" % (round(n_tier2_bin1/len(filt_ns_grp), 3), n_tier2_bin1))
print("   %% w/ 0 bin 1 tier III off-targets: %s (%s total)" % (round(n_tier3_bin1/len(filt_ns_grp), 3), n_tier3_bin1))


# ## 4. predicted off-targets of validated sgRNAs

# In[32]:


lib_ns_sub = lib_ns[["sgRNA", "# Off-Target Tier I Match Bin I Matches", 
                        "# Off-Target Tier II Match Bin I Matches", "# Off-Target Tier III Match Bin I Matches",
                        "# Off-Target Tier I Match Bin II Matches", "# Off-Target Tier II Match Bin II Matches",
                        "# Off-Target Tier III Match Bin II Matches", "# Off-Target Tier I Match Bin III Matches",
                        "# Off-Target Tier II Match Bin III Matches", "# Off-Target Tier III Match Bin III Matches"]]
val = val.merge(lib_ns_sub, left_on="guide_sequence", right_on="sgRNA", how="left")
val = val.drop_duplicates()
val[["gene_name", "guide_num", "guide_sequence", "# Off-Target Tier I Match Bin I Matches",
     "# Off-Target Tier II Match Bin I Matches"]]


# ## 5. predicted off-targets of FOXD3-AS1 sgRNAs

# In[35]:


filt_ns[filt_ns["transcript_name"] == "FOXD3-AS1-004"][["sgRNA", "gene_name", "transcript_name", "sgRNA_l2fc",
                                                        "# Off-Target Tier I Match Bin I Matches",
                                                        "# Off-Target Tier II Match Bin I Matches"]].sort_values(by="sgRNA_l2fc", ascending=False)


# ## 6. relationship b/w distance to TSS and guide performance

# In[36]:


lib_ns.columns


# In[37]:


lib_ns = lib_ns.merge(data[["sgRNA", "control_status", "sgRNA_l2fc", "sgRNA_l2fcSE"]], on="sgRNA")
lib_ns.sample(5)


# In[38]:


lib_ns["sgRNA 'Cut' Site TSS Offset"].min()


# In[39]:


lib_ns["sgRNA 'Cut' Site TSS Offset"].max()


# In[40]:


def bin_cut(row):
    if np.abs(row["sgRNA 'Cut' Site TSS Offset"]) <= 50:
        return "0-50"
    elif np.abs(row["sgRNA 'Cut' Site TSS Offset"]) <= 100:
        return "51-100"
    elif np.abs(row["sgRNA 'Cut' Site TSS Offset"]) <= 150:
        return "101-150"
    elif np.abs(row["sgRNA 'Cut' Site TSS Offset"]) <= 200:
        return "151-200"
    else:
        return "201+"
    
lib_ns["bin_cut"] = lib_ns.apply(bin_cut, axis=1)
lib_ns.bin_cut.value_counts()


# In[41]:


order = ["0-50", "51-100", "101-150", "151-200"]


# In[43]:


fig = plt.figure(figsize=(2, 2))

controls = lib_ns[lib_ns["control_status"] == "control"]
ax = sns.boxplot(data=controls, x="bin_cut", y="sgRNA_l2fc", order=order, flierprops=dict(marker='o', markersize=5),
                 color=sns.color_palette()[2])
mimic_r_boxplot(ax)

ax.set_xlabel("# bp between sgRNA PAM & annotated TSS")
ax.set_ylabel("log2 fold change\n(undifferentiated/differentiated)")
ax.set_title("positive control sgRNAs")


# In[ ]:




