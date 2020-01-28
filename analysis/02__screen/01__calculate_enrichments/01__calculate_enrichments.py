
# coding: utf-8

# # 01__calculate_enrichments
# 
# in this notebook, i filter sgRNAs and calculate transcript-level enrichment scores. (briefly: sgRNA-level enrichments are calculated by dividing their frequency in the undifferentiated population by their frequency in the differentiated population, and then the scores of the top 3 sgRNAs are averaged to get transcript-level enrichments).
# 
# figures in this notebook:
# - Fig 3C (scatter plot showing sgRNA frequencies in Endo-- replicates)
# - Fig 3D (boxplot comparing positive control sgRNA enrichments to negative control sgRNA enrichments)
# - Fig 3E (scatter plot showing sgRNA enrichments across replicates after filtering)
# - Fig S5A (density plots showing sgRNA counts in each sample)
# - Fig S5C (plots showing control rankings after setting Day Zero filter at various thresholds)

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
from enrich_utils import *

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
mpl.rcParams['figure.autolayout'] = False


# In[2]:


sns.set(**PAPER_PRESET)
fontsize = PAPER_FONTSIZE


# ## functions

# In[3]:


def get_gene_enrichments(df, thresh, min_sgRNAs_represented, top_sgRNAs_to_rank, n_sgRNAs_for_std, rank_col, enrich_col):
    
    # get col prefix
    prefix = rank_col.split("_")[0]
    rep = rank_col.split("__")[1]
    
    filt_df = df[df["Day_Zero__%s" % rep] >= thresh]
    
    
    if len(filt_df) == 0:
        return None

    # only examine genes with at least x sgRNAs represented
    df_grp = filt_df.groupby(["group_id", "ctrl_status", "gene_name", "transcript_name",
                              "endo_ctrl_val"])["sgRNA"].agg("count").reset_index()
    good_groups = df_grp[df_grp["sgRNA"] >= min_sgRNAs_represented]["group_id"]
    filt_df = filt_df[filt_df["group_id"].isin(good_groups)]

    # now take the top x sgRNAs per group_id
    enrich_sort = filt_df.sort_values(by=["group_id", enrich_col], ascending=False)[["group_id", 
                                                                                     "ctrl_status",
                                                                                     "endo_ctrl_val",
                                                                                     "gene_name",
                                                                                     "transcript_name",
                                                                                     enrich_col,
                                                                                     rank_col]]

    enrich_top = enrich_sort.groupby(["group_id", "ctrl_status", "endo_ctrl_val",
                                      "gene_name", "transcript_name"]).head(top_sgRNAs_to_rank)
    
    enrich_grp_med = enrich_top.groupby(["group_id", 
                                         "ctrl_status",
                                         "endo_ctrl_val",
                                         "gene_name", 
                                         "transcript_name"])[enrich_col].agg("median").reset_index()
    
    enrich_for_std = enrich_sort.groupby(["group_id", "ctrl_status", "endo_ctrl_val",
                                          "gene_name", "transcript_name"]).head(n_sgRNAs_for_std)
    enrich_grp_std = enrich_for_std.groupby(["group_id", 
                                             "ctrl_status", 
                                             "endo_ctrl_val",
                                             "gene_name", 
                                             "transcript_name"])[enrich_col].agg("std").reset_index()
    enrich_grp_std[enrich_col].replace(0, 1, inplace=True)
    enrich_grp = enrich_grp_med.merge(enrich_grp_std, on=["group_id", 
                                                          "ctrl_status", 
                                                          "endo_ctrl_val",
                                                          "gene_name", 
                                                          "transcript_name"],
                                      suffixes=("_med", "_std"))

    
    weighted_score_col = "%s_score__%s" % (prefix, rep)
    #enrich_grp[weighted_score_col] = enrich_grp["%s_med" % enrich_col] * (1-enrich_grp["%s_std" % enrich_col])
    enrich_grp[weighted_score_col] = enrich_grp["%s_med" % enrich_col]
    enrich_grp["%s_score_rank__%s" % (prefix, rep)] = enrich_grp[weighted_score_col].rank(ascending=False)
    
        
    return enrich_grp


# ## variables

# In[4]:


index_f = "../../../data/01__design/02__final_sgRNAs/crispri_with_primers.txt"


# In[5]:


endo_ctrls_validated_f = "../../../misc/04__pos_ctrls/endo_ctrls_validated.updated.txt"


# In[6]:


data_f = "../../../data/02__screen/01__normalized_counts/normalized_sgRNA_counts.txt"


# In[7]:


min_sgRNAs_represented = 3
top_sgRNAs_to_rank = 3
n_sgRNAs_for_std = 3


# ## 1. load data

# In[8]:


index = pd.read_table(index_f, sep="\t")
index.head()


# In[9]:


endo_ctrls_validated = pd.read_table(endo_ctrls_validated_f, sep="\t", header=None)
endo_ctrls_validated.columns = ["gene_name"]
print(len(endo_ctrls_validated))
endo_ctrls_validated.head()


# In[10]:


data = pd.read_table(data_f, sep="\t")
data_cols = [x for x in data.columns if "BFP+" in x or "Day_Zero" in x or x == "sgRNA"]
data_cols = [x for x in data_cols if "enrichment" not in x]
data = data[data_cols]
data.head()


# ## 2. add control ID information to index

# In[11]:


len(index.transcript_id.unique())


# In[12]:


# fix index ID columns that can have duplicates in them due to previous aggregation
index["gene_id"] = index.apply(fix_id_dupes, column="gene_id", axis=1)
index["gene_name"] = index.apply(fix_id_dupes, column="gene_name", axis=1)
index["transcript_id"] = index.apply(fix_id_dupes, column="transcript_id", axis=1)
index["transcript_name"] = index.apply(fix_id_dupes, column="transcript_name", axis=1)
index = index.drop_duplicates()
len(index.transcript_id.unique())


# In[13]:


# get control IDs
index["scramble_id"] = index.apply(get_scramble_id, axis=1)
index["group_id"] = index.apply(get_group_id, axis=1)
index["endo_ctrl_val"] = index["gene_name"].isin(endo_ctrls_validated["gene_name"])
index.endo_ctrl_val.value_counts()


# In[14]:


index["ctrl_status_fixed"] = index.apply(fix_ctrl_status_w_DE_mRNAs, axis=1)


# In[15]:


index.groupby("ctrl_status")["sgRNA"].agg("count")


# In[16]:


index.groupby("ctrl_status_fixed")["sgRNA"].agg("count")


# In[17]:


index[["tss_id_hg38", "ctrl_status_fixed"]].drop_duplicates().groupby("ctrl_status_fixed")["tss_id_hg38"].agg("count")


# In[25]:


tmp = index[["tss_id_hg38", "gene_name", "transcript_name", "ctrl_status_fixed"]].drop_duplicates()
ctrls_in_idx = tmp[tmp["ctrl_status_fixed"] == "control"]
ctrls_in_idx


# In[27]:


endo_ctrls_validated[~endo_ctrls_validated["gene_name"].isin(ctrls_in_idx["gene_name"])]


# ## 3. merge sgRNA counts w/ index

# In[18]:


data = data.merge(index, on="sgRNA")
data.sample(5)


# In[19]:


data.columns


# ## 3. calculate enrichments of endo-- over endo++ per sgRNA

# In[20]:


data["BFP+_enrichment__rep1"] = (data["BFP+_Endo--__rep1"])/(data["BFP+_Endo++__rep1"])
data["BFP+_log_enrichment__rep1"] = np.log2(data["BFP+_enrichment__rep1"])

data["BFP+_enrichment__rep2"] = (data["BFP+_Endo--__rep2"])/(data["BFP+_Endo++__rep2"])
data["BFP+_log_enrichment__rep2"] = np.log2(data["BFP+_enrichment__rep2"])

data["BFP+_enrichment__mean"] = data[["BFP+_enrichment__rep1", "BFP+_enrichment__rep2"]].mean(axis=1)
data["BFP+_log_enrichment__mean"] = np.log2(data["BFP+_enrichment__mean"])

data.sort_values(by="BFP+_log_enrichment__mean", ascending=False).head()


# ## 4. plot distributions of counts in each population

# In[21]:


cols = ["Day_Zero__rep1", "Day_Zero__rep2", "BFP+_Endo++__rep1", "BFP+_Endo++__rep2", 
        "BFP+_Endo--__rep1", "BFP+_Endo--__rep2"]
names = ["Day Zero (rep 1)", "Day Zero (rep 2)", "Differentiated Endoderm (rep 1)", "Differentiated Endoderm (rep 2)",
         "Undifferentiated (rep 1)", "Undifferentiated (rep 2)"]

c = 1
for col, name in zip(cols, names):
    
    fig = plt.figure(figsize=(2, 1))
    ax = sns.distplot(np.log10(data[col]+1), hist=False)
    ax.set_ylabel("density")
    ax.set_xlabel("log10(counts + 1)")
    ax.set_title(name)
    plt.show()
    fig.savefig("FigS5A_%s.pdf" % c, figure="dpi", bbox_inches="tight")
    plt.close()
    c += 1


# ## 5. investigate where main controls (FOXA2, SOX17, GATA6) rank based on day 0 cut-offs

# In[22]:


data["BFP+_rank__rep1"] = data["BFP+_log_enrichment__rep1"].rank(ascending=False)
data["BFP+_rank__rep2"] = data["BFP+_log_enrichment__rep2"].rank(ascending=False)
data["BFP+_rank__mean"] = data["BFP+_log_enrichment__mean"].rank(ascending=False)


# In[23]:


filt_threshs = list(np.linspace(2, 10, 17))
filt_threshs


# In[24]:


ctrls = ["FOXA2", "SOX17", "GATA6"]
all_ctrl_ranks = {}

enrich_cols = ["BFP+_log_enrichment__rep1", "BFP+_log_enrichment__rep2"]

rank_cols = ["BFP+_rank__rep1", "BFP+_rank__rep2"]

for enrich_col, rank_col in zip(enrich_cols, rank_cols):
    prefix = enrich_col.split("_")[0]
    rep = enrich_col.split("__")[1]
    out_col = "%s_score_rank__%s" % (prefix, rep)
    ctrl_ranks = {}
    for ctrl in ctrls:
        ranks_per_ctrl = {}
        for thresh in filt_threshs:
            all_enrich_grp = get_gene_enrichments(data, thresh, min_sgRNAs_represented, 
                                                  top_sgRNAs_to_rank, n_sgRNAs_for_std, rank_col, enrich_col)
            if len(all_enrich_grp) > 0:
                ctrl_df = all_enrich_grp[all_enrich_grp["gene_name"] == ctrl]
                if len(ctrl_df) > 0:
                    ctrl_rank = ctrl_df[out_col].iloc[0]
                else:
                    ctrl_rank = np.nan
            else:
                ctrl_rank = np.nan
            ranks_per_ctrl[thresh] = {"rank": ctrl_rank}
        ctrl_ranks[ctrl] = ranks_per_ctrl
    all_ctrl_ranks["%s_%s" % (prefix, rep)] = ctrl_ranks


# In[25]:


score_cols = ["BFP+_rep1", "BFP+_rep2"]

c = 1
for score_col in score_cols:
    FOXA2_ranks = pd.DataFrame.from_dict(all_ctrl_ranks[score_col]["FOXA2"]).T.reset_index()
    SOX17_ranks = pd.DataFrame.from_dict(all_ctrl_ranks[score_col]["SOX17"]).T.reset_index()
    GATA6_ranks = pd.DataFrame.from_dict(all_ctrl_ranks[score_col]["GATA6"]).T.reset_index()
    
    # plot
    fig = plt.figure(figsize=(2,1.5))
    plt.plot(FOXA2_ranks["index"], FOXA2_ranks["rank"], '--', color=sns.color_palette("Set2")[0], label="FOXA2")
    plt.plot(SOX17_ranks["index"], SOX17_ranks["rank"], '--', color=sns.color_palette("Set2")[1], label="SOX17")
    plt.plot(GATA6_ranks["index"], GATA6_ranks["rank"], '--', color=sns.color_palette("Set2")[2], label="GATA6")
    plt.ylabel("gene ranking based on sgRNA enrichment\n(top %s sgRNAs)" % top_sgRNAs_to_rank)
    plt.xlabel("count threshold (based on Day Zero)")
    plt.title(score_col)
    plt.axvline(x=5, color="gray")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()
    fig.savefig("FigS5C_%s.pdf" % c, dpi="figure", bbox_inches="tight")
    plt.close()
    c += 1


# ## 6. filter sgRNAs based on Day Zero counts outlined above

# In[26]:


cpm_thresh = 5


# In[27]:


all_enrich_grp_1 = get_gene_enrichments(data, cpm_thresh, min_sgRNAs_represented, 
                                      top_sgRNAs_to_rank, n_sgRNAs_for_std, "BFP+_rank__rep1", 
                                      "BFP+_enrichment__rep1")

all_enrich_grp_2 = get_gene_enrichments(data, cpm_thresh, min_sgRNAs_represented, 
                                      top_sgRNAs_to_rank, n_sgRNAs_for_std, "BFP+_rank__rep2", 
                                      "BFP+_enrichment__rep2")

all_enrich_grp = all_enrich_grp_1.merge(all_enrich_grp_2, on=["group_id", "ctrl_status",
                                                              "endo_ctrl_val", "gene_name", "transcript_name"])
all_enrich_grp.sample(5)


# In[28]:


all_enrich_grp["BFP+_score__mean"] = all_enrich_grp[["BFP+_score__rep1", "BFP+_score__rep2"]].mean(axis=1)
all_enrich_grp["BFP+_score_rank__mean"] = all_enrich_grp["BFP+_score__mean"].rank(ascending=False)
all_enrich_grp.sort_values(by="BFP+_score_rank__mean").head()


# ## 7. plot enrichment of scrambled genes vs. controls

# In[29]:


pal = {"control": sns.color_palette()[2], "experimental": "black", "scramble": "gray"}
sns.palplot(pal.values())


# In[30]:


data["ctrl_status_fixed"] = data.apply(fix_ctrl_status, axis=1)
all_enrich_grp["ctrl_status_fixed"] = all_enrich_grp.apply(fix_ctrl_status, axis=1)


# In[31]:


print(len(data))
print(len(all_enrich_grp))


# In[32]:


data_filt = data[(data["Day_Zero__rep1"] >= cpm_thresh) | (data["Day_Zero__rep2"] >= cpm_thresh)]
print(len(data_filt))
data_filt_top = data_filt.sort_values(by="BFP+_log_enrichment__mean", ascending=False).head(round(len(data_filt)/4))


# In[33]:


tmp = data_filt[data_filt["ctrl_status_fixed"].isin(["scramble", "control"])][["sgRNA", "ctrl_status_fixed",
                                                                               "BFP+_enrichment__rep1",
                                                                               "BFP+_enrichment__rep2"]]
tmp = pd.melt(tmp, id_vars=["sgRNA", "ctrl_status_fixed"])
tmp.head()


# In[34]:


hue_order = ["scramble", "control"]
order = ["BFP+_enrichment__rep1", "BFP+_enrichment__rep2"]
fig = plt.figure(figsize=(1.75, 2))
ax = sns.boxplot(data=tmp, x="variable", hue="ctrl_status_fixed", y="value", 
                 flierprops = dict(marker='o', markersize=5), order=order, hue_order=hue_order, palette=pal)
mimic_r_boxplot(ax)
ax.set_yscale("log")
ax.set_ylabel("")
ax.set_ylabel("sgRNA enrichment")
ax.set_xticklabels(["rep 1", "rep 2"])
ax.set_xlabel("")

# calc p-vals b/w dists
rep1 = tmp[tmp["variable"] == "BFP+_enrichment__rep1"]
rep1_dist1 = np.asarray(rep1[rep1["ctrl_status_fixed"] == "scramble"]["value"])
rep1_dist2 = np.asarray(rep1[rep1["ctrl_status_fixed"] == "control"]["value"])

rep1_dist1 = rep1_dist1[~np.isnan(rep1_dist1)]
rep1_dist2 = rep1_dist2[~np.isnan(rep1_dist2)]

u, pval = stats.mannwhitneyu(rep1_dist1, rep1_dist2, alternative="less", use_continuity=False)
print(pval)
annotate_pval(ax, -0.1, 0.1, 5, 0, 4.8, pval, fontsize)

rep2 = tmp[tmp["variable"] == "BFP+_enrichment__rep2"]
rep2_dist1 = np.asarray(rep2[rep2["ctrl_status_fixed"] == "scramble"]["value"])
rep2_dist2 = np.asarray(rep2[rep2["ctrl_status_fixed"] == "control"]["value"])

rep2_dist1 = rep2_dist1[~np.isnan(rep2_dist1)]
rep2_dist2 = rep2_dist2[~np.isnan(rep2_dist2)]

u, pval = stats.mannwhitneyu(rep2_dist1, rep2_dist2, alternative="less", use_continuity=False)
print(pval)
annotate_pval(ax, 0.9, 1.1, 5, 0, 4.8, pval, fontsize)
        
        
plt.legend(loc=2, bbox_to_anchor=(1.05, 1.05))
plt.show()
fig.savefig("Fig3D.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# ## 8. check correlations of scores across reps

# In[35]:


print(len(data[(data["Day_Zero__rep1"] < cpm_thresh) & (data["Day_Zero__rep2"] < cpm_thresh)]))
print(len(data[(data["Day_Zero__rep1"] < cpm_thresh) & (data["Day_Zero__rep2"] >= cpm_thresh)]))
print(len(data[(data["Day_Zero__rep1"] >= cpm_thresh) & (data["Day_Zero__rep2"] < cpm_thresh)]))
print(len(data[(data["Day_Zero__rep1"] >= cpm_thresh) & (data["Day_Zero__rep2"] >= cpm_thresh)]))


# In[36]:


print(len(data[(data["BFP+_Endo--__rep1"] < 1) & (data["BFP+_Endo--__rep2"] < 1)]))
print(len(data[(data["BFP+_Endo--__rep1"] < 1) & (data["BFP+_Endo--__rep2"] >= 1)]))
print(len(data[(data["BFP+_Endo--__rep1"] >= 1) & (data["BFP+_Endo--__rep2"] < 1)]))
print(len(data[(data["BFP+_Endo--__rep1"] >= 1) & (data["BFP+_Endo--__rep2"] >= 1)]))


# In[37]:


fig = plt.figure(figsize=(2, 2))

no_nan = data[(~pd.isnull(data["BFP+_Endo--__rep1"])) & (~pd.isnull(data["BFP+_Endo--__rep2"]))]
no_nan["BFP+_Endo--__rep1_log"] = np.log10(no_nan["BFP+_Endo--__rep1"]+1)
no_nan["BFP+_Endo--__rep2_log"] = np.log10(no_nan["BFP+_Endo--__rep2"]+1)
g = sns.jointplot(data=no_nan, x="BFP+_Endo--__rep1_log", y="BFP+_Endo--__rep2_log", color="slategray", 
                   joint_kws={"rasterized": False, "s": 10, "alpha": 0.5, "linewidths": 0.5, 
                              "edgecolors": "white"},
                   marginal_kws={"bins": 15, "hist": False, "kde": True}, size=2, space=0.5)

g.set_axis_labels("log10(undifferentiated cpm + 1)\nreplicate 1", "log10(undifferentiated cpm + 1)\nreplicate 2")
g.ax_joint.axhline(y=np.log10(2), color="black", linestyle="dashed")
g.ax_joint.axvline(x=np.log10(2), color="black", linestyle="dashed")
g.ax_joint.set_xlim((-0.2, 3))
g.ax_joint.set_ylim((-0.2, 3))

g.ax_joint.set_xticks([0, 1, 2, 3])
g.ax_joint.set_yticks([0, 1, 2, 3])

plt.show()
g.savefig("Fig3C.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# In[38]:


# stringently filter data based on endo counts and day zero counts
data_stringent_filt = data_filt[(data_filt["Day_Zero__rep1"] >= cpm_thresh) & (data_filt["Day_Zero__rep2"] >= cpm_thresh)]
print(len(data_stringent_filt))
data_stringent_filt = data_stringent_filt[(data_stringent_filt["BFP+_Endo--__rep1"] >= 1) & 
                                          (data_stringent_filt["BFP+_Endo--__rep2"] >= 1)]
len(data_stringent_filt)


# In[39]:


# for plotting purposes, sample from those with low enrichment rates
high_enrich = data_stringent_filt[(data_stringent_filt["BFP+_log_enrichment__rep1"] >= 1) & 
                                  (data_stringent_filt["BFP+_log_enrichment__rep2"] >= 1)]
to_sample = data_stringent_filt[~data_stringent_filt["sgRNA"].isin(high_enrich["sgRNA"])]

exp_sample = to_sample[to_sample["ctrl_status_fixed"] == "experimental"].sample(4000)
scr_sample = to_sample[to_sample["ctrl_status_fixed"] == "scramble"]
ctr_sample = to_sample[to_sample["ctrl_status_fixed"] == "control"]

for_plot = high_enrich.append(exp_sample).append(scr_sample).append(ctr_sample)
len(for_plot)


# In[40]:


fig, ax = plt.subplots(figsize=(2, 2), nrows=1, ncols=1)

exp = for_plot[for_plot["ctrl_status_fixed"] == "experimental"]
scr = for_plot[for_plot["ctrl_status_fixed"] == "scramble"]
ctrl = for_plot[for_plot["ctrl_status_fixed"] == "control"]

ax.scatter(exp["BFP+_enrichment__rep1"], exp["BFP+_enrichment__rep2"], color=pal["experimental"], s=5, alpha=0.7)
ax.scatter(scr["BFP+_enrichment__rep1"], scr["BFP+_enrichment__rep2"], color=pal["scramble"], s=5, alpha=0.7)
ax.scatter(ctrl["BFP+_enrichment__rep1"], ctrl["BFP+_enrichment__rep2"], color=pal["control"], alpha=1, s=5)

ax.set_xscale('symlog', linscale=20)
ax.set_yscale('symlog', linscale=20)

ax.plot([-0.18, 10000], [-0.18, 10000], "--", color="gray", linewidth=1)
ax.plot([2, 2], [2, 10000], '--', color="black", linewidth=1)
ax.plot([2, 10000], [2, 2], '--', color="black", linewidth=1)

ax.set_xlim((-0.18, 10000))
ax.set_ylim((-0.18, 10000))

# annotate corr
no_nan = data_stringent_filt[(~pd.isnull(data_stringent_filt["BFP+_enrichment__rep1"])) & 
                             (~pd.isnull(data_stringent_filt["BFP+_enrichment__rep2"]))]
r, p = pearsonr(no_nan["BFP+_enrichment__rep1"], no_nan["BFP+_enrichment__rep2"])
print(r)
# ax.text(0.95, 0.2, "r = {:.2f}".format(r), ha="right", va="top", fontsize=fontsize, fontweight="bold",
#         transform=ax.transAxes)

# filter data to those that are reproducibly enriched in both reps
filt = data_stringent_filt[(data_stringent_filt["BFP+_enrichment__rep1"] >= 2) & 
                           (data_stringent_filt["BFP+_enrichment__rep2"] >= 2)]

ax.text(0.97, 0.17, "filtered sgRNAs\n(n = %s)" % (len(no_nan)), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.3, 0.97, "reproducibly enriched\nsgRNAs\n(n = %s)" % (len(filt)), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)

plt.xlabel("sgRNA score rep 1")
plt.ylabel("sgRNA score rep 2")
plt.show()
fig.savefig("Fig3E.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# In[41]:


fig, ax = plt.subplots(figsize=(2, 2), nrows=1, ncols=1)

exp = for_plot[for_plot["ctrl_status_fixed"] == "experimental"]
scr = for_plot[for_plot["ctrl_status_fixed"] == "scramble"]
ctrl = for_plot[for_plot["ctrl_status_fixed"] == "control"]

ax.scatter(exp["BFP+_enrichment__rep1"], exp["BFP+_enrichment__rep2"], color=pal["experimental"], s=5, alpha=0.7)
ax.scatter(scr["BFP+_enrichment__rep1"], scr["BFP+_enrichment__rep2"], color=pal["scramble"], s=5, alpha=0.7)
ax.scatter(ctrl["BFP+_enrichment__rep1"], ctrl["BFP+_enrichment__rep2"], color=pal["control"], alpha=1, s=5)

ax.set_xscale('symlog', linscale=20)
ax.set_yscale('symlog', linscale=20)

ax.plot([-0.18, 10000], [-0.18, 10000], "--", color="gray", linewidth=1)

ax.set_xlim((-0.18, 10000))
ax.set_ylim((-0.18, 10000))

# annotate corr
no_nan = data_stringent_filt[(~pd.isnull(data_stringent_filt["BFP+_enrichment__rep1"])) & 
                             (~pd.isnull(data_stringent_filt["BFP+_enrichment__rep2"]))]
r, p = pearsonr(no_nan["BFP+_enrichment__rep1"], no_nan["BFP+_enrichment__rep2"])
print(r)
# ax.text(0.95, 0.2, "r = {:.2f}".format(r), ha="right", va="top", fontsize=fontsize, fontweight="bold",
#         transform=ax.transAxes)

# filter data to those that are reproducibly enriched in both reps
filt = data_stringent_filt[(data_stringent_filt["BFP+_enrichment__rep1"] >= 2) & 
                           (data_stringent_filt["BFP+_enrichment__rep2"] >= 2)]

ax.text(0.97, 0.17, "filtered sgRNAs\n(n = %s)" % (len(no_nan)), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.3, 0.97, "reproducibly enriched\nsgRNAs\n(n = %s)" % (len(filt)), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)

plt.xlabel("sgRNA score rep 1")
plt.ylabel("sgRNA score rep 2")
plt.show()
fig.savefig("Fig3E_JeffTalk.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# In[42]:


tmp = filt[["BFP+_enrichment__rep1", "BFP+_enrichment__rep2"]]
len(tmp)


# In[43]:


tmp["log1"] = np.log2(tmp["BFP+_enrichment__rep1"])
tmp["log2"] = np.log2(tmp["BFP+_enrichment__rep2"])
tmp[["log1", "log2"]].corr(method="pearson")


# In[44]:


rep_scr = filt[filt["ctrl_status_fixed"] == "scramble"]
len(rep_scr["group_id"].unique())


# In[45]:


len(rep_scr)/len(filt)


# ## 9. annotate stringently-filtered sgRNAs

# In[46]:


filt.head()


# In[47]:


all_enrich_grp["stringent_filt"] = all_enrich_grp["group_id"].isin(filt["group_id"])
len(all_enrich_grp[all_enrich_grp["stringent_filt"]])


# ## 10. write file

# In[48]:


f = "../../../data/02__screen/02__enrichment_data/enrichment_values.tmp"
print(len(all_enrich_grp))
all_enrich_grp.to_csv(f, sep="\t", index=False)


# In[49]:


f = "../../../data/02__screen/02__enrichment_data/filtered_sgRNAs.tmp"
print(len(filt))
filt.to_csv(f, sep="\t", index=False)


# In[50]:


f = "../../../data/02__screen/02__enrichment_data/data_stringent_filt.tmp"
print(len(data_stringent_filt))
data_stringent_filt.to_csv(f, sep="\t", index=False)

