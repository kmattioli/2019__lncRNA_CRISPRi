
# coding: utf-8

# # 02__calculate_pvals
# 
# in this notebook, i calculate p values for each transcript-level enrichment score. i do this by comparing the score for a lncRNA to a randomly sampled set of 2000 negative control "transcripts" (i.e., grouped scrambled sgRNAs). i do it in each replicate and then combine the p values using stouffer's method, and adjust the p values using the benjamini-hochberg procedure.
# 
# figures in this notebook:
# - Fig 3F: plot showing transcript-level enrichments vs. adjusted p values for our reproducible hits
# 
# supplemental tables in this notebook:
# - Supp Table S2: enrichment scores & pvalues for CRISPRi screen

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


# ## functions

# In[3]:


def get_null_scores(neg_ctrls, thresh, n_bootstrap, min_sgRNAs_represented, top_sgRNAs_to_rank, n_sgRNAs_for_std, rank_col, enrich_col, score_col):
    n_nulls = 0
    all_nulls = []
    while n_nulls <= n_bootstrap:
        samp = neg_ctrls.sample(10)
        samp["group_id"] = "arbitrary_id"
        i_results = get_gene_enrichments(samp, thresh, min_sgRNAs_represented, top_sgRNAs_to_rank, 
                                         n_sgRNAs_for_std, rank_col, enrich_col)
        if i_results is None:
            continue
        elif len(i_results) > 0:
            null_score = i_results[score_col].iloc[0]
            all_nulls.append(null_score)
            n_nulls += 1
    return all_nulls


# In[4]:


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


# In[5]:


def get_pvalue(actual_score, null_scores):
    if actual_score == np.nan:
        pval = np.nan
    else:
        n_larger_than_actual = len([x for x in null_scores if x >= actual_score])
        pval = (n_larger_than_actual+1)/(len(null_scores)+1)
    return pval


# In[6]:


# combine p values
def combine_pvals(row, reps):
    pvals = np.asarray(list(row[reps]))
    non_na_pvals = np.asarray([float(x) for x in pvals if not "NA" in str(x)])
    non_na_pvals = non_na_pvals[~np.isnan(non_na_pvals)]
    if len(non_na_pvals) > 1:
        new_pval = stats.combine_pvalues(non_na_pvals, method="stouffer")[1]
    else:
        new_pval = np.nan
    return new_pval


# In[7]:


def classify_sgRNA(row):
    if row["Day_Zero__rep1"] >= cpm_thresh and row["Day_Zero__rep2"] >= cpm_thresh:
        if row["BFP+_enrichment__rep1"] >= 2 and row["BFP+_enrichment__rep2"] >= 2:
            return "day zero & enrich filt met"
        else:
            return "day zero filt met"
    else:
        if row["BFP+_enrichment__rep1"] >= 2 and row["BFP+_enrichment__rep2"] >= 2:
            return "enrich filt met"
        else:
            return "no filt met"


# ## variables

# In[8]:


filt_f = "../../../data/02__screen/02__enrichment_data/filtered_sgRNAs.tmp"


# In[9]:


all_enrich_grp_f = "../../../data/02__screen/02__enrichment_data/enrichment_values.tmp"


# In[10]:


data_stringent_filt_f = "../../../data/02__screen/02__enrichment_data/data_stringent_filt.tmp"


# In[11]:


cpm_thresh = 5


# In[12]:


min_sgRNAs_represented = 3
top_sgRNAs_to_rank = 3
n_sgRNAs_for_std = 3


# ## 1. import data

# In[13]:


all_enrich_grp = pd.read_table(all_enrich_grp_f, sep="\t")
all_enrich_grp.head()


# In[14]:


filt = pd.read_table(filt_f, sep="\t")
filt.head()


# In[15]:


data_stringent_filt = pd.read_table(data_stringent_filt_f, sep="\t")
len(data_stringent_filt)


# ## 2. filter group ids to those with stringently filtered sgRNA

# In[16]:


filt_enrich_grp = all_enrich_grp[all_enrich_grp["stringent_filt"]].drop_duplicates()
len(filt_enrich_grp)


# In[17]:


neg_ctrls = data_stringent_filt[data_stringent_filt["ctrl_status_fixed"] == "scramble"]
len(neg_ctrls)


# ## 3. get p-value per group id

# In[18]:


np.random.seed(1234)


# In[19]:


n_bootstraps = 2000

print("n scrambles: %s" % len(neg_ctrls))

# rep1
rep1_group_scores = filt_enrich_grp[filt_enrich_grp["ctrl_status"] != "scramble"][["group_id", 
                                                                                   "BFP+_score__rep1"]].set_index("group_id")
rep1_group_scores_dict = rep1_group_scores.to_dict(orient="index")
print("n TSSs: %s" % (len(rep1_group_scores_dict)))

print("calculating %s null scores...%s" % (n_bootstraps, time.ctime()))
rep1_null_scores = get_null_scores(neg_ctrls, cpm_thresh, n_bootstraps, min_sgRNAs_represented, top_sgRNAs_to_rank,
                                   n_sgRNAs_for_std, "BFP+_rank__rep1", "BFP+_enrichment__rep1", "BFP+_score__rep1")

rep1_results = {}
for i, k in enumerate(rep1_group_scores_dict):
    if i % 50 == 0:
        print("...TSS #%s... %s" % (i, time.ctime()))
    actual_score = rep1_group_scores_dict[k]["BFP+_score__rep1"]
    pval = get_pvalue(actual_score, rep1_null_scores)
    rep1_results[k] = pval


# In[20]:


# rep2
rep2_group_scores = filt_enrich_grp[filt_enrich_grp["ctrl_status"] != "scramble"][["group_id", 
                                                                                   "BFP+_score__rep2"]].set_index("group_id")
rep2_group_scores_dict = rep2_group_scores.to_dict(orient="index")
print("n TSSs: %s" % (len(rep2_group_scores_dict)))

print("calculating %s null scores...%s" % (n_bootstraps, time.ctime()))
rep2_null_scores = get_null_scores(neg_ctrls, cpm_thresh, n_bootstraps, min_sgRNAs_represented, top_sgRNAs_to_rank,
                                   n_sgRNAs_for_std, "BFP+_rank__rep2", "BFP+_enrichment__rep2", "BFP+_score__rep2")

rep2_results = {}
for i, k in enumerate(rep2_group_scores_dict):
    if i % 50 == 0:
        print("...TSS #%s... %s" % (i, time.ctime()))
    actual_score = rep2_group_scores_dict[k]["BFP+_score__rep2"]
    pval = get_pvalue(actual_score, rep2_null_scores)
    rep2_results[k] = pval


# ## 4. merge & plot p-values

# In[21]:


rep1_pvals = pd.DataFrame.from_dict(rep1_results, orient="index").reset_index()
rep1_pvals.columns = ["group_id", "pval__rep1"]

rep2_pvals = pd.DataFrame.from_dict(rep2_results, orient="index").reset_index()
rep2_pvals.columns = ["group_id", "pval__rep2"]

pvals = rep1_pvals.merge(rep2_pvals, on="group_id")
pvals.sample(5)


# In[22]:


len(pvals)


# In[23]:


len(pvals[(pd.isnull(pvals["pval__rep1"])) | (pd.isnull(pvals["pval__rep2"]))])


# In[24]:


# p value histograms
fig = plt.figure(figsize=(2, 1))
sns.distplot(pvals["pval__rep1"], kde=False)
plt.xlabel("rep1 pvalues")
plt.ylabel("frequency")
#fig.savefig("pval_hist_rep1.pdf", dpi="figure", bbox_inches="tight")


# In[25]:


# p value histograms
fig = plt.figure(figsize=(2, 1))
sns.distplot(pvals["pval__rep2"], kde=False)
plt.xlabel("rep2 pvalues")
plt.ylabel("frequency")
#fig.savefig("pval_hist_rep2.pdf", dpi="figure", bbox_inches="tight")


# ## 5. combine & plot p-values

# In[26]:


pvals["combined_pval"] = pvals.apply(combine_pvals, axis=1, reps=["pval__rep1", "pval__rep2"])
pvals.sort_values(by="combined_pval").head(10)


# In[27]:


# p value histograms
fig = plt.figure(figsize=(2, 1))
sns.distplot(pvals["combined_pval"], kde=False)
plt.xlabel("combined pvalues")
plt.ylabel("frequency")
#fig.savefig("pval_hist_combined.pdf", dpi="figure", bbox_inches="tight")


# ## 6. adjust p-values

# In[28]:


# adjust p-values
pvals_nonan = pvals[~pd.isnull(pvals["combined_pval"])]
pvals_nonan["padj"] = multicomp.multipletests(pvals_nonan["combined_pval"], method="fdr_bh")[1]
pvals = pvals.merge(pvals_nonan[["group_id", "padj"]], on="group_id", how="left")
pvals.sort_values(by="padj").head()


# In[29]:


grp_data = filt_enrich_grp.merge(pvals, on="group_id", how="left")
#grp_data = filt_enrich_grp.copy()
grp_data.sort_values(by="BFP+_score_rank__mean").head()


# In[30]:


grp_data[grp_data["padj"] < 0.05].ctrl_status_fixed.value_counts()


# In[31]:


grp_data[grp_data["padj"] < 0.1].ctrl_status_fixed.value_counts()


# ## 7. plot hit rankings

# In[32]:


grp_data["neg_log_padj"] = -np.log10(grp_data["padj"])
sig = grp_data[grp_data["padj"] < 0.1]
not_sig = grp_data[grp_data["padj"] >= 0.1]
ctrl = grp_data[grp_data["ctrl_status_fixed"] == "control"]
exp = grp_data[grp_data["ctrl_status_fixed"] == "experimental"]
scr = grp_data[grp_data["ctrl_status_fixed"] == "scramble"]


# In[33]:


pal = {"control": sns.color_palette()[2], "experimental": "black", "scramble": "gray"}
sns.palplot(pal.values())


# In[34]:


fig, ax = plt.subplots(figsize=(2,2), nrows=1, ncols=1)

ax.scatter(exp["BFP+_score__mean"], exp["neg_log_padj"], color="slategray", s=15, alpha=0.5)
ax.scatter(ctrl["BFP+_score__mean"], ctrl["neg_log_padj"], color=pal["control"], s=20, alpha=1, edgecolors="black",
           linewidths=0.5)
ax.scatter(scr["BFP+_score__mean"], scr["neg_log_padj"], color="gray", s=15, alpha=0.5)
ax.axhline(y=1, linestyle="dashed", color="black")
ax.set_xlabel("transcript enrichment score")
ax.set_ylabel("-log10(adjusted p-value)")

# annotate #s
n_sig = len(sig)
n_not_sig = len(not_sig)
ax.text(0.05, 0.95, "FDR < 0.1\n(n=%s)" % (n_sig), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.95, 0.27, "FDR > 0.1\n(n=%s)" % (n_not_sig), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.set_xscale('symlog')
fig.savefig("Fig3E.pdf", dpi="figure", bbox_inches="tight")


# ## 8. merge data w/ p-vals

# In[35]:


all_grp_data = all_enrich_grp.merge(grp_data[["group_id", "pval__rep1", "pval__rep2", "combined_pval",
                                              "padj", "neg_log_padj"]], on="group_id", how="left")
len(all_grp_data)


# ## 9. write final file

# In[36]:


f = "../../../data/02__screen/02__enrichment_data/enrichment_values.txt"
all_grp_data.to_csv(f, sep="\t", index=False)


# In[37]:


all_grp_data[all_grp_data["gene_name"] == "DIGIT"]


# ## 10. write supplemental file

# In[38]:


endo_ctrls_validated = pd.read_table("../../../misc/04__pos_ctrls/endo_ctrls_validated.updated.txt", sep="\t", header=None)
endo_ctrls_validated.columns = ["gene_name"]


# In[39]:


index_f = "../../../data/01__design/02__final_sgRNAs/crispri_with_primers.txt"
index = pd.read_table(index_f, sep="\t")


# In[40]:


index["endo_ctrl_val"] = index["gene_name"].isin(endo_ctrls_validated["gene_name"])


# In[41]:


def fix_ctrl_status_w_DE_mRNAs(row):
    if row.endo_ctrl_val:
        return "control mRNA"
    else:
        if row.ctrl_status == "scramble":
            return "scramble"
        elif row.ctrl_status == "control":
            return "diff. expr. mRNA"
        else:
            return "lncRNA"
        
index["ctrl_status_fixed"] = index.apply(fix_ctrl_status_w_DE_mRNAs, axis=1)


# In[42]:


index_dedupe = index[["tss_id_hg38", "transcript_id", "gene_id", "ctrl_status_fixed", "gene_name"]].drop_duplicates()
len(index_dedupe)


# In[43]:


index_dedupe = index_dedupe[index_dedupe["tss_id_hg38"] != "scramble"]
len(index_dedupe.tss_id_hg38.unique())


# In[44]:


supp = index_dedupe.merge(all_grp_data, left_on="tss_id_hg38", right_on="group_id", how="left")


# In[45]:


len(supp.tss_id_hg38.unique())


# In[46]:


supp[["tss_id_hg38", "ctrl_status_fixed_x"]].drop_duplicates().ctrl_status_fixed_x.value_counts()


# In[47]:


supp = supp[["tss_id_hg38", "gene_name_x", "gene_id", "transcript_id", "ctrl_status_fixed_x", "stringent_filt",
             "BFP+_score__rep1", "BFP+_score__rep2", "BFP+_score__mean", "padj"]]


# In[48]:


supp.columns = ["tss_id_hg38", "gene_name", "gene_id", "transcript_id", "ctrl_status", "meets_filter", "rep1_score",
                "rep2_score", "mean_score", "padj"]
supp.head()


# In[49]:


supp["refseq"] = supp["tss_id_hg38"].str.split(":", expand=True)[0]
supp["tss_start_hg38"] = supp["tss_id_hg38"].str.split(":", expand=True)[2]
supp["tss_strand_hg38"] = supp["tss_id_hg38"].str.split(":", expand=True)[1]
supp.head()


# In[50]:


f = "../../../misc/05__refseq/chr_to_refseq.txt"
refseq = pd.read_table(f, sep="\t", header=None)
refseq.columns = ["tss_chr_hg38", "refseq"]


# In[51]:


supp = supp.merge(refseq, on="refseq")
supp.head()


# In[52]:


supp = supp[["gene_name", "gene_id", "transcript_id", "ctrl_status", "tss_chr_hg38", "tss_start_hg38", 
             "tss_strand_hg38", "meets_filter", "rep1_score", "rep2_score", "mean_score", "padj"]]
supp = supp.sort_values(by="mean_score", ascending=False)
supp.fillna("filter not met", inplace=True)
supp.head()


# In[53]:


print(len(supp))


# In[54]:


f = "../../../data/02__screen/02__enrichment_data/SuppTable.screen_results.txt"
supp.to_csv(f, sep="\t", index=False)

