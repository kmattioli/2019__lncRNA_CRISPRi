
# coding: utf-8

# # 04__results
# 
# in this notebook, i aggregate results from CRISPhieRmix and plot.
# 
# figures in this notebook:
# - Fig 3F: plot showing transcript-level enrichments vs. CRISPhieRmix FDR
# 
# supplemental tables in this notebook:
# - Table S3: enrichment scores & FDRs for all targeted TSSs

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

# ## variables

# In[3]:


data_filt_f = "../../../data/02__screen/02__enrichment_data/data_filt.with_batch.tmp"


# In[4]:


crisphie_diff_f = "../../../data/02__screen/02__enrichment_data/CRISPhieRmix_diff.with_batch.txt"


# In[5]:


crisphie_dz_f = "../../../data/02__screen/02__enrichment_data/CRISPhieRmix_dz.with_batch.txt"


# In[6]:


## liu results
liu_res_f = "../../../misc/11__liu_data/iPSC_results.txt"
liu_meta_f = "../../../misc/11__liu_data/tss_metadata.txt"


# ## 1. import data

# In[7]:


data_filt = pd.read_table(data_filt_f, sep="\t")
data_filt.head()


# In[8]:


crisphie_diff = pd.read_table(crisphie_diff_f, sep="\t")
len(crisphie_diff)


# In[9]:


crisphie_dz = pd.read_table(crisphie_dz_f, sep="\t")
len(crisphie_dz)


# ## 2. parse results from CRISPhieRmix

# In[10]:


# if using gene level to group
# crisphie["gene_name"] = crisphie["groups"].str.split("__", expand=True)[0]
# crisphie["ctrl_status_fixed"] = crisphie["groups"].str.split("__", expand=True)[1]

# if using transcript level
crisphie_diff["transcript_name"] = crisphie_diff["groups"].str.split(";;;", expand=True)[0]
crisphie_diff["group_id"] = crisphie_diff["groups"].str.split(";;;", expand=True)[1]
crisphie_diff["ctrl_status"] = crisphie_diff["groups"].str.split(";;;", expand=True)[2]
crisphie_diff["transcript_biotype_status"] = crisphie_diff["groups"].str.split(";;;", expand=True)[3]
crisphie_diff["n_sgRNA"] = crisphie_diff["groups"].str.split(";;;", expand=True)[4].astype(int)
crisphie_diff.head()


# In[11]:


# if using transcript level
crisphie_dz["transcript_name"] = crisphie_dz["groups"].str.split(";;;", expand=True)[0]
crisphie_dz["group_id"] = crisphie_dz["groups"].str.split(";;;", expand=True)[1]
crisphie_dz["ctrl_status"] = crisphie_dz["groups"].str.split(";;;", expand=True)[2]
crisphie_dz["transcript_biotype_status"] = crisphie_dz["groups"].str.split(";;;", expand=True)[3]
crisphie_dz["n_sgRNA"] = crisphie_dz["groups"].str.split(";;;", expand=True)[4].astype(int)
crisphie_dz.head()


# ## 3. calculate effect size -- estimate as top 3 sgRNA median

# In[12]:


# need to remove ' from data_filt to merge with the crisphiermix table
# looks like R removes these by default
data_filt["gene_name"] = data_filt["gene_name"].str.replace("'", '')
data_filt["transcript_name"] = data_filt["transcript_name"].str.replace("'", '')
data_filt[data_filt["gene_name"].str.contains("AC007128.1")][["gene_name", "l2fc_diff"]]


# In[13]:


data_filt = data_filt.sort_values(by=["group_id", "ctrl_status", "l2fc_diff"], ascending=False)
eff_size = data_filt.groupby(["group_id", "ctrl_status"]).head(3)
eff_size = eff_size.groupby(["group_id", "ctrl_status"])["l2fc_diff"].agg("median").reset_index()
eff_size.head()


# ## 4. merge effect size w/ CRISPhieRmix FDR

# In[14]:


crisphie_diff = crisphie_diff.merge(eff_size, on=["group_id", "ctrl_status"], how="left")
print(len(crisphie_diff))
crisphie_diff.sort_values(by="FDR").head()


# In[15]:


gene_map = data_filt[["group_id", "gene_name"]].drop_duplicates()
crisphie_dz = crisphie_dz.merge(gene_map, on="group_id", how="left")
len(crisphie_dz)


# ## 5. plot results

# In[16]:


crisphie_diff["neg_log_FDR"] = -np.log10(crisphie_diff["FDR"]+1e-12)
sig = crisphie_diff[crisphie_diff["FDR"] < 0.1]
not_sig = crisphie_diff[crisphie_diff["FDR"] >= 0.1]
ctrl = crisphie_diff[crisphie_diff["ctrl_status"] == "control"]
exp = crisphie_diff[crisphie_diff["ctrl_status"] == "experimental"]
mrna = crisphie_diff[crisphie_diff["ctrl_status"] == "mRNA"]
scr = crisphie_diff[crisphie_diff["ctrl_status"] == "scramble"]


# In[17]:


sig.ctrl_status.value_counts()


# In[18]:


sig.transcript_biotype_status.value_counts()


# In[19]:


sig[sig["transcript_biotype_status"] == "promoter_overlap"][["transcript_name", "FDR"]].sort_values(by="transcript_name", ascending=True)


# In[20]:


uniq_txs = list(exp[exp["FDR"] < 0.1]["transcript_name"].unique())
print(len(uniq_txs))

genes = []
for tx in uniq_txs:
    if tx.startswith("["):
        gene = tx.split(",")[0]
        gene = gene[1:-4]
        genes.append(gene)
    else:
        gene = tx[0:-4]
        genes.append(gene)

uniq_loci = set(genes)
print(len(uniq_loci))


# In[21]:


nopromoverlap = sig[(sig["transcript_biotype_status"] != "promoter_overlap") & (sig["ctrl_status"] == "experimental")]
len(nopromoverlap)


# In[22]:


nopromoverlap.sort_values(by="FDR")[["transcript_name", "transcript_biotype_status"]]


# In[23]:


promoverlap = sig[(sig["transcript_biotype_status"] == "promoter_overlap") & (sig["ctrl_status"] == "experimental")]
promoverlap[["transcript_name", "transcript_biotype_status", "FDR", "l2fc_diff"]]


# In[24]:


pal = {"control": sns.color_palette()[2], "experimental": "black", "scramble": "gray"}
sns.palplot(pal.values())


# In[25]:


# list of promoter overlap hits where no mrnas whose tss is within 1000 bp of the lncrna tsss are expr (>=1 tpm)
# in either hESCs or endo RNA-seq samples
# (found in the "hit_expression" downstream notebook; copy-pasting here to highlight in plot)
promoverlap_noexp_ids = ['RP11-326C3.12-001', 'RP11-402J6.1-002', 'HOXC-AS1-002', 'AC011523.2-001', 'RP4-680D5.8-001']
promoverlap_noexp = promoverlap[promoverlap["transcript_name"].isin(promoverlap_noexp_ids)]
len(promoverlap_noexp)


# In[26]:


fig = plt.figure(figsize=(2,2))

ax = plt.gca()

ax.scatter(exp["l2fc_diff"], exp["neg_log_FDR"], color="slategray", s=15, alpha=0.5)
ax.scatter(nopromoverlap["l2fc_diff"], nopromoverlap["neg_log_FDR"], color="black", edgecolors="black",
           linewidths=0.5, s=15, alpha=1)
ax.scatter(mrna["l2fc_diff"], mrna["neg_log_FDR"], color="white", s=15, alpha=1, edgecolors="black", linewidths=0.5)
ax.scatter(ctrl["l2fc_diff"], ctrl["neg_log_FDR"], color=pal["control"], s=20, alpha=1, edgecolors="black",
           linewidths=0.5)
ax.scatter(scr["l2fc_diff"], scr["neg_log_FDR"], color="gray", s=15, alpha=0.5)
ax.axhline(y=1, linestyle="dashed", color="black")
ax.set_xlabel("transcript enrichment score")
ax.set_ylabel("-log10(CRISPhieRmix FDR)")
ax.set_xscale('symlog')

# annotate #s
n_sig = len(sig)
n_not_sig = len(not_sig)
ax.text(0.05, 0.85, "FDR < 0.1\n(n=%s)" % (n_sig), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)

fig.savefig("Fig3F.pdf", dpi="figure", bbox_inches="tight")


# In[27]:


ctrl[ctrl["FDR"] < 0.1]


# ## 6. check enrichment of liu (CRiNCL) lncRNAs in our day zero drop-out results

# In[28]:


liu_res = pd.read_table(liu_res_f, sep="\t", skiprows=3, names=["tss_id", "score_t12_1", "score_t12_2", "score_t12_av",
                                                                "hit_t12", "score_t18_1", "score_t18_2", 
                                                                "score_t18_av", "hit_t18"])
liu_res.head()


# In[29]:


liu_res.min(axis=0)


# In[30]:


liu_hits = list(liu_res[(~pd.isnull(liu_res["hit_t12"])) | (~pd.isnull(liu_res["hit_t18"]))]["tss_id"])
len(liu_hits)


# In[31]:


liu_tss = pd.read_table(liu_meta_f, sep="\t")
liu_tss.head()


# In[32]:


## subset results to those lncRNAs that are in the liu library
crisphie_liu = crisphie_dz[crisphie_dz["gene_name"].isin(liu_tss["gene name"])]
print(len(crisphie_liu))
print(len(crisphie_liu.gene_name.unique()))


# In[33]:


liu_hits_tss = liu_tss[liu_tss["Gene ID"].isin(liu_hits)]
liu_hits_tss.head()


# In[34]:


liu_hits_annot = liu_hits_tss[liu_hits_tss["gene name"] != "-"]
print(len(liu_hits_annot))
liu_hits_annot.sample(5)


# In[35]:


sig_dz = crisphie_dz[crisphie_dz["FDR"] < 0.1]
len(sig_dz)


# In[36]:


n_mHit_lHit = len(crisphie_liu[(crisphie_liu["gene_name"].isin(sig_dz["gene_name"])) & 
                               (crisphie_liu["gene_name"].isin(liu_hits_tss["gene name"]))]["gene_name"].unique())
n_mHit_lNoHit = len(crisphie_liu[(crisphie_liu["gene_name"].isin(sig_dz["gene_name"])) & 
                                 (~crisphie_liu["gene_name"].isin(liu_hits_tss["gene name"]))]["gene_name"].unique())
n_mNoHit_lHit = len(crisphie_liu[(~crisphie_liu["gene_name"].isin(sig_dz["gene_name"])) & 
                                 (crisphie_liu["gene_name"].isin(liu_hits_tss["gene name"]))]["gene_name"].unique())
n_mNoHit_lNoHit = len(crisphie_liu[(~crisphie_liu["gene_name"].isin(sig_dz["gene_name"])) & 
                                   (~crisphie_liu["gene_name"].isin(liu_hits_tss["gene name"]))]["gene_name"].unique())

print("# genes that are hits in our screen and hits in CRiNCL: %s" % n_mHit_lHit)
print("# genes that are hits in our screen and not hits in CRiNCL: %s" % n_mHit_lNoHit)
print("# genes that are not hits in our screen and hits in CRiNCL: %s" % n_mNoHit_lHit)
print("# genes that are not hits in our screen and not hits in CRiNCL: %s" % n_mNoHit_lNoHit)


# In[37]:


stats.fisher_exact([[n_mHit_lHit, n_mHit_lNoHit], [n_mNoHit_lHit, n_mNoHit_lNoHit]])


# In[38]:


plot_dict = {"perc_crincl_hits_in_our_hits": (n_mHit_lHit/(n_mHit_lHit+n_mHit_lNoHit))*100,
             "perc_crincl_hits_in_our_nohits": (n_mNoHit_lHit/(n_mNoHit_lHit+n_mNoHit_lNoHit))*100,
             "perc_our_hits_in_crincl_hits": (n_mHit_lHit/(n_mHit_lHit+n_mNoHit_lHit))*100,
             "perc_our_hits_in_crincl_nohits": (n_mHit_lNoHit/(n_mHit_lNoHit+n_mNoHit_lNoHit))*100}
plot_df = pd.DataFrame.from_dict(plot_dict, orient="index").reset_index()
plot_df


# ## 9. write final file(s)

# In[39]:


def is_hit(row):
    if row["FDR"] < 0.1:
        return "hit"
    else:
        return "no hit"
    
crisphie_diff["hit_status"] = crisphie_diff.apply(is_hit, axis=1)


# In[40]:


supp = crisphie_diff[["group_id", "transcript_name", "ctrl_status", "transcript_biotype_status", "FDR", "l2fc_diff", 
                 "hit_status", "n_sgRNA"]]
supp.columns = ["group_id", "transcript_name", "ctrl_status", "transcript_biotype_status", "CRISPhieRmix_FDR", 
                "effect_size", "hit_status", "n_sgRNA"]
supp = supp.sort_values(by="CRISPhieRmix_FDR")
print(len(supp))
supp.head()


# In[41]:


supp.hit_status.value_counts()


# In[42]:


supp.sort_values(by="CRISPhieRmix_FDR", ascending=True).head(10)


# In[43]:


supp[supp["transcript_name"].str.contains("RP11-541P9.3")]


# In[44]:


f = "../../../data/02__screen/02__enrichment_data/SuppTable_S3.CRISPhieRmix_results.txt"
supp.to_csv(f, sep="\t", index=False)

