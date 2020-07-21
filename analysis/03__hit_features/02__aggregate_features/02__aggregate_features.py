#!/usr/bin/env python
# coding: utf-8

# # 02__aggregate_features
# 
# in this notebook, i aggregate all features examined (in order to make plots for Fig 5 and for the cluster analysis in Fig 6)
# 
# tables in this notebook:
# - Table S5: features for all lncRNAs in the screen

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


# features
feature_dir = "../../../misc/09__model_features"
splicing_f = "%s/gene_splicing_efficiency.with_DIGIT.txt" % feature_dir
gc_f = "%s/gc_content.with_DIGIT.txt" % feature_dir
n_tss_f = "%s/n_tss_within_100bp.with_DIGIT.txt" % feature_dir
n_enh_f = "%s/n_enhancers_within_1mb.with_DIGIT.txt" % feature_dir
closest_enh_tss_f = "%s/closest_enh_to_TSS.with_DIGIT.fixed.txt" % feature_dir
closest_enh_tran_f = "%s/closest_enh_to_transcript.with_DIGIT.txt" % feature_dir
prom_cons_f = "%s/promoter_conservation.500buff.with_DIGIT.txt" % feature_dir
exon_cons_f = "%s/exon_conservation.with_DIGIT.txt" % feature_dir
dna_len_f = "%s/transcript_length.with_DIGIT.txt" % feature_dir
rna_len_f = "%s/transcript_length_RNA.with_DIGIT.txt" % feature_dir
n_exons_f = "%s/n_exons_per_transcript.with_DIGIT.txt" % feature_dir
closest_DE_enh_tss_f = "%s/closest_DE_enh_to_TSS.with_DIGIT.fixed.txt" % feature_dir
closest_DE_enh_tran_f = "%s/closest_DE_enh_to_transcript.with_DIGIT.txt" % feature_dir


# In[6]:


# gwas file
gwas_dir = "../../../misc/06__gwas"
closest_endo_f = "%s/transcript_coords.closest_endo_cancer_snp.with_DIGIT.bed" % gwas_dir


# In[7]:


gene_map_f = "../../../misc/00__gene_metadata/gencode.v25lift37.GENE_ID_TRANSCRIPT_ID_MAP.with_DIGIT.fixed.txt"


# In[8]:


hits_f = "../../../data/02__screen/02__enrichment_data/enrichment_values.with_rna_seq.UPDATED.txt"


# ## 1. import data

# In[9]:


splicing = pd.read_table(splicing_f)


# In[10]:


gc = pd.read_table(gc_f, header=None)
gc.columns = ["transcript_id", "gc"]


# In[11]:


n_tss = pd.read_table(n_tss_f, delim_whitespace=True, header=None)
n_tss.columns = ["n_tss", "transcript_id"]


# In[12]:


n_enh = pd.read_table(n_enh_f, delim_whitespace=True, header=None)
n_enh.columns = ["n_enh", "transcript_id"]


# In[13]:


closest_enh_tss = pd.read_table(closest_enh_tss_f, header=None)
closest_enh_tss.columns = ["chr", "start", "end", "transcript_id", "score", "strand", "enh_chr", "enh_start", 
                           "enh_end", "closest_enh_id", "enh_len", "enh_strand", "enh_tss1", "enh_tss2",
                           "enh_blocks", "enh_nblocks", "enh_distblocks", "enh_endblocks", "enh_tss_dist"]
closest_enh_tss = closest_enh_tss[["transcript_id", "enh_tss_dist"]]


# In[14]:


closest_enh_tran = pd.read_table(closest_enh_tran_f, header=None)
closest_enh_tran.columns = ["chr", "start", "end", "transcript_id", "enh_chr", "enh_start", 
                           "enh_end", "closest_enh_id", "enh_len", "enh_strand", "enh_tss1", "enh_tss2",
                           "enh_blocks", "enh_nblocks", "enh_distblocks", "enh_endblocks", "enh_tran_dist"]
closest_enh_tran = closest_enh_tran[["transcript_id", "enh_tran_dist"]]


# In[15]:


prom_cons = pd.read_table(prom_cons_f)


# In[16]:


exon_cons = pd.read_table(exon_cons_f)


# In[17]:


dna_len = pd.read_table(dna_len_f, header=None)
dna_len.columns = ["transcript_id", "dna_len"]


# In[18]:


rna_len = pd.read_table(rna_len_f, header=None)
rna_len.columns = ["transcript_id", "rna_len"]


# In[19]:


n_exons = pd.read_table(n_exons_f, header=None)
n_exons.columns = ["n_exons", "gene_id", "transcript_id"]


# In[20]:


closest_endo = pd.read_table(closest_endo_f, sep="\t", header=None)
closest_endo.columns = ["chr", "start", "end", "transcript_id", "snp_chr", "snp_start", "snp_end",
                        "closest_endo_snp_id", "closest_endo_snp_disease", "closest_endo_snp_distance"]


# In[21]:


closest_DE_enh_tss = pd.read_table(closest_DE_enh_tss_f, header=None)
closest_DE_enh_tss.columns = ["chr", "start", "end", "transcript_id", "score", "strand", "enh_chr", "enh_start", 
                           "enh_end", "DE_enh_tss_dist"]
closest_DE_enh_tss = closest_DE_enh_tss[["transcript_id", "DE_enh_tss_dist"]]


# In[22]:


closest_DE_enh_tran = pd.read_table(closest_DE_enh_tran_f, header=None)
closest_DE_enh_tran.columns = ["chr", "start", "end", "transcript_id", "enh_chr", "enh_start", 
                           "enh_end", "DE_enh_tran_dist"]
closest_DE_enh_tran = closest_DE_enh_tran[["transcript_id", "DE_enh_tran_dist"]]


# In[23]:


gene_map = pd.read_table(gene_map_f, header=None)
gene_map.columns = ["gene_id", "transcript_id"]


# In[24]:


hits = pd.read_table(hits_f)


# ## 2. join transcript-level data w/ gene id

# In[25]:


print(len(gc))
gc = gc.merge(gene_map, on="transcript_id")
print(len(gc))


# In[26]:


print(len(n_tss))
n_tss = n_tss.merge(gene_map, on="transcript_id")
print(len(n_tss))


# In[27]:


print(len(n_enh))
n_enh = n_enh.merge(gene_map, on="transcript_id")
print(len(n_enh))


# In[28]:


print(len(closest_enh_tss))
closest_enh_tss = closest_enh_tss.merge(gene_map, on="transcript_id")
print(len(closest_enh_tss))


# In[29]:


print(len(closest_enh_tran))
closest_enh_tran = closest_enh_tran.merge(gene_map, on="transcript_id")
print(len(closest_enh_tran))


# In[30]:


print(len(prom_cons))
prom_cons = prom_cons.merge(gene_map, left_on="name", right_on="transcript_id")
print(len(prom_cons))


# In[31]:


print(len(exon_cons))
exon_cons = exon_cons.merge(gene_map, left_on="name", right_on="transcript_id")
print(len(exon_cons))


# In[32]:


print(len(dna_len))
dna_len = dna_len.merge(gene_map, on="transcript_id")
print(len(dna_len))


# In[33]:


print(len(rna_len))
rna_len = rna_len.merge(gene_map, on="transcript_id")
print(len(rna_len))


# In[34]:


print(len(closest_endo))
closest_endo = closest_endo.merge(gene_map, on="transcript_id")
print(len(closest_endo))


# In[35]:


print(len(closest_DE_enh_tss))
closest_DE_enh_tss = closest_DE_enh_tss.merge(gene_map, on="transcript_id")
print(len(closest_DE_enh_tss))


# In[36]:


print(len(closest_DE_enh_tran))
closest_DE_enh_tran = closest_DE_enh_tran.merge(gene_map, on="transcript_id")
print(len(closest_DE_enh_tran))


# ## 3. aggregate features to gene level

# In[37]:


gc_gene = gc.groupby("gene_id")["gc"].agg("mean").reset_index()
print(len(gc_gene))


# In[38]:


n_tss_gene = n_tss.groupby("gene_id")["n_tss"].agg("max").reset_index()
print(len(n_tss_gene))


# In[39]:


n_enh_gene = n_enh.groupby("gene_id")["n_enh"].agg("max").reset_index()
print(len(n_enh_gene))


# In[40]:


closest_enh_tss_gene = closest_enh_tss.groupby("gene_id")["enh_tss_dist"].agg("min").reset_index()
print(len(closest_enh_tss_gene))


# In[41]:


closest_enh_tran_gene = closest_enh_tran.groupby("gene_id")["enh_tran_dist"].agg("min").reset_index()
print(len(closest_enh_tran_gene))


# In[42]:


closest_DE_enh_tss_gene = closest_DE_enh_tss.groupby("gene_id")["DE_enh_tss_dist"].agg("min").reset_index()
print(len(closest_DE_enh_tss_gene))


# In[43]:


closest_DE_enh_tran_gene = closest_DE_enh_tran.groupby("gene_id")["DE_enh_tran_dist"].agg("min").reset_index()
print(len(closest_DE_enh_tran_gene))


# In[44]:


prom_cons_gene = prom_cons.groupby("gene_id")["median"].agg("max").reset_index()
prom_cons_gene.columns = ["gene_id", "prom_cons"]
print(len(prom_cons_gene))


# In[45]:


exon_cons_tx = exon_cons.groupby(["name", "gene_id"])["median"].agg("mean").reset_index()
exon_cons_gene = exon_cons_tx.groupby("gene_id")["median"].agg("max").reset_index()
exon_cons_gene.columns = ["gene_id", "exon_cons"]
print(len(exon_cons_gene))


# In[46]:


dna_len_gene = dna_len.groupby("gene_id")["dna_len"].agg("max").reset_index()
print(len(dna_len_gene))


# In[47]:


rna_len_gene = rna_len.groupby("gene_id")["rna_len"].agg("max").reset_index()
print(len(rna_len_gene))


# In[48]:


n_exons_gene = n_exons.groupby("gene_id")["n_exons"].agg("max").reset_index()
print(len(n_exons_gene))


# In[49]:


# same thing for RNA-seq data: sum up transcript expression levels
endo_exp = hits[["gene_id", "hESC_mean", "endo_mean"]].groupby("gene_id")[["hESC_mean", "endo_mean"]].agg("sum").reset_index()
print(len(endo_exp))


# In[50]:


# take transcript w/ maximum logfc expression
endo_fc = hits[["gene_id", "endo_hESC_abslog2fc"]].groupby("gene_id")["endo_hESC_abslog2fc"].agg("max").reset_index()
print(len(endo_fc))


# In[51]:


# need to also do this for gwas
closest_endo_gene = closest_endo[["gene_id", 
                                  "closest_endo_snp_distance"]].groupby("gene_id")["closest_endo_snp_distance"].agg("min").reset_index()
closest_endo_gene = closest_endo_gene.merge(closest_endo[["gene_id", "closest_endo_snp_distance", 
                                                          "closest_endo_snp_id", "closest_endo_snp_disease"]],
                                            on=["gene_id", 
                                                "closest_endo_snp_distance"]).drop_duplicates(subset=["gene_id", 
                                                                                                      "closest_endo_snp_distance"])
print(len(closest_endo_gene))


# ## 4. merge all genomic features into 1 dataframe

# In[52]:


data = splicing.merge(gc_gene, on="gene_id", how="outer")
print(len(data))


# In[53]:


data = data.merge(n_tss_gene, on="gene_id", how="left").merge(n_enh_gene, on="gene_id", how="left")
print(len(data))


# In[54]:


data = data.merge(closest_enh_tss_gene, on="gene_id", how="left").merge(closest_enh_tran_gene, 
                                                                        on="gene_id", how="left")
print(len(data))


# In[55]:


data = data.merge(closest_DE_enh_tss_gene, on="gene_id", how="left").merge(closest_DE_enh_tran_gene, 
                                                                           on="gene_id", how="left")
print(len(data))


# In[56]:


data = data.merge(prom_cons_gene, on="gene_id", how="left")
print(len(data))


# In[57]:


data = data.merge(exon_cons_gene, on="gene_id", how="left").merge(dna_len_gene, on="gene_id", how="left")
print(len(data))


# In[58]:


data = data.merge(rna_len_gene, on="gene_id", how="left").merge(n_exons_gene, on="gene_id", how="left")
print(len(data))


# In[59]:


data = data.merge(closest_endo_gene[["gene_id", "closest_endo_snp_distance", "closest_endo_snp_id",
                                     "closest_endo_snp_disease"]], on="gene_id", how="left")
print(len(data))


# In[60]:


# for n tss and n enh, NAs do not mean lack of data but mean 0 -- so replace NAs in these cols with 0
data["n_tss"] = data["n_tss"].fillna(0)
data["n_enh"] = data["n_enh"].fillna(0)


# In[61]:


data["short_gene_id"] = data["gene_id"].str.split("_", expand=True)[0]
data["shorter_gene_id"] = data["short_gene_id"].str.split(".", expand=True)[0]
data["minimal_biotype"] = data.apply(min_biotype, axis=1)
data.minimal_biotype.value_counts()


# In[62]:


# remove bad biotypes we don't care about (like pseudogenes) which will have a null gene_name value
data_filt = data[~pd.isnull(data["gene_name"])]
len(data_filt)


# ## 5. create df including only genes in screen + include endo RNAseq features

# In[63]:


genes_in_screen = hits["gene_id"].unique()
len(genes_in_screen)


# In[64]:


hits.is_hit.value_counts()


# In[65]:


gene_hit_status = hits[["gene_id", "gene_name", "ctrl_status", "cleaner_gene_biotype",
                        "is_hit"]].sort_values(by=["gene_id", "is_hit"]).drop_duplicates(subset="gene_id", 
                                                                                           keep="first")
gene_hit_status.head()


# In[66]:


print(len(data))
df_screen = data.merge(gene_hit_status, on=["gene_id", "gene_name", "cleaner_gene_biotype"])
print(len(df_screen))
df_screen.head()


# In[67]:


df_screen = df_screen.merge(endo_exp, on="gene_id").merge(endo_fc, on="gene_id")
len(df_screen)


# In[68]:


df_screen[df_screen["is_hit"] == "hit"].minimal_biotype.value_counts()


# In[69]:


tmp = df_screen[df_screen["is_hit"] == "hit"][["gene_name", "minimal_biotype", "ctrl_status", "cleaner_gene_biotype"]]
tmp


# In[70]:


tmp.groupby(["minimal_biotype", "ctrl_status"])["gene_name"].agg("count")


# In[71]:


tmp[(tmp["minimal_biotype"] == "mRNA") & (tmp["ctrl_status"] == "experimental")]


# ## 6. write final files

# ### general features for all genes

# In[72]:


meta_cols = ["gene_id", "gene_name", "csf", "cleaner_gene_biotype", "minimal_biotype"]
sub_feature_cols = ['max_eff', 'max_exp', 'gc', 'n_tss', 'n_enh', 'enh_tss_dist', 'enh_tran_dist', 'prom_cons',
                    'exon_cons', 'dna_len', 'rna_len', 'n_exons']


# In[73]:


all_cols = meta_cols + sub_feature_cols


# In[74]:


supp = data_filt[all_cols]
supp.head()


# In[75]:


supp[supp["gene_name"] == "DIGIT"]


# In[76]:


supp.to_csv("../../../data/03__features/all_features.tmp", sep="\t", index=False)


# ### + endo features for screen genes

# In[77]:


meta_cols = ["gene_id", "gene_name", "csf", "cleaner_gene_biotype", "minimal_biotype", "is_hit"]
sub_feature_cols = ['max_eff', 'max_exp', 'gc', 'n_tss', 'n_enh', 'enh_tss_dist', 'enh_tran_dist', 'prom_cons',
                    'exon_cons', 'dna_len', 'rna_len', 'n_exons', 'hESC_mean', 'endo_mean', 'endo_hESC_abslog2fc',
                    'closest_endo_snp_distance', 'closest_endo_snp_id', 'closest_endo_snp_disease', 'DE_enh_tss_dist',
                    'DE_enh_tran_dist']


# In[78]:


all_cols = meta_cols + sub_feature_cols


# In[79]:


supp = df_screen[all_cols]
supp.head()


# In[80]:


supp.to_csv("../../../data/03__features/SuppTable_S5.locus_features.txt", sep="\t", index=False)


# In[81]:


### look at GWAS hits
tmp = supp[supp["minimal_biotype"] == "lncRNA"]
tmp = tmp[tmp["is_hit"] == "hit"]
tmp.sort_values(by="closest_endo_snp_distance")[["gene_name", "closest_endo_snp_distance",
                                                 "closest_endo_snp_id", "closest_endo_snp_disease"]]


# In[82]:


supp[supp["gene_name"] == "FOXD3-AS1"].iloc[0]


# In[83]:


supp[supp["gene_name"] == "LINC00623"].iloc[0]


# In[ ]:




