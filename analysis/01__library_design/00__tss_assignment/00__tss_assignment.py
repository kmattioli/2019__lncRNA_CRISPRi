#!/usr/bin/env python
# coding: utf-8

# # 00__tss_assignment
# 
# in this notebook, i assign CAGE TSSs to transcripts (either Fantom-CAT assignments, when possible, or closest CAGE peak if there is one w/in 400bp). 
# 
# figures in this notebook:
# - Fig S3C: overview of TSS assignments for library
# - Fig S3D: overview of number of transcripts with multiple TSS assignments

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybedtools as pb
import seaborn as sns
import sys

from scipy import stats

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


def get_id_assignment(row):
    if not pd.isnull(row["cage_id_lv4_stringent"]):
        return row["cage_id_lv4_stringent"]
    elif not pd.isnull(row["cage_id_lv3_robust"]):
        return row["cage_id_lv3_robust"]
    elif not pd.isnull(row["cage_id_lv2_permissive"]):
        return row["cage_id_lv2_permissive"]
    elif not pd.isnull(row["cage_id_lv1_raw"]):
        return row["cage_id_lv1_raw"]
    else:
        return "none"


# In[4]:


def get_id_assignment_type(row):
    if not pd.isnull(row["cage_id_lv4_stringent"]):
        return "lv4_stringent"
    elif not pd.isnull(row["cage_id_lv3_robust"]):
        return "lv3_robust"
    elif not pd.isnull(row["cage_id_lv2_permissive"]):
        return "lv2_permissive"
    elif not pd.isnull(row["cage_id_lv1_raw"]):
        return "lv1_raw"
    else:
        return "none"


# In[5]:


def get_id_closest(row):
    if not pd.isnull(row["closest_cage_lv4_stringent"]):
        return row["closest_cage_lv4_stringent"]
    elif not pd.isnull(row["closest_cage_lv3_robust"]):
        return row["closest_cage_lv3_robust"]
    elif not pd.isnull(row["closest_cage_lv2_permissive"]):
        return row["closest_cage_lv2_permissive"]
    elif not pd.isnull(row["closest_cage_lv1_raw"]):
        return row["closest_cage_lv1_raw"]
    else:
        return "none"


# In[6]:


def get_id_closest_type(row):
    if not pd.isnull(row["closest_cage_lv4_stringent"]):
        return "lv4_stringent"
    elif not pd.isnull(row["closest_cage_lv3_robust"]):
        return "lv3_robust"
    elif not pd.isnull(row["closest_cage_lv2_permissive"]):
        return "lv2_permissive"
    elif not pd.isnull(row["closest_cage_lv1_raw"]):
        return "lv1_raw"
    else:
        return "none"


# In[7]:


def get_final_cage(row):
    if row.cage_id_assignment != "none" and row.cage_id_assignment_type != "lv1_raw":
        return row.cage_id_assignment
    elif row.cage_id_closest != "none":
        return row.cage_id_closest
    else:
        return "none"


# In[8]:


def get_final_type(row):
    if row.cage_id_assignment != "none" and row.cage_id_assignment_type != "lv1_raw":
        return "FANTOM CAT assignment"
    elif row.cage_id_closest != "none":
        return "closest CAGE"
    else:
        return "annotation"


# In[9]:


def get_final_level(row):
    if row.cage_id_assignment_type != "none" and row.cage_id_assignment_type != "lv1_raw":
        return row.cage_id_assignment_type
    elif row.cage_id_closest_type != "none":
        return row.cage_id_closest_type
    else:
        return "none"


# In[10]:


def get_final_tss_start(row):
    if not pd.isnull(row.tss_start):
        return row.tss_start
    else:
        return row.start


# In[11]:


def get_final_tss_end(row):
    if not pd.isnull(row.tss_end):
        return row.tss_end
    else:
        return row.end


# ## variables

# In[12]:


expr_f = "../../../data/00__rna_seq/01__processed_results/rna_seq_results.tsv"


# In[13]:


tss_f = "../../../misc/00__gene_metadata/gencode.v25lift37.annotation.TRANSCRIPT_TSS_MAP.txt"


# In[14]:


fantom_dir = "../../../misc/03__fantom"


# In[15]:


fantom_types = ["lv1_raw", "lv2_permissive", "lv3_robust", "lv4_stringent"]


# In[16]:


fantom_bed_ending = "CAGE_cluster.bed.gz"


# In[17]:


fantom_id_ending = "info_table.ID_mapping.tsv.gz"


# In[18]:


pos_ctrl_dir = "../../../misc/04__pos_ctrls"


# ## 1. import data

# In[19]:


fs = ["hESC_ctrls.txt", "endo_ctrls.txt", "meso_ctrls.txt"]
pos_ctrls = pd.DataFrame()
for f in fs:
    tmp = pd.read_table("%s/%s" % (pos_ctrl_dir, f), header=None)
    tmp.columns = ["gene_name"]
    tmp["lin"] = f.split("_")[0]
    pos_ctrls = pos_ctrls.append(tmp)
print(len(pos_ctrls))
pos_ctrls.sample(5)


# In[20]:


expr = pd.read_table(expr_f, sep="\t")
expr.head()


# In[21]:


tss = pd.read_table(tss_f, sep="\t", header=None)
tss.columns = ["transcript_id", "gene_id", "gene_name", "transcript_name", "chr", "start", "end", "strand"]
tss.head()


# In[22]:


fantom_beds = {}
fantom_ids = {}
for t in fantom_types:
    bed_f = "%s/FANTOM_CAT.%s.%s" % (fantom_dir, t, fantom_bed_ending)
    ids_f = "%s/FANTOM_CAT.%s.%s" % (fantom_dir, t, fantom_id_ending)
    bed = pd.read_table(bed_f, sep="\t", header=None)
    bed.columns = ["chr", "peak_start", "peak_end", "peak_id", "score", "strand", "tss_start", "tss_end", "x", "y", "z", "a"]
    bed.drop(["x", "y", "z", "a"], axis=1, inplace=True)
    ids = pd.read_table(ids_f, sep="\t")
    ids.columns = ["gene_id", "transcript_id", "cage_id_%s" % t]
    fantom_beds[t] = bed
    fantom_ids[t] = ids


# In[23]:


fantom_beds["lv1_raw"].head()


# In[24]:


fantom_ids["lv1_raw"].head()


# ## 2. filter transcripts to those expr in lineage & non-coding & pos ctrls

# ### get list of pos ctrls - top 100 transcripts in each lineage

# In[25]:


hESC_manual_adds = ["FOXD3", "SMARCA4", "FOXO1", "FZD7", "POU5F1"]
hESC_manual_removes = ["ENST00000512818.5_1"]
endo_manual_adds = ["SMARCD3"]


# In[26]:


expr_pos = pd.DataFrame()
for lin in ["hESC", "endo", "meso"]:
    genes = list(pos_ctrls[pos_ctrls["lin"] == lin]["gene_name"])
    expr_sub = expr[expr["gene_name"].isin(genes)]
    expr_samp = expr_sub.sort_values(by="%s_mean" % lin, ascending=False).head(100)
    if lin == "hESC":
        extra = expr[(expr["gene_name"].isin(hESC_manual_adds)) & (expr["threshold"] == "expressed") &
                     ~(expr["transcript_id"].isin(hESC_manual_removes))]
        expr_samp = expr_samp.append(extra)
    elif lin == "endo":
        extra = expr[(expr["gene_name"].isin(endo_manual_adds)) & (expr["threshold"] == "expressed")]
        expr_samp = expr_samp.append(extra)
    expr_pos = expr_pos.append(expr_samp)
print(len(expr_pos))
expr_pos.sample(5)


# In[27]:


expr.csf.value_counts()


# In[28]:


expr_filt = expr[(expr["threshold"] == "expressed") & (expr["csf"] == "lncRNA_good_csf")].append(expr_pos)
len(expr_filt)


# ## 3. merge TSS info

# In[29]:


expr_tss = expr_filt.merge(tss, on=["transcript_id", "gene_id", "gene_name"], how="left")
expr_tss.head()


# In[30]:


len(expr_tss)


# In[31]:


sum(pd.isnull(expr_tss["chr"]))


# DIGIT is null because it was not in GENCODE file -- add manually

# ## 4. see if our transcripts have TSSs already assigned by fantom cat

# In[32]:


expr_tss["transcript_id_short"] = expr_tss["transcript_id"].str.split("_", expand=True)[0]


# In[33]:


for t in fantom_types:
    ids = fantom_ids[t]
    expr_tss = expr_tss.merge(ids, left_on="transcript_id_short", right_on="transcript_id", how="left", 
                              suffixes=["", "_tmp"])
    expr_tss.drop(["gene_id_tmp", "transcript_id_tmp"], axis=1, inplace=True)
expr_tss.head()


# In[34]:


expr_tss["cage_id_assignment"] = expr_tss.apply(get_id_assignment, axis=1)


# In[35]:


expr_tss["cage_id_assignment_type"] = expr_tss.apply(get_id_assignment_type, axis=1)


# In[36]:


expr_tss.drop(["cage_id_lv1_raw", "cage_id_lv2_permissive", "cage_id_lv3_robust", "cage_id_lv4_stringent"],
              axis=1, inplace=True)
expr_tss.head()


# In[37]:


expr_tss.cage_id_assignment_type.value_counts()


# we can assign ~half of transcripts a fantom cat cage peak based on their existing annotations (though not sure we can trust those level 1 assignments)

# ## 5. find closest annotated CAGE peak

# In[38]:


expr_tss = expr_tss[~pd.isnull(expr_tss["chr"])]
len(expr_tss)


# In[39]:


expr_tss_tmp = expr_tss[["chr", "start", "end", "transcript_id_short", "overall_mean", "strand"]]
expr_tss_tmp["start"] = expr_tss_tmp["start"].astype(int)
expr_tss_tmp["end"] = expr_tss_tmp["end"].astype(int)
expr_tss_tmp = expr_tss_tmp.sort_values(by=["chr", "start"])
expr_tss_tmp.head()


# In[40]:


# write this to csv temporarily to do bedtools intersections
expr_tss_tmp.to_csv("transcripts.tmp.bed", sep="\t", header=False, index=False)
expr_tss_tmp_bed = pb.BedTool("transcripts.tmp.bed")


# In[41]:


# for every fantom type, intersect our transcripts with cage peaks
for t in fantom_types:
    fantom_bed_f = "%s/FANTOM_CAT.%s.%s" % (fantom_dir, t, fantom_bed_ending)
    fantom_bed = pb.BedTool(fantom_bed_f).to_dataframe()
    fantom_bed.columns = ["chr", "start", "end", "id", "score", "strand", "tss_start", "tss_end", "x", "y", "z", "a"]
    tss_bed = fantom_bed[["chr", "tss_start", "tss_end", "id", "score", "strand"]].sort_values(by=["chr", 
                                                                                                   "tss_start"])
    tss_bed.to_csv("cage.tmp.bed", sep="\t", header=False, index=False)
    tss_bed = pb.BedTool("cage.tmp.bed")
    closest = expr_tss_tmp_bed.closest(tss_bed, s=True, d=True).to_dataframe()
    closest.columns = ["t_chr", "t_start", "t_end", "t_id", "t_score", "t_strand",
                       "c_chr", "c_start", "c_end", "c_id", "c_score", "c_strand", "dist"]
    closest_sub = closest[closest["dist"] < 400][["t_id", "c_id", "c_score"]]
    
    # find max score within these
    closest_max = closest.groupby(["t_id"])["c_score"].agg("max").reset_index()
    closest_max_sub = closest_max.merge(closest_sub, on=["c_score", "t_id"], how="left")
    closest_max_sub.drop("c_score", axis=1, inplace=True)
    closest_max_sub.columns = ["transcript_id_short", "closest_cage_%s" % t]
    expr_tss = expr_tss.merge(closest_max_sub, on="transcript_id_short", how="left")

expr_tss.head()


# In[42]:


expr_tss["cage_id_closest"] = expr_tss.apply(get_id_closest, axis=1)


# In[43]:


expr_tss["cage_id_closest_type"] = expr_tss.apply(get_id_closest_type, axis=1)


# In[44]:


expr_tss.drop(["closest_cage_lv1_raw", "closest_cage_lv2_permissive", "closest_cage_lv3_robust", 
               "closest_cage_lv4_stringent"], axis=1, inplace=True)
expr_tss.head()


# ## 6. classify transcripts into different tss categories

# In[45]:


expr_tss["cage_id_final"] = expr_tss.apply(get_final_cage, axis=1)


# In[46]:


expr_tss["tss_type_final"] = expr_tss.apply(get_final_type, axis=1)


# In[47]:


expr_tss["fantom_level_final"] = expr_tss.apply(get_final_level, axis=1)


# In[48]:


expr_tss.head()


# ## 7. get TSSs for cage peaks

# In[49]:


expr_tss = expr_tss.merge(fantom_beds["lv1_raw"][["peak_id", "tss_start", "tss_end"]], left_on="cage_id_final",
                          right_on="peak_id", how="left")
expr_tss.head()


# In[50]:


expr_tss["final_start"] = expr_tss.apply(get_final_tss_start, axis=1)


# In[51]:


expr_tss["final_end"] = expr_tss.apply(get_final_tss_end, axis=1)
expr_tss.head()


# In[52]:


expr_tss.columns


# In[53]:


expr_tss_final = expr_tss[["transcript_id", "transcript_name", "gene_id", "gene_name", "csf", "cleaner_gene_biotype",
                           "hESC_mean", "endo_mean", "meso_mean", "overall_mean", "threshold",
                           "qval_hESC_endo", "qval_hESC_meso", "endo_hESC_log2fc", "meso_hESC_log2fc",
                           "chr", "final_start", "final_end", "strand", "cage_id_final", "tss_type_final", 
                           "fantom_level_final"]]
expr_tss_final["final_start"] = expr_tss_final["final_start"].astype(int)
expr_tss_final["final_end"] = expr_tss_final["final_end"].astype(int)
expr_tss_final.head()


# ## 8. clean up column names

# In[54]:


expr_tss_final.columns = ["transcript_id", "transcript_name", "gene_id", "gene_name", "csf", "cleaner_biotype",
                           "hESC_mean", "endo_mean", "meso_mean", "overall_mean", "threshold",
                           "qval_hESC_endo", "qval_hESC_meso", "endo_hESC_log2fc", "meso_hESC_log2fc",
                           "chr", "tss_start", "tss_end", "strand", "cage_id", "tss_type", "fantom_level"]
expr_tss_final = expr_tss_final[["transcript_id", "transcript_name", "gene_id", "gene_name", "csf", "cleaner_biotype",
                                 "hESC_mean", "endo_mean", "meso_mean", "overall_mean", "threshold",
                                 "qval_hESC_endo", "qval_hESC_meso", "endo_hESC_log2fc", "meso_hESC_log2fc",
                                 "chr", "tss_start", "tss_end", "strand", "tss_type", "fantom_level", "cage_id"]]
expr_tss_final.head()


# In[55]:


expr_tss_final.drop_duplicates(inplace=True)
len(expr_tss_final)


# In[56]:


expr_tss_final.groupby(["transcript_id"])["tss_start"].agg("count").reset_index().sort_values(by="tss_start", ascending=False).head()


# ## 9. plot the #s of tss categories

# In[57]:


expr_tss_final_grp = expr_tss_final.groupby(["tss_type", "fantom_level"])["cage_id"].agg("count").unstack()
expr_tss_final_grp


# In[58]:


sns.palplot(sns.color_palette("deep"))


# In[59]:


pal = {"none": "gray", "lv1_raw": sns.color_palette()[3], "lv2_permissive": sns.color_palette()[1],
       "lv3_robust": sns.color_palette()[0], "lv4_stringent": sns.color_palette()[2]}
colors = [sns.color_palette()[3], sns.color_palette()[1], sns.color_palette()[0], sns.color_palette()[2], "gray"]


# In[60]:


ax = expr_tss_final_grp.plot.bar(stacked=True, color=colors, figsize=(2, 1.75))

# annotate totals
expr_tss_final_sum = expr_tss_final_grp.sum(axis=1).reset_index()
for i, x in enumerate(list(expr_tss_final_sum["tss_type"])):
    y = expr_tss_final_sum[expr_tss_final_sum["tss_type"] == x][0].iloc[0]
    ax.text(i, y+100, int(y), horizontalalignment="center")
    
ax.set_ylim((0, 8000))
ax.set_ylabel("count of transcripts")
ax.set_xlabel("")
ax.set_title("count of transcripts by TSS type\n& FANTOM CAT level (where applicable)")
ax.set_xticklabels(["FANTOM CAT assignment", "annotation", "closest CAGE"], rotation=50, ha="right", va="top")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

plt.savefig("FigS3C.pdf", dpi="figure", bbox_inches="tight")


# ## 10. count # of expressed TSSs per gene

# In[61]:


expr_tss_final.head()


# In[62]:


count_per_gene = expr_tss_final.groupby("gene_name")["transcript_id"].agg("count").reset_index()
count_per_gene.sample(5)


# In[63]:


plt.figure(figsize=(5, 1.5))
ax = sns.countplot(x="transcript_id", data=count_per_gene)

# annotate totals
for p in ax.patches:
    w = p.get_width()
    x = p.get_x()
    y = p.get_y()
    h = p.get_height()
    
    ax.text(x + 0.75, h, int(h), ha="center", va="bottom", rotation=50) 

ax.set_ylim((0, 8000))
plt.title("count of genes with x # of TSSs expressed")
plt.xlabel("count of TSSs")
plt.ylabel("count of lncRNA genes")
plt.savefig("FigS3D.pdf", dpi="figure", bbox_inches="tight")


# In[64]:


count_per_gene.sort_values(by="transcript_id", ascending=False).head(10)


# In[65]:


expr_tss_final[expr_tss_final["gene_name"] == "PABPC1L2B-AS1"]


# In[66]:


expr_tss_morethan1 = expr_tss_final[expr_tss_final["gene_name"].isin(count_per_gene[count_per_gene["transcript_id"]>1]["gene_name"])]
hESC_std_per_gene = expr_tss_morethan1.groupby(["gene_name"])["hESC_mean"].agg(["std", "count"]).reset_index()
endo_std_per_gene = expr_tss_morethan1.groupby(["gene_name"])["endo_mean"].agg(["std", "count"]).reset_index()
meso_std_per_gene = expr_tss_morethan1.groupby(["gene_name"])["meso_mean"].agg(["std", "count"]).reset_index()
hESC_std_per_gene.head()


# In[67]:


fig = plt.figure(figsize=(2, 1))
sns.kdeplot(np.log(hESC_std_per_gene["std"]+1), label="hESC", shade=True)
sns.kdeplot(np.log(endo_std_per_gene["std"]+1), label="endo", shade=True)
sns.kdeplot(np.log(meso_std_per_gene["std"]+1), label="meso", shade=True)
plt.xlabel("log(standard deviation of TSS expression per gene)")
plt.ylabel("density")
plt.title("expression variance for genes with >1 TSS")
#plt.savefig("expr_var_for_genes_with_mult_tss", dpi="figure", bbox_inches="tight")


# ## 12. write final files

# In[68]:


filename = "../../../data/01__design/00__tss_list/all_lncRNA_and_ctrl_TSSs_final.txt"
expr_tss_final.to_csv(filename, sep="\t", index=False)


# In[69]:


filename = "../../../misc/04__pos_ctrls/top_100_picked_positive_ctrls.txt"
expr_pos = expr_pos.merge(pos_ctrls, on="gene_name")
expr_pos.to_csv(filename, sep="\t", index=False)


# In[70]:


expr_pos[expr_pos["gene_name"]=="CER1"]


# In[71]:


len(expr_pos)


# In[72]:


get_ipython().system('rm cage.tmp.bed')


# In[73]:


get_ipython().system('rm transcripts.tmp.bed')


# In[ ]:




