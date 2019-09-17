
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


def get_csf_class(row):
    if row.biotype == "protein_coding":
        return "protein_coding"
    elif row.gene_id == "DIGIT":
        return "lncRNA_good_csf"
    elif not pd.isnull(row.csf_class):
        return row.csf_class
    else:
        return "unclear"


# In[3]:


def get_transcript_biotype(row):
    if row.biotype == "protein_coding":
        return "protein_coding"
    elif row.gene_id == "DIGIT":
        return "tss_overlapping__lncRNA"
    elif not pd.isnull(row.biotype_lncRNA):
        return row.biotype_lncRNA
    else:
        return "biotype not considered"


# In[4]:


def get_gene_biotype(row):
    if row.biotype == "protein_coding":
        return "protein_coding"
    elif row.gene_id == "DIGIT":
        return "tss_overlapping__lncRNA"
    elif not pd.isnull(row.gene_biotype):
        return row.gene_biotype
    else:
        return "biotype not considered"


# In[5]:


def get_expr_thresh(row):
    thresh = 0.1
    if row.hESC_mean > thresh or row.endo_mean > thresh or row.meso_mean > thresh:
        return "expressed"
    else:
        return "not expressed"


# In[6]:


def clean_biotype(row, col):
    if "," in row[col]:
        biotypes = row[col].split(",")
        biotypes = [x.split("__")[0] for x in biotypes]
        biotypes = set(biotypes)
        biotypes = list(biotypes)
        biotypes.sort()
        biotypes_str = ", ".join(biotypes)
        return biotypes_str
    else:
        return row[col]


# In[7]:


def cleaner_biotype(row, col):
    if "divergent" in row[col] or "tss_overlapping" in row[col]:
        return "promoter_overlap"
    elif "antisense" in row[col] or "sense_overlapping" in row[col]:
        return "transcript_overlap"
    elif "gene_" in row[col]:
        return "gene_nearby"
    else:
        return row[col]


# In[8]:


def coding_type(row):
    if row.csf == "protein_coding":
        return "coding"
    else:
        return "non-coding"


# In[9]:


def hESC_thresh(row):
    thresh = 0.1
    if row.hESC_mean > thresh:
        return "expressed"
    else:
        return "not expressed"

def endo_thresh(row):
    thresh = 0.1
    if row.endo_mean > thresh:
        return "expressed"
    else:
        return "not expressed"

def meso_thresh(row):
    thresh = 0.1
    if row.meso_mean > thresh:
        return "expressed"
    else:
        return "not expressed"


# In[10]:


def get_expr_profile(row):
    if row.hESC_mean > 0.1 and row.endo_mean > 0.1 and row.meso_mean > 0.1:
        return "hESC, endo, and meso"
    if row.hESC_mean > 0.1 and row.endo_mean > 0.1:
        return "hESC and endo"
    if row.endo_mean > 0.1 and row.meso_mean > 0.1:
        return "endo and meso"
    if row.hESC_mean > 0.1 and row.meso_mean > 0.1:
        return "hESC and meso"
    if row.hESC_mean > 0.1:
        return "hESC only"
    if row.endo_mean > 0.1:
        return "endo only"
    if row.meso_mean > 0.1:
        return "meso only"
    else:
        return "not expressed"

