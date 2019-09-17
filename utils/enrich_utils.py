
# coding: utf-8

# In[ ]:


import pandas as pd
from ast import literal_eval


# In[ ]:


def fix_id_dupes(row, column):
    x = row[column]
    if "," in x:
        x = literal_eval(x)
        x = list(set(x))
    return str(x)


# In[ ]:


def get_scramble_id(row):
    if row.ctrl_status != "scramble":
        return "none"
    else:
        scramble_id = row.sgRNA_id.split("__")[0]
        return scramble_id


# In[ ]:


def get_group_id(row):
    if row.ctrl_status == "scramble":
        return row.scramble_id
    else:
        return row.tss_id_hg38


# In[ ]:


def fix_ctrl_status(row):
    if row.endo_ctrl_val:
        return "control"
    else:
        if row.ctrl_status == "scramble":
            return "scramble"
        else:
            return "experimental"


# In[ ]:


def fix_ctrl_status_w_DE_mRNAs(row):
    if row.endo_ctrl_val:
        return "control"
    else:
        if row.ctrl_status == "scramble":
            return "scramble"
        elif row.ctrl_status == "control":
            return "mRNA"
        else:
            return "experimental"


# In[ ]:


def tidy_split(df, column, sep='|', keep=False):
    """
    Split the values of a column and expand so the new DataFrame has one split
    value per row. Filters rows where the column is missing.
    Params
    ------
    df : pandas.DataFrame
        dataframe with the column to split and expand
    column : str
        the column to split and expand
    sep : str
        the string used to split the column's values
    keep : bool
        whether to retain the presplit value as it's own row
    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the same columns as `df`.
    """
    indexes = list()
    new_values = list()
    df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes, :].copy()
    new_df[column] = new_values
    return new_df


# In[ ]:


def clean_split_col(row, col):
    old = row[col].strip()
    if old.startswith("["):
        new = old.strip()[2:-1]
    elif old[-1] == "]":
        new = old.strip()[1:-2]
    elif old[0] == "'":
        new = old.strip()[1:-1]
    else:
        new = old
    return new


# In[ ]:


def is_hit(row):
    if pd.isnull(row.padj):
        return "no hit"
    elif row.padj < 0.1:
        return "stringent hit"
    else:
        return "lenient hit"

