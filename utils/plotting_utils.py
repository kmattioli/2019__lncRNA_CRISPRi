#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


PAPER_PRESET = {"style": "ticks", "font": "Helvetica", "context": "paper", 
                "rc": {"font.size":7,"axes.titlesize":7,
                       "axes.labelsize":7, 'axes.linewidth':0.5,
                       "legend.fontsize":6, "xtick.labelsize":6,
                       "ytick.labelsize":6, "xtick.major.size": 3.0,
                       "ytick.major.size": 3.0, "axes.edgecolor": "black",
                       "xtick.major.pad": 3.0, "ytick.major.pad": 3.0}}
PAPER_FONTSIZE = 7


# In[ ]:


def mimic_r_boxplot(ax):
    for i, patch in enumerate(ax.artists):
        r, g, b, a = patch.get_facecolor()
        col = (r, g, b, 1)
        patch.set_facecolor((r, g, b, .5))
        patch.set_edgecolor((r, g, b, 1))

        # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        line_order = ["lower", "upper", "whisker_1", "whisker_2", "med", "fliers"]
        for j in range(i*6,i*6+6):
            elem = line_order[j%6]
            line = ax.lines[j]
            if "whisker" in elem:
                line.set_visible(False)
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)
            if "fliers" in elem:
                line.set_alpha(0.5)


# In[ ]:


def annotate_pval(ax, x1, x2, y, h, text_y, val, fontsize):
    from decimal import Decimal
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c="black", linewidth=0.5)
    if val < 0.0001:
        #text = "{:.2e}".format(Decimal(val))
        text = "**"
    elif val < 0.05:
        #text = "%.4f" % val
        text = "*"
    else:
        #text = "%.4f" % val
        text = "n.s."
    ax.text((x1+x2)*.5, text_y, text, ha='center', va='bottom', color="black", size=fontsize)

