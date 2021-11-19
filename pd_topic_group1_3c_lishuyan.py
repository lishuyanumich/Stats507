# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Topics in Pandas
# **Stats 507, Fall 2021** 
# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
# + [Duplicate labels](#Duplicate-labels) 
# + [Topic 2 Title](#Topic-2-Title)
#
# # Duplicate labels
#
# **Xinyi Liu**
# **xyiliu@umich.edu**

import numpy as np
import pandas as pd

# * Some methods cannot be applied on the data series which have duplicate labels (such as `.reindex()`, it will cause error!),
# * Error message of using the function above: "cannot reindex from a duplicate axis".

series1 = pd.Series([0, 0, 0], index=["A", "B", "B"])
#series1.reindex(["A", "B", "C"]) 

# * When we slice the unique label, it returns a series,
# * when we slice the duplicate label, it will return a dataframe.

df1 = pd.DataFrame([[1,1,2,3],[1,1,2,3]], columns=["A","A","B","C"])
df1["B"]
df1["A"]

# * Check if the label of the row is unique by apply `index.is_unique ` to the dataframe, will return a boolean, either True or False.
# * Check if the column label is unique by `columns.is_unique`, will return a boolean, either True or False.

df1.index.is_unique
df1.columns.is_unique


# * When we moving forward of the data which have duplicated lables, to keep the data clean, we do not want to keep duplicate labels. 
# * In pandas version 1.2.0 (cannot work in older version), we can make it disallowing duplicate labels as we continue to construct dataframe by `.set_flags(allows_duplicate_labels=False)`
# * This function applies to both row and column labels for a DataFrame.

# +
#df1.set_flags(allows_duplicate_labels=False) 
## the method above cannot work on my end since my panda version is 1.0.5
# -

# Reference: https://pandas.pydata.org/docs/user_guide/duplicates.html

# ## Topics in Pandas
# **Stats 507, Fall 2021** 
# ## Contents
#
# + [Function](Function) 
# + [Topic 2 Title](#Topic-2-Title)
#
# ## Function
# Defining parts of a Function
#
# **Sachie Kakehi**
#
# sachkak@umich.edu

# ## What is a Function
#
#   - A *function* is a block of code which only runs when it is called.
#   - You can pass data, known as *parameters* or *arguments*, into a function.
#   - A function can return data as a result.

# ## Parts of a Function
#
#   - A function is defined using the $def$ keyword
#   - Parameters or arguments are specified after the function name, inside the parentheses.
#   - Within the function, the block of code is defined, often $print$ or $return$ values.

# ## Example
#
#   - The following function multiplies the parameter, x, by 5:
#   - Note : it is good practice to add a docstring explaining what the function does, and what the parameters and returns are. 

def my_function(x):
    """
    The function multiplies the parameter by 5.
    
    Parameters
    ----------
    x : A float or integer.
    
    Returns
    -------
    A float or integer multiplied by 5. 
    """
    return 5 * x
print(my_function(3))

# ## Topics in Pandas
# **Stats 507, Fall 2021** 
# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
# + [Duplicate labels](#Duplicate-labels) 
# + [Topic 2 Title](#Topic-2-Title)
# # Duplicate labels
#
# **Shuyan Li**
# **lishuyan@umich.edu**
#
# Real-world data is always messy. Since index objects are not required to be unique, sometimes we can have duplicate rows or column labels. 
# In this section, we first show how duplicate labels change the behavior of certain operations. Then we will use pandas to detect them if there are any duplicate labels, or to deal with duplicate labels.
#
# - Consequences of duplicate labels
# - Duplicate label detection
# - Deal with duplicate labels

import pandas as pd
import numpy as np

# Generate series with duplicate labels
s1 = pd.Series([0,4,6], index=["A", "B", "B"])

# ## Consequences of duplicate labels
# Some pandas methods (`Series.reindex()` for example) don’t work with duplicate indexes. The output can’t be determined, and so pandas raises.

s1.reindex(["A", "B", "C"])

# Other methods, like indexing, can cause unusual results. Normally indexing with a scalar will reduce dimensionality. Slicing a DataFrame with a scalar will return a Series. Slicing a Series with a scalar will return a scalar. However, with duplicate labels, this isn’t the case.

df1 = pd.DataFrame([[0, 1, 2], [3, 4, 5]], columns=["A", "A", "B"])
df1

# If we slice 'B', we get back a Series.

df1["B"] # This is a series

# But slicing 'A' returns a DataFrame. Since there are two "A" columns.

df1["A"] # This is a dataframe

# This applies to row labels as well.

df2 = pd.DataFrame({"A": [0, 1, 2]}, index=["a", "a", "b"])
df2

df2.loc["b", "A"]  # This is a scalar.

df2.loc["a", "A"]  # This is a Series.

# ## Duplicate Label Detection
# We can check whether an Index (storing the row or column labels) is unique with `Index.is_unique`:

df2.index.is_unique # There are duplicate indexes in df2.

df2.columns.is_unique # Column names of df2 are unique.

# `Index.duplicated()` will return a boolean ndarray indicating whether a label is repeated.

df2.index.duplicated()

# ## Deal with duplicate labels
# - `Index.duplicated()` can be used as a boolean filter to drop duplicate rows.

df2.loc[~df2.index.duplicated(), :]

# - We can use `groupby()` to handle duplicate labels, rather than just dropping the repeats. 
#
# For example, we’ll resolve duplicates by taking the average of all rows with the same label.

df2.groupby(level=0).mean()

# Reference: https://pandas.pydata.org/docs/user_guide/duplicates.html


