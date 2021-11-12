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

# # Name: Li Shuyan
# # UMID: 63161545

# # Question 0 - Topics in Pandas [25 points]

# For this question, please pick a topic - such as a function, class, method, recipe or idiom related to the pandas python library and create a short tutorial or overview of that topic. 

# # Duplicate labels

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
