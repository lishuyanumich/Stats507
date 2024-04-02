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

# # Question 0 - R-Squared Warmup [20 points]

# In this question you will fit a model to the ToothGrowth data used in the notes on Resampling and Statsmodels-OLS. Read the data, log transform tooth length, and then fit a model with indpendent variables for supplement type, dose (as categorical), and their interaction. Demonstrate how to compute the R-Squared and Adjusted R-Squared values and compare your compuations to the attributes (or properties) already present in the result object.

# model imports
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import statsmodels.formula.api as smf
import statsmodels.api as sm
from os.path import exists

# use the "ToothGrowth" data from the R datasets package.
file = 'tooth_growth.feather'
if exists(file):
    tg_data = pd.read_feather(file)
else: 
    tooth_growth = sm.datasets.get_rdataset('ToothGrowth')
    #print(tooth_growth.__doc__)
    tg_data = tooth_growth.data
    tg_data.to_feather(file)

# log transform tooth length and transform supplement type, dose as categorical type
trans_tg_data = tg_data
trans_tg_data["log_len"] = np.log(trans_tg_data["len"])
trans_tg_data['supp'] = pd.get_dummies(trans_tg_data['supp'])['OJ']
trans_tg_data

# fit a model
mod1 = sm.OLS.from_formula('log_len ~ supp*dose', data=trans_tg_data)
res1 = mod1.fit()
res1.summary2()

# The $R^2 = 0.683$ and $R_{Adj}^2 = 0.666$ 

# How to compute R-squared
mean_log_len = np.mean(trans_tg_data["log_len"])
trans_tg_data["SST"] = trans_tg_data.apply(lambda x: (x["log_len"]-mean_log_len)**2, axis=1)
TSS = sum(trans_tg_data["SST"])
RSS = np.sum(res1.resid**2)
R_Squared = 1 - RSS/TSS
print(R_Squared)
R_adj = 1 - (res1.df_resid+res1.df_model-1)/(res1.df_resid-1)*(RSS/TSS)
print(R_adj)

# The calculated R-squared and Adjusted R-Squared are the same with the table above.
