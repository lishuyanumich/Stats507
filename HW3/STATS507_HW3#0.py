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

# # Question 0 - RECS and Replicate Weights [15 points]

import pandas as pd

# ## Data Files
# Three data files are needed:
#
# - 2015 RECS microdata files:
# https://www.eia.gov/consumption/residential/data/2015/csv/recs2015_public_v4.csv
# - 2009 RECS microdata files:
# https://www.eia.gov/consumption/residential/data/2009/csv/recs2009_public.csv
# - 2009 data year the replicate weights file:
# https://www.eia.gov/consumption/residential/data/2009/csv/recs2009_public_repweights.csv

base_url = "https://www.eia.gov/consumption/residential/data/"

## get link from the website
url_2015 = base_url + "2015/csv/recs2015_public_v4.csv"
## read the csv file
df_2015 = pd.read_csv(url_2015)
## make column names in lower case and display year 2015 recs file
df_2015 = df_2015.rename(columns=str.lower)
df_2015

## get link from the website
url_2009 = base_url + "2009/csv/recs2009_public.csv"
## read the csv file
df_2009 = pd.read_csv(url_2009)
## make column names in lower case and display year 2009 recs file
df_2009 = df_2009.rename(columns=str.lower)
df_2009

## get link from the website
url_2009_weight = base_url + "2009/csv/recs2009_public_repweights.csv"
## read the csv file
df_2009_weight = pd.read_csv(url_2009_weight)
## make column names in lower case and display year 2009 recs weight file
df_2009_weight = df_2009_weight.rename(columns=str.lower)
df_2009_weight

# ## Variables
# Requirement:
# Using the codebooks for the assoicated data files, determine what variables you will need to answer the following question. Be sure to include variables like the unit id and sample weights.

# Question: Estimate the average number of heating and cooling degree days for residences in each Census region for both 2009 and 2015.

## There are five variables we are interested related to this question:
## 'DOEID', 'REGIONC', 'HDD65', 'CDD65', 'NWEIGHT'
## Then, we create two new dataframes to store these variables for year 2009 and 2015
df_2009_new = df_2009[['doeid', 'regionc', 'hdd65', 'cdd65', 'nweight']]
df_2015_new = df_2015[['doeid', 'regionc', 'hdd65', 'cdd65', 'nweight']]

# ## Weights and Replicate Weights

# 1. Find a link explaining how to use the replicate weights:
# https://www.eia.gov/consumption/residential/methodology/2009/pdf/using-microdata-022613.pdf

# 2. Briefly explain how the replicate weights are used to estimate standard errors for weighted point estimates:

# In your explanation please retype the key equation making sure to document what each variable in the equation is. Don’t forget to include the Fay coefficient and its value(s) for the specific replicate weights included with the survey data.

# First, this method roots in Fay’s method of the balanced repeated replication (BRR) technique. This method uses replicate weights to repeatedly estimate the statistic of interest and calculate the differences between these estimates and the full-sample estimate.</br>

# The variance of $\hat\theta$ is estimated by: 

# $$\hat{V}(\tilde{\theta})={\frac{1}{R(1-\epsilon)^2}}\displaystyle \sum^{R}_{r=1}{(\hat\theta_r - \hat\theta)^2}$$

# - $\theta$ is a population parameter of interest
# - $\hat\theta$ is the estimate from the full sample for $\theta$
# - $\hat\theta_r$ is the estimate from the r-th replicate subsample by using replicate weights
# - $\epsilon$ is the Fay coefficient, $0≤\epsilon<1$.

# For the 2009 RECS, R=244(the number of replicate subsamples) and $\epsilon=.5$.
#
# For the 2015 RECS, R=96(the number of replicate subsamples) and $\epsilon=.5$.
#
# The formula for calculating the esitimated standard error is:
# $$\sqrt{\hat{V}(\tilde{\theta})}$$
