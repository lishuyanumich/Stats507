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

# # Question 1 - Data Preparation [20 points]

import pandas as pd
import numpy as np
from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib as mpl
## To check if the file exist in local, we should import this
from os.path import exists
import pickle

## get data from website link
base_url = "https://www.eia.gov/consumption/residential/data/"

## get link from the website
url_2015 = base_url + "2015/csv/recs2015_public_v4.csv"
## Use pandas to read data directly from the URL
df_2015 = pd.read_csv(url_2015)
## make column names in lower case and display year 2015 recs file
df_2015 = df_2015.rename(columns=str.lower)

## get link from the website
url_2009 = base_url + "2009/csv/recs2009_public.csv"
## Use pandas to read data directly from the URL
df_2009 = pd.read_csv(url_2009)
## make column names in lower case and display year 2009 recs file
df_2009 = df_2009.rename(columns=str.lower)

## get link from the website
url_2009_weight = base_url + "2009/csv/recs2009_public_repweights.csv"
## Use pandas to read data directly from the URL
df_2009_weight = pd.read_csv(url_2009_weight)
## make column names in lower case and display year 2009 recs weight file
df_2009_weight = df_2009_weight.rename(columns=str.lower)


# Perform the requirement in question 1: 
#
# Structure your code to use `exists` from the (built-in) `sys` module so the data are only downloaded when not already available locally.

def getdata(file_path, file_name, download_to_local):
    """
    Function getdata tests the input file is available locally and return data we want.

    Parameters
    ----------
    file_path: string
               It is a local file path.
    file_name: string
               It is the name of the file we want to get access to.
    download_to_local: boolean
               True if we want to download the file. False if we do not want download the data.
    

    Returns
    -------
    A dataframe that is the same as file_name.

    """
    
    if (file_name == "df_2009_weight"):
        if exists(file_path+'df_2009_weight.pkl'):
            print("The file is already in your local path!")
            with open(file_path+'df_2009_weight.pkl', 'rb') as f:
                data = pickle.load(f)
            return(data)
        else:
            print("The file is downloading to your preferred file path...")
            ## get link from the website
            url_2009_weight = base_url + "2009/csv/recs2009_public_repweights.csv"
            ## Use pandas to read data directly from the URL
            df_2009_weight = pd.read_csv(url_2009_weight)
            ## make column names in lower case and display year 2009 recs weight file
            df_2009_weight = df_2009_weight.rename(columns=str.lower)
            if (download_to_local == True):
                ## download 'df_2009_weight' to local
                df_2009_weight.to_pickle(file_path+'df_2009_weight.pkl')
            return df_2009_weight
    elif (file_name == "df_2009"):
        if exists(file_path+'df_2009.csv'):
            print("The file is already in your local path!")
            with open(file_path+'df_2009.pkl', 'rb') as f:
                data = pickle.load(f)
            return(data)
        else:
            print("The file is downloading to your preferred file path...")
            ## get link from the website
            url_2009 = base_url + "2009/csv/recs2009_public.csv"
            ## Use pandas to read data directly from the URL
            df_2009 = pd.read_csv(url_2009)
            ## make column names in lower case and display year 2009 recs file
            df_2009 = df_2009.rename(columns=str.lower)
            if (download_to_local == True):
                ## download 'df_2009' to local
                df_2009.to_pickle(file_path+'df_2009_weight.pkl')
            return df_2009_weight
    elif (file_name == "df_2015"):
        if exists(file_path+'df_2015.csv'):
            print("The file is already in your local path!")
            with open(file_path+'df_2015.pkl', 'rb') as f:
                data = pickle.load(f)
            return(data)
        else:
            print("The file is downloading to your preferred file path...")
            ## get link from the website
            url_2015 = base_url + "2015/csv/recs2015_public.csv"
            ## Use pandas to read data directly from the URL
            df_2015 = pd.read_csv(url_2015)
            ## make column names in lower case and display year 2015 recs file
            df_2015 = df_2015.rename(columns=str.lower)
            if (download_to_local == True):
                ## download 'df_2015' to local
                df_2015.to_pickle(file_path+'df_2009_weight.pkl')
            return df_2015
    else:
        print("There is no such file!!!")


## Demo of how to use Function getdata(file_path, file_name, download_to_local)
getdata(file_path = "/Users/ShuyanLi/Desktop/", file_name = "df_2009_weight", download_to_local = True)

# ## part a)

# Separately for 2009 and 2015, construct datasets containing just the minimal necessary variables identified in the warmup, excluding the replicate weights. Choose an appropriate format for each of the remaining columns, particularly by creating categorical types where appropriate.

## There are five variables we are interested related to this question:
## 'DOEID', 'REGIONC', 'HDD65', 'CDD65', 'NWEIGHT'
## Then, we create two new dataframes to store these variables for year 2009 and 2015
df_2009_new = df_2009[['doeid', 'regionc', 'hdd65', 'cdd65', 'nweight']]
df_2015_new = df_2015[['doeid', 'regionc', 'hdd65', 'cdd65', 'nweight']]

df_2009_new.dtypes

# We need to change "REGIONC" column type into "categorical". The other column format is appropriate.

df_2009_new['regionc'] = pd.Categorical(df_2009_new['regionc'].replace({
    1:"Northeast Census Region",
    2:"Midwest Census Region",
    3:"South Census Region",
    4:"West Census Region"
}))
df_2015_new['regionc'] = pd.Categorical(df_2015_new['regionc'].replace({
    1:"Northeast Census Region",
    2:"Midwest Census Region",
    3:"South Census Region",
    4:"West Census Region"
}))

## display year 2009 dataframe only containing variables we are interested in
df_2009_new.dtypes

## display year 2015 dataframe only containing variables we are interested in
df_2015_new.dtypes

# ## part b)

# Separatley for 2009 and 2015, construct datasets containing just the unique case ids and the replicate weights (not the primary final weight) in a “long” format with one weight and residence per row.

# For 2009:

## create the brr_weight2009_list containing "DOEID" and all "brr_weight" names
brr_weight2009_list = ["doeid"]
## create the weight_number2009_list containing numbers from 1 to 244
weight_number2009_list = []
for i in range(1,245):
    weight_number2009_list.append(i)
    brr_weight2009_list.append("brr_weight_"+str(i))

brr_weight2009 = df_2009_weight[brr_weight2009_list]
brr_weight2009_long = brr_weight2009.set_index(['doeid'])
brr_weight2009_long.columns = [len(weight_number2009_list)*['2009_rep_weights'], 1*weight_number2009_list]
brr_weight2009_long.columns.names = (None, 'brr_weight_number')
brr_weight2009_long = brr_weight2009_long.stack()
brr_weight2009_long.reset_index(inplace=True)
brr_weight2009_long = brr_weight2009_long[['doeid', '2009_rep_weights', 'brr_weight_number']]
new_col = ['id', 'replicated weights','No. brr weight']
brr_weight2009_long.columns = new_col
## display datasets containing just the unique case ids and the replicate weights of year 2009
brr_weight2009_long

# For 2015:

## create the brr_weight2015_list containing "DOEID" and all "BRRWT" names
brr_weight2015_list = ["doeid"]
## create the weight_number2015_list containing numbers from 1 to 96
weight_number2015_list = []
## search the 2015 rec file, we find 96 brr_weight data
for i in range(1,97):
    weight_number2015_list.append(i)
    brr_weight2015_list.append("brrwt"+str(i))

brr_weight2015 = df_2015[brr_weight2015_list]
brr_weight2015_long = brr_weight2015.set_index(['doeid'])
brr_weight2015_long.columns = [len(weight_number2015_list)*['2015_rep_weights'], 1*weight_number2015_list]
brr_weight2015_long.columns.names = (None, 'brr_weight_number')
brr_weight2015_long = brr_weight2015_long.stack()
brr_weight2015_long.reset_index(inplace=True)
brr_weight2015_long = brr_weight2015_long[['doeid', '2015_rep_weights', 'brr_weight_number']]
new_col = ['id', 'replicated weights','No. brr weight']
brr_weight2015_long.columns = new_col
## display datasets containing just the unique case ids and the replicate weights of year 2015
brr_weight2015_long

# # Question 2 - Construct and report the estimates [45 points]

# ## part a)

## From recs data
R_2009 = 244
R_2015 = 96
## epsilon is Fay coefficient, We set it as 0.5
epsilon = 0.5
## alpha is confidence level
## We want to get 95% confidence interval, so we set it as 1.96
alpha = 1.96

# Estimate the average number of heating and cooling degree days for residences in each Census region for both 2009 and 2015. You should construct both point estimates (using the weights) and 95% confidece intervals (using standard errors estiamted with the repliacte weights). Report your results in a nicely formatted table.
#
# For this question, you should use pandas DataFrame methods wherever possible. Do not use a module specifically supporting survey weighting.

# ### First, we deal with year 2009 data:
# 1) In this part we will get the average number of heating and cooling degree days for residences in each Census region for 2009. 
#
# 2) Then we will construct both point estimates (using the weights) and 95% confidece intervals and store them in DataFrame `merged_2009` .
#
# 3) At last we report the results in a nicely formatted table.

# +
## import df_2009_new including variables: doeid, regionc, hdd65, cdd65, nweight
## df_2009_new is created in Question1 (a)
df_2009_nwt = df_2009_new

## We multiple every hdd and cdd with its nweight
df_2009_nwt['nwt_hdd65'] = df_2009_nwt['hdd65'] * df_2009_nwt['nweight']
df_2009_nwt['nwt_cdd65'] = df_2009_nwt['cdd65'] * df_2009_nwt['nweight']

# +
## calculate year 2009's theta hat after grouped by 'regionc'
grouped_2009 = df_2009_nwt.groupby('regionc').sum().reset_index()

## calculate average number of heating and cooling degree days in each Census region
grouped_2009['avg_hdd'] = (grouped_2009['nwt_hdd65']/grouped_2009['nweight']).round(2)
grouped_2009['avg_cdd'] = (grouped_2009['nwt_cdd65']/grouped_2009['nweight']).round(2)

## output the average number of heating and cooling degree days in each Census region
#grouped_2009[['regionc', 'avg_hdd', 'avg_cdd']]
# -

# For 2009 HDD $\hat\theta_R$:

# +
## substract year 2009's all brr_weight and put them into dataframe weight_2009
weight_2009 = df_2009_weight.loc[:, "brr_weight_1":"brr_weight_244"]
## generate a dataframe to store year 2009's brr_weight with their regionc
weight_2009_region = pd.concat([df_2009_new['regionc'], weight_2009], axis=1)
weight_2009_region = weight_2009_region.groupby('regionc').sum().reset_index()
## exclude column 'regionc'
weight_2009_region = weight_2009_region.loc[:, "brr_weight_1":"brr_weight_244"]
## multiply the two dataframe, and get every brr_weight * hdd
hdd_bw_product_2009 = weight_2009.multiply(df_2009_new['hdd65'].values, axis='rows')
## Add column 'regionc' to hdd_bw_product_2009
df_region_brr_2009 = pd.concat([df_2009_new[['regionc']], hdd_bw_product_2009], axis=1)
## groupby 'regionc'
df1 = df_region_brr_2009.groupby('regionc').sum()
df2 = df1.reset_index()
## subtract all brr_weighted data(thetaR_hat)
df3 = df2.loc[:, "brr_weight_1":"brr_weight_244"]
df4 = df3/weight_2009_region
## thetaR_hat - hdd_theta_hat
df5 = df4.sub(grouped_2009['avg_hdd'], axis=0)
print(df5)
df6 = df5**2

df6['row_sum'] = df6.apply(lambda x: x.sum(), axis=1)
print(df6)
df6['hdd_region_sd'] = ((df6['row_sum']*(1/(R_2009*(1-epsilon)**2)))**(1/2)).round(2)
df6['hdd_lwr'] = (grouped_2009['avg_hdd'] - alpha*df6['hdd_region_sd']).round(2)
df6['hdd_upr'] = (grouped_2009['avg_hdd'] + alpha*df6['hdd_region_sd']).round(2)
## generate a column to store CI
df6['95% CI of heating days'] ="(" + (df6['hdd_lwr']).astype(str) +","+ (df6['hdd_upr']).astype(str) + ")"
## df6 stores hdd_sd
hdd_region_CI = pd.concat([grouped_2009[['avg_hdd']], df6[['95% CI of heating days', 'hdd_region_sd']]], axis=1)
# -

# The same for 2009 CDD $\hat\theta_R$:

## substract year 2009's all brr_weight and put them into dataframe weight_2009
weight_2009 = df_2009_weight.loc[:, "brr_weight_1":"brr_weight_244"]
## generate a dataframe to store year 2009's brr_weight with their regionc
weight_2009_region = pd.concat([df_2009_new['regionc'], weight_2009], axis=1)
weight_2009_region = weight_2009_region.groupby('regionc').sum().reset_index()
## exclude column 'regionc'
weight_2009_region = weight_2009_region.loc[:, "brr_weight_1":"brr_weight_244"]
## multiply the two dataframe, and get every brr_weight * cdd
cdd_bw_product_2009 = weight_2009.multiply(df_2009_new['cdd65'].values, axis='rows')
## Add column 'regionc' to cdd_bw_product_2009
df_region_brr_2009 = pd.concat([df_2009_new[['regionc']], cdd_bw_product_2009], axis=1)
## groupby 'regionc'
df1 = df_region_brr_2009.groupby('regionc').sum()
df2 = df1.reset_index()
## subtract all brr_weighted data(thetaR_hat)
df3 = df2.loc[:, "brr_weight_1":"brr_weight_244"]
df4 = df3/weight_2009_region
## thetaR_hat - cdd_theta_hat
df5 = df4.sub(grouped_2009['avg_cdd'], axis=0)
df6 = df5**2
df6['row_sum'] = df6.apply(lambda x: x.sum(), axis=1)
df6['cdd_region_sd'] = ((df6['row_sum']*(1/(R_2009*(1-epsilon)**2)))**(1/2)).round(2)
df6['cdd_lwr'] = (grouped_2009['avg_cdd'] - alpha*df6['cdd_region_sd']).round(2)
df6['cdd_upr'] = (grouped_2009['avg_cdd'] + alpha*df6['cdd_region_sd']).round(2)
## generate a column to store CI
df6['95% CI of cooling days'] ="(" + (df6['cdd_lwr']).astype(str) +","+ (df6['cdd_upr']).astype(str) + ")"
## df6 stores cdd_sd
cdd_region_CI = pd.concat([grouped_2009[['avg_cdd']], df6[['95% CI of cooling days', 'cdd_region_sd']]], axis=1)

## merge 'regionc', year 2009's hdd and cdd
merged_2009 = pd.concat([grouped_2009[['regionc']], hdd_region_CI, cdd_region_CI], axis=1)
newname_merged_2009 = merged_2009[['regionc', 'avg_hdd', '95% CI of heating days', 'avg_cdd', '95% CI of cooling days']]
new_col_2009 = ['regionc', 'average number of heating degree days', '95% CI of heating days', 
                'average number of cooling degree days', '95% CI of cooling days']
newname_merged_2009.columns = new_col_2009

# +
# construct a table, include a caption: ---------------------------------------
cap = """
<b> Table 1.</b> <em> Esitimate average number of heating and cooling degree days in 2009</em>
"""
t1 = newname_merged_2009.to_html(index=False)
t1 = t1.rsplit('\n')
t1.insert(1, cap)
tab1 = ''
for i, line in enumerate(t1):
    tab1 += line
    if i < (len(t1) - 1):
        tab1 += '\n'

display(HTML(tab1))
# -

# ### Second, we deal with year 2015 data:
# 1) In this part we will get the average number of heating and cooling degree days for residences in each Census region for 2015. 
#
# 2) Then we will construct both point estimates (using the weights) and 95% confidece intervals and store them in DataFrame `merged_2015` .
#
# 3) At last we report the results in a nicely formatted table.

# +
## import df_2015_new including variables: doeid, regionc, hdd65, cdd65, nweight
## df_2015_new is created in Question1 (a)
df_2015_nwt = df_2015_new

## We multiple every hdd and cdd with its nweight
df_2015_nwt['nwt_hdd65'] = df_2015_nwt['hdd65'] * df_2015_nwt['nweight']
df_2015_nwt['nwt_cdd65'] = df_2015_nwt['cdd65'] * df_2015_nwt['nweight']

# +
## calculate year 2015's theta hat after grouped by 'regionc'
grouped_2015 = df_2015_nwt.groupby('regionc').sum().reset_index()

## calculate average number of heating and cooling degree days in each Census region
grouped_2015['avg_hdd'] = (grouped_2015['nwt_hdd65']/grouped_2015['nweight']).round(2)
grouped_2015['avg_cdd'] = (grouped_2015['nwt_cdd65']/grouped_2015['nweight']).round(2)

## output the average number of heating and cooling degree days in each Census region
#grouped_2015[['regionc', 'avg_hdd', 'avg_cdd']]
# -

# For 2015 HDD $\hat\theta_R$:

## substract year 2015's all brr_weight and put them into dataframe weight_2015
weight_2015 = df_2015.loc[:, "brrwt1":"brrwt96"]
## generate a dataframe to store year 2015's brr_weight with their regionc
weight_2015_region = pd.concat([df_2015_new['regionc'], weight_2015], axis=1)
weight_2015_region = weight_2015_region.groupby('regionc').sum().reset_index()
## exclude column 'regionc'
weight_2015_region = weight_2015_region.loc[:, "brrwt1":"brrwt96"]
## Add column 'regionc' to hdd_bw_product_2015
hdd_bw_product_2015 = weight_2015.multiply(df_2015_new['hdd65'].values, axis='rows')
## Add column 'regionc' to hdd_bw_product_2015
df_region_brr_2015 = pd.concat([df_2015_new[['regionc']], hdd_bw_product_2015], axis=1)
## groupby 'regionc'
df1 = df_region_brr_2015.groupby('regionc').sum()
df2 = df1.reset_index()
## subtract all brr_weighted data(thetaR_hat)
df3 = df2.loc[:, "brrwt1":"brrwt96"]
df4 = df3/weight_2015_region
## thetaR_hat - hdd_theta_hat
df5 = df4.sub(grouped_2015['avg_hdd'], axis=0)
df6 = df5**2
df6['row_sum'] = df6.apply(lambda x: x.sum(), axis=1)
df6['hdd_region_sd'] = ((df6['row_sum']*(1/(R_2015*(1-epsilon)**2)))**(1/2)).round(2)
df6['hdd_lwr'] = (grouped_2015['avg_hdd'] - alpha*df6['hdd_region_sd']).round(2)
df6['hdd_upr'] = (grouped_2015['avg_hdd'] + alpha*df6['hdd_region_sd']).round(2)
##generate a column to store CI
df6['95% CI of heating days'] ="(" + (df6['hdd_lwr']).astype(str) +","+ (df6['hdd_upr']).astype(str) + ")"
## df6 stores hdd_sd
hdd_region_CI = pd.concat([grouped_2015[['avg_hdd']], df6[['95% CI of heating days', 'hdd_region_sd']]], axis=1)

# The same for 2015 CDD $\hat\theta_R$:

## substract year 2015's all brr_weight and put them into dataframe weight_2015
weight_2015 = df_2015.loc[:, "brrwt1":"brrwt96"]
## generate a dataframe to store year 2015's brr_weight with their regionc
weight_2015_region = pd.concat([df_2015_new['regionc'], weight_2015], axis=1)
weight_2015_region = weight_2015_region.groupby('regionc').sum().reset_index()
## exclude column 'regionc'
weight_2015_region = weight_2015_region.loc[:, "brrwt1":"brrwt96"]
## multiply the two dataframe, and get every brr_weight * cdd
cdd_bw_product_2015 = weight_2015.multiply(df_2015_new['cdd65'].values, axis='rows')
## Add column 'regionc' to cdd_bw_product_2015
df_region_brr_2015 = pd.concat([df_2015_new[['regionc']], cdd_bw_product_2015], axis=1)
## groupby 'regionc'
df1 = df_region_brr_2015.groupby('regionc').sum()
df2 = df1.reset_index()
## subtract all brr_weighted data(thetaR_hat)
df3 = df2.loc[:, "brrwt1":"brrwt96"]
df4 = df3/weight_2015_region
## thetaR_hat - hdd_theta_hat
df5 = df4.sub(grouped_2015['avg_cdd'], axis=0)
df6 = df5**2
df6['row_sum'] = df6.apply(lambda x: x.sum(), axis=1)
df6['cdd_region_sd'] = ((df6['row_sum']*(1/(R_2015*(1-epsilon)**2)))**(1/2)).round(2)
df6['cdd_lwr'] = (grouped_2015['avg_cdd'] - alpha*df6['cdd_region_sd']).round(2)
df6['cdd_upr'] = (grouped_2015['avg_cdd'] + alpha*df6['cdd_region_sd']).round(2)
##generate a column to store CI
df6['95% CI of cooling days'] ="(" + (df6['cdd_lwr']).astype(str) +","+ (df6['cdd_upr']).astype(str) + ")"
## df6 stores hdd_sd
cdd_region_CI = pd.concat([grouped_2015[['avg_cdd']], df6[['95% CI of cooling days', 'cdd_region_sd']]], axis=1)

##合并地区和2015年的hdd和cdd
merged_2015 = pd.concat([grouped_2015[['regionc']], hdd_region_CI, cdd_region_CI], axis=1)
newname_merged_2015 = merged_2015[['regionc', 'avg_hdd', '95% CI of heating days', 'avg_cdd', '95% CI of cooling days']]
new_col_2015 = ['regionc', 'average number of heating degree days', '95% CI of heating days', 
                'average number of cooling degree days', '95% CI of cooling days']
newname_merged_2015.columns = new_col_2015

# +
# construct a table, include a caption: ---------------------------------------
cap = """
<b> Table 2.</b> <em> Estimate average number of heating and cooling degree days in 2015</em>
"""
t2 = newname_merged_2015.to_html(index=False)
t2 = t2.rsplit('\n')
t2.insert(1, cap)
tab2 = ''
for i, line in enumerate(t2):
    tab2 += line
    if i < (len(t2) - 1):
        tab2 += '\n'

display(HTML(tab2))
# -

# ## part b)

# Using the estimates and standard errors from part a, estimate the change in heating and cooling degree days between 2009 and 2015 for each Census region. In constructing interval estimates, use the facts that the estimators for each year are independent and that,
# $$var(\hat\theta_0, \hat\theta_1)=var(\hat\theta_0)+var(\hat\theta_1)$$
# when the estimators $\hat\theta_0$ and $\hat\theta_1$ are independent.

change_data = pd.DataFrame()
## calculate delta_hdd and delta_cdd
change_data['change of heating days'] = merged_2015['avg_hdd'] - merged_2009['avg_hdd']
change_data['change of cooling days'] = merged_2015['avg_cdd'] - merged_2009['avg_cdd']
## calculate delta_hdd and delta_cdd standard deriviation
change_data['se_delta_hdd'] = ((merged_2015['hdd_region_sd']**2 + merged_2009['hdd_region_sd']**2)**(1/2)).round(2)
change_data['se_delta_cdd'] = ((merged_2015['cdd_region_sd']**2 + merged_2009['cdd_region_sd']**2)**(1/2)).round(2)
## calculate delta_hdd CI
change_data['delta_hdd_lwr'] = (change_data['change of heating days'] - alpha*change_data['se_delta_hdd']).round(2)
change_data['delta_hdd_upr'] = (change_data['change of heating days'] + alpha*change_data['se_delta_hdd']).round(2)
change_data['95% CI of change of heating days'] = "(" + (change_data['delta_hdd_lwr']).astype(str) + "," + (change_data['delta_hdd_upr']).astype(str) + ")"
## calculate delta_cdd CI
change_data['delta_cdd_lwr'] = (change_data['change of cooling days'] - alpha*change_data['se_delta_cdd']).round(2)
change_data['delta_cdd_upr'] = (change_data['change of cooling days'] + alpha*change_data['se_delta_cdd']).round(2)
change_data['95% CI of change of cooling days'] = "(" + (change_data['delta_cdd_lwr']).astype(str) + "," + (change_data['delta_cdd_upr']).astype(str) + ")"
## Subtract what we are interested in change_data DataFrame
change_result = pd.concat([merged_2015['regionc'],
                           change_data[['change of heating days', '95% CI of change of heating days', 
                                        'change of cooling days', '95% CI of change of cooling days']]], axis=1)

# +
# construct a table, include a caption: ---------------------------------------
cap = """
<b> Table 3.</b> <em>The change in heating and cooling degree days between 2009 and 2015 for each Census region</em>
"""
t3 = change_result.to_html(index=False)
t3 = t3.rsplit('\n')
t3.insert(1, cap)
tab3 = ''
for i, line in enumerate(t3):
    tab3 += line
    if i < (len(t3) - 1):
        tab3 += '\n'

display(HTML(tab3))
# -

# # Question 3 - [20 points]

# Use pandas and/or matplotlib to create visualizations for the results reported as tables in parts a and b of question 2. As with the tables, your figures should be “polished” and professional in appearance, with well-chosen axis and tick labels, English rather than code_speak, etc. Use an adjacent markdown cell to write a caption for each figure.

## import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl

region_series = merged_2009['regionc'].astype(str)
type(region_series)

label_array = np.transpose(merged_2009.values)[0]
labels = list(label_array)

# Draw a bar and errorbar plot about average heating degree days comparing 2009 and 2015.

width = 0.4
fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(10,8))
fig1.tight_layout()
_ = plt.bar(x = np.arange(len(labels)) - width / 2, 
            height = merged_2009['avg_hdd'], 
            color = 'lightblue',
            label ='year 2009',
            width = width, 
            align = 'center',
            yerr = merged_2009['hdd_region_sd'],
            ecolor = 'brown',
            capsize=10
           )
## show the label
_ = plt.legend(loc=1)
_ = plt.bar(x = np.arange(len(labels)) + width / 2, 
            height = merged_2015['avg_hdd'], 
            color = 'pink',
            label = 'year 2015',
            width = width, 
            align = 'center',
            yerr = merged_2015['hdd_region_sd'],
            ecolor = 'brown', 
            capsize=10
            )
## show the label
_ = plt.legend(loc=1)
_ = plt.xticks(np.arange(len(labels)), labels)
_ = ax1.set_xlabel('regionc')
_ = plt.ylabel('Number of average heating degree days')
_ = plt.title('Average heating days in 2009 and 2015')

# We can see that the number of average heating degree days are decreasing.

# Draw a bar and errorbar plot about average cooling degree days comparing 2009 and 2015.

width = 0.4
fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(10,8))
fig1.tight_layout()
_ = plt.bar(x = np.arange(len(labels)) - width / 2, 
            height = merged_2009['avg_cdd'], 
            color = 'tan',
            label ='year 2009',
            width = width, 
            align = 'center',
            yerr = merged_2009['cdd_region_sd'],
            ecolor = 'brown',
            capsize=10
           )
## show the label
_ = plt.legend(loc=1)
_ = plt.bar(x = np.arange(len(labels)) + width / 2, 
            height = merged_2015['avg_cdd'], 
            color = 'green',
            label = 'year 2015',
            width = width, 
            align = 'center',
            yerr = merged_2015['cdd_region_sd'],
            ecolor = 'brown', 
            capsize=10
            )
## show the label
_ = plt.legend(loc=1)
_ = plt.xticks(np.arange(len(labels)), labels)
_ = ax1.set_xlabel('regionc')
_ = plt.ylabel('Number of average cooling degree days')
_ = plt.title('Average cooling days in 2009 and 2015')

# We can see that the number of average cooling degree days are rising.

width = 0.4
fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(10,8))
fig1.tight_layout()
_ = plt.bar(x = np.arange(len(labels)) - width / 2, 
            height = change_data['change of heating days'], 
            color = 'red',
            label ='change of heating days',
            width = width, 
            align = 'center',
            yerr = change_data['se_delta_hdd'],
            ecolor = 'black',
            capsize=10
           )
## show the label
_ = plt.legend(loc=1)
_ = plt.bar(x = np.arange(len(labels)) + width / 2, 
            height = change_data['change of cooling days'], 
            color = 'blue',
            label = 'change of cooling days',
            width = width, 
            align = 'center',
            yerr = change_data['se_delta_cdd'],
            ecolor = 'black', 
            capsize=10
            )
## show the label
_ = plt.legend(loc=1)
_ = plt.xticks(np.arange(len(labels)), labels)
_ = ax1.set_xlabel('regionc')
_ = plt.ylabel('Change of average heating and cooling degree days')
_ = plt.title('Change of average heating and cooling days from 2009 to 2015')

# Growing number of average cooling degree days and decreasing number of average heating degree days illustrate the global warming effect!
