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

# # Question 1 - NHANES Table 1 [35 points]

# ## part b) 
# The variable OHDDESTS contains the status of the oral health exam. Merge this variable into the demographics data.
#
# Use the revised demographic data from part a and the oral health data from PS2 to create a clean dataset with the following variables:
#
# - id (from SEQN)
# - gender
# - age
# - under_20 if age < 20
# - college - with two levels:
#     - ‘some college/college graduate’ or
#     - ‘No college/<20’ where the latter category includes everyone under 20 years of age.
# - exam_status (RIDSTATR)
# - ohx_status - (OHDDESTS)//dentition_status	

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy import stats
from IPython.core.display import display, HTML
import feather
from os.path import exists

filepath = "/Users/ShuyanLi/desktop/Umich_lsy/STATS507/HW4"

# read data files
if exists((filepath+"/demo.feather") and (filepath+"/ohx.feather")):
    demo = pd.read_feather(filepath+"/demo.feather")
    ohx = pd.read_feather(filepath+"/ohx.feather")
else:
    print("There is no such file!")
demo

ohx

#extact "id", "dentition_status", "cohort" from ohx dataframe
ohx_mini = ohx[["id", "dentition_status", "cohort"]]
# change "dentition_status" column name into "ohx_status"
ohx_mini.columns = ["id", "ohx_status", "cohort"]

# extract "id", "gender", "age", "exam_status", "education" from demo dataframe
demo_mini = demo[["id", "gender", "age", "exam_status", "education", "cohort"]]
# Add a column "under_20" 
demo_mini["under_20"] = demo_mini.apply(lambda x: "Yes" if x['age']<20 else "No", axis=1)


# +
# Add column "college"
def function(edu, age):
    """
    Define the category of "college": 
    "some college/college graduate" or "No college/<20".

    Parameters
    ----------
    edu : string
        The string representing the education level. 
    age : int
        Values of age. 

    Returns
    -------
    The category of "college": 
    "some college/college graduate" or "No college/<20". 

    """
    if ((edu=="Some college or AA degree" or edu=="College graduate or above") and age>=20):
        return ("some college/college graduate")
    else:
        return ("No college/<20")

demo_mini["college"] = demo_mini.apply(lambda x: function(x["education"], x["age"]), axis=1)
# -

mergedata_b = pd.merge(demo_mini[["id", "gender", "age", "under_20", "college", "exam_status", "cohort"]], ohx_mini, on=["id", "cohort"], how="left")

mergedata_b = mergedata_b[["id", "gender", "age", "under_20", "college", "exam_status", "ohx_status", "cohort"]]
mergedata_b

# ## part c)
# Remove rows from individuals with exam_status != 2 as this form of missingness is already accounted for in the survey weights. Report the number of subjects removed and the number remaining.

data_c = mergedata_b.drop(index=(mergedata_b.loc[(mergedata_b['exam_status']!='Both interviewed and MEC examined')].index))
data_c

# The number of subjects removed = 39156 - 37399 = 1757
#
# The number remaining = 37399

# ## part d)
# Construct a table with ohx (complete / missing) in columns and each of the following variables summarized in rows:
#
# - age
# - under_20
# - gender
# - college
#
# For the rows corresponding to categorical variable in your table, each cell should provide a count (n) and a percent (of the row) as a nicely formatted string. For the continous variable age, report the mean and standard deviation [Mean (SD)] for each cell.
#
# Include a column ‘p-value’ giving a p-value testing for a mean difference in age or an association beween each categorical varaible and missingness. Use a chi-squared test comparing the 2 x 2 tables for each categorical characteristic and OHX exam status and a t-test for the difference in age.
#
# **Hint*: Use scipy.stats for the tests.

## For college
data_college = data_c
# modify "ohx_status" as "complete" or "missing"
data_college["ohx status"] = data_c.apply(lambda x: "complete" if x["ohx_status"]=="Complete" else "missing", axis=1)
ohxstatus_college = data_college.groupby(["college","ohx status"]).agg({"ohx status":"count"})
ohxstatus_count = data_college.groupby("college").agg("count")
college_ohxstatus = round(ohxstatus_college.div(ohxstatus_count, level="college")*100,3)
college_ohxstatus = college_ohxstatus.take([5], axis=1)
data_college_ohx = pd.concat([ohxstatus_college, college_ohxstatus], axis=1)
data_college_ohx.columns = ["count", "ratio"]
data_college_ohx["count and ratio"] = data_college_ohx.apply(lambda x: str(x["count"])+"("+str(x["ratio"])+"%)", axis=1)
res_college_ohx = data_college_ohx.take([2], axis=1)
# Chi-squared test
df = data_college_ohx.take([0], axis=1).values.transpose()
df_college = np.vstack(np.hsplit(df,2))
kf_college = chi2_contingency(df_college)
print('chisq-statistic=%.4f, p-value=%.50f, df=%i expected_frep=%s'%kf_college)
res_college_ohx["p value"] = kf_college[1]

## For gender
data_gender = data_c
# modify "ohx_status" as "complete" or "missing"
data_gender["ohx status"] = data_c.apply(lambda x: "complete" if x["ohx_status"]=="Complete" else "missing", axis=1)
ohxstatus_gender = data_gender.groupby(["gender","ohx status"]).agg({"ohx status":"count"})
ohxstatus_count = data_gender.groupby("gender").agg("count")
gender_ohxstatus = round(ohxstatus_gender.div(ohxstatus_count, level="gender")*100,3)
gender_ohxstatus = gender_ohxstatus.take([5], axis=1)
gender_ohxstatus
data_gender_ohx = pd.concat([ohxstatus_gender, gender_ohxstatus], axis=1)
data_gender_ohx.columns = ["count", "ratio"]
data_gender_ohx["count and ratio"] = data_gender_ohx.apply(lambda x: str(x["count"])+"("+str(x["ratio"])+"%)", axis=1)
res_gender_ohx = data_gender_ohx.take([2], axis=1)
# Chi-squared test
df = data_gender_ohx.take([0], axis=1).values.transpose()
df_gender = np.vstack(np.hsplit(df,2))
kf_gender = chi2_contingency(df_gender)
print('chisq-statistic=%.4f, p-value=%.50f, df=%i expected_frep=%s'%kf_gender)
res_gender_ohx["p value"] = kf_gender[1]

# For under_20
# modify "age" into "Under 20" or "20 or older"
data_d = data_c
data_d["a"] = data_d.apply(lambda x: "Under 20" if x["age"]<20 else "20 or older", axis=1)
data_d = data_d[["id", "gender", "a", "college", "exam_status", "ohx_status", "cohort"]]
data_d = data_d.rename(columns={"a":"under_20"})
# modify "ohx_status" as "complete" or "missing"
data_d["ohx status"] = data_d.apply(lambda x: "complete" if x["ohx_status"]=="Complete" else "missing", axis=1)
ohxstatus_age = data_d.groupby(["under_20","ohx status"]).agg({"ohx status":"count"})
ohxstatus_count = data_d.groupby("under_20").agg("count")
age_ohxstatus = round(ohxstatus_age.div(ohxstatus_count, level="under_20")*100,3)
age_ohxstatus = age_ohxstatus.take([5], axis=1)
data_age_ohx = pd.concat([ohxstatus_age, age_ohxstatus], axis=1)
data_age_ohx.columns = ["count", "ratio"]
data_age_ohx["count and ratio"] = data_age_ohx.apply(lambda x: str(x["count"])+"("+str(x["ratio"])+"%)", axis=1)
res_age_ohx = data_age_ohx.take([2], axis=1)
#chi-squared test
df = data_age_ohx.take([0], axis=1).values.transpose()
df_age = np.vstack(np.hsplit(df,2))
kf_age = chi2_contingency(df_age)
print('chisq-statistic=%.4f, p-value=%.100f, df=%i expected_frep=%s'%kf_age)
res_age_ohx["p value"] = kf_age[1]

data_age_continuous = data_c
# modify "ohx_status" as "complete" or "missing"
data_age_continuous["ohx status"] = data_c.apply(lambda x: "complete" if x["ohx_status"]=="Complete" else "missing", axis=1)
data_age_continuous["AGE"] = "age"
ohxstatus_age_c_mean = data_age_continuous.groupby(["AGE", "ohx status"]).mean()
ohxstatus_age_c_mean = ohxstatus_age_c_mean.take([1], axis=1)
ohxstatus_age_c_mean.columns = ["mean"]
ohxstatus_age_c_sd = data_age_continuous.groupby(["AGE", "ohx status"]).std()
ohxstatus_age_c_sd = ohxstatus_age_c_sd.take([1], axis=1)
ohxstatus_age_c_sd.columns = ["std"]
result_age = pd.concat([ohxstatus_age_c_mean, ohxstatus_age_c_sd], axis=1)
result_age["count and ratio"] = result_age.apply(lambda x: "mean="+str(x["mean"])+", std="+str(x["std"]), axis=1)
result_age = result_age[["count and ratio"]]
result_age

# +
data_age_continuous = data_c
# modify "ohx_status" as "complete" or "missing"
data_age_continuous["ohx status"] = data_c.apply(lambda x: "complete" if x["ohx_status"]=="Complete" else "missing", axis=1)
# Separate age by ohx status(complete and missing)
complete_age = data_age_continuous.loc[data_age_continuous['ohx status']=='complete']
complete_ttest = list(complete_age["age"].values)
missing_age = data_age_continuous.loc[data_age_continuous['ohx status']=='missing']
missing_ttest = list(missing_age["age"].values)

result_age["p value"] = stats.ttest_ind(complete_ttest, missing_ttest).pvalue

result_age
# -

result_d = pd.concat([result_age, res_age_ohx, res_gender_ohx, res_college_ohx], 
                      keys=["age", "under_20", "gender", "college"],
                      axis=0)
result_d

# construct a table, include a caption: ---------------------------------------
cap = """
<b> Table 1.</b> <em> Age, under_20, gender and college with different OHX exam status</em>
For the rows corresponding to categorical variable in the table, each cell should provide a count (n) and a percent (of the row). 
For the continous variable age, report the mean and standard deviation [Mean (SD)] for each cell.
Include a column ‘p-value’ giving a p-value testing for a mean difference in age or an association 
between each categorical varaible and missingness. Use a chi-squared test comparing the 2 x 2 tables 
for each categorical characteristic and OHX exam status and a t-test for the difference in age.
"""
res = pd.DataFrame(result_d)
t1 = res.to_html(index=True, sparsify=False)
t1 = t1.rsplit('\n')
t1.insert(1, cap)
tab1 = ''
for i, line in enumerate(t1):
    tab1 += line
    if i < (len(t1) - 1):
        tab1 += '\n'
display(HTML(tab1))


