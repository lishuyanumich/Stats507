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

# # Question 1 - NHANES Dentition [50 points]
# In this question you will use the NHANES dentition and demographics data from problem sets 2 and 4.

# a. [30 points] Pick a single tooth (OHXxxTC) and model the probability that a permanent tooth is present (look up the corresponding statuses) as a function of age using logistic regression. For simplicity, assume the data are iid and ignore the survey weights and design. Use a B-Spline basis to allow the probability to vary smoothly with age. Perform model selection using AIC or another method to choose the location of knots and the order of the basis (or just use degree=3 (aka order) and focus on knots).
#
# Control for other demographics included in the data as warranted. You may also select these by minimizing AIC or you may choose to include some demographics regardless of whether they improve model fit. Describe your model building decisions and/or selection process and the series of models fit.
#
# Update October 27: When placing knots, be careful not to place knots at ages below (or equal to) the minimum age at which the tooth you are modeling is present in the data. Doing so will lead to an issue known as perfect separation and make your model non-identifiable. To make the assignment easier you may (but are not required to) limit the analyses to those age 12 and older and use no knots below age 14.

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf
from scipy import stats
from IPython.core.display import display, HTML
import feather
import matplotlib.pyplot as plt
from os.path import exists

filepath = "/Users/ShuyanLi/desktop/Umich_lsy/STATS507/HW5"

# read data files
if exists((filepath+"/demo.feather") and (filepath+"/ohx.feather")):
    demo = pd.read_feather(filepath+"/demo.feather")
    ohx = pd.read_feather(filepath+"/ohx.feather")
else:
    print("There is no such file!")
demo

for i in range(1, 33):
    colname = 'tc_' + str(i).zfill(2)
    ohx[colname] = ohx.apply(lambda x: 1 if x[colname] == "Permanent tooth present" else 0, axis=1)
tc_vars = ['tc_' + str(i).zfill(2) for i in range (1,33)]

# Substract id and age from demo and merge ohx and age data to get age and tooth situation.

# merge ohx and age data to get age and tooth situation
df_ohx = pd.merge(ohx, demo, on = ["id"])

# Compute the marginal proportion by age.

p_hat = df_ohx.groupby('age')[tc_vars].mean()
p_hat

# We can abandon age<12
age_12_older = df_ohx.groupby('age')[tc_vars].mean().dropna().reset_index()
age_12_older = age_12_older[age_12_older['age']>11]
age_12_older

# For each tooth, find the first age where the marginal proportion is greater than 0 and the age at which the proportions peak.

# +
# find the first age where the marginal proportion is greater than 0 
min_age = {}
peak_age = {}
# subtract max so peak age is associated with zero
p_max = p_hat.transform(lambda x: x - np.max(x))

for y in tc_vars:
    #minimum age at which permanent tooth appears
    age = (
        p_hat[[y]]
        .reset_index()
        .query(y + ' > 0')
        .iloc[0, 0]
    )
    min_age.update({y: age})
    
    # age at which present of permanent tooth peaks
    age = (
        p_max[[y]]
        .reset_index()
        .query(y + ' == 0')
        .iloc[0, 0]
    )
    peak_age.update({y:age})
# -

# Here we divide the mouth into quadrants and assign a name to each tooth using the Universal Numbering System. This will make it easier to visualize teeth in a coherent way.

# tooth names and mouth quadrant:
position = (
    list(range(1, 9)) +
    list(reversed(range(9, 17))) +
    list(range(17, 25)) +
    list(reversed(range(25, 33)))
)
tooth_names = (
    '3rd Molar', '2nd Molar', '1st Molar',
    '2nd biscuspid', '1st biscuspid', 'cuspid',
    'lateral incisor', 'central incisor')
areas = ('upper right', 'upper left', 'lower left', 'lower right')

# visualize the marginal proportions:(include all ages)
fig, ax = plt.subplots(nrows=8, ncols=4, sharex=True, sharey=True)
fig.set_size_inches(16,24)
for i in range(32):
    r = (position[i] - 1) % 8
    c = i // 8
    (p_hat[tc_vars[i]]
    .plot
    .line(ax=ax[r, c])
    )
    if r == 0:
        ax[r, c].set_title(areas[c])
    if c == 0:
        ax[r, c].set_ylabel(tooth_names[r])

# Now we can do some model building.

# If we do not frop data that age are less than 12. We find that the values of AIC are very high.

# +
# Store all AIC into the dictionary
ori_AIC = {}
# the first model is special
# we set df = 4
print(tc_vars[0])
mod0 = smf.logit('tc_01 ~ bs(age, df=4, degree=3)', data=df_ohx)
res0 = mod0.fit()
res0.summary()
# fit age into the model and get the prediction
df_ohx['tc_01_hat'] = mod0.predict(params=res0.params)
(df_ohx
 .groupby('age')[['tc_01', 'tc_01_hat']]
 .mean()
 .plot
 .line()
)
ori_AIC[0] = res0.aic
print((res0.aic, res0.df_model))

# for tc_02 to tc_15, we set df = 6
for i in range(1, 15):
    print(tc_vars[i])
    y = tc_vars[i]
    y_hat = y + '_hat'
    mod = smf.logit('%s ~ bs(age, df=6, degree=3)'%y, data=df_ohx)
    res = mod.fit()
    res.summary()
    # fit age into the model and get the prediction
    df_ohx[y_hat] = mod.predict(params=res.params)
    (df_ohx
     .groupby('age')[['%s' %y, '%s' %y_hat]]
     .mean()
     .plot
     .line()
    )
    ori_AIC[i] = res.aic
    print((res.aic, res.df_model))
    
# for tc_16 to tc_32, we set df = 4
for i in range(15, 32):
    print(tc_vars[i])
    y = tc_vars[i]
    y_hat = y + '_hat'
    mod = smf.logit('%s ~ bs(age, df=4, degree=3)'%y, data=df_ohx)
    res = mod.fit()
    res.summary()
    # fit age into the model and get the prediction
    df_ohx[y_hat] = mod.predict(params=res.params)
    (df_ohx
     .groupby('age')[['%s' %y, '%s' %y_hat]]
     .mean()
     .plot
     .line()
    )
    ori_AIC[i] = res.aic
    print((res.aic, res.df_model))
# -

# Then we use the data that we have already dropped age under 12. We tested two methods: use parameter "df" or "knots".

# Let 1st tooth as an example
# Try different df:
for i in range (4, 10):
    print("df = %d" %i)
    mod1 = smf.logit(
        'tc_01 ~ bs(age, df = %d , degree=3)' %i ,
        data = age_12_older
    )
    res1 = mod1.fit()
    age_12_older['tc_01_hat'] = mod1.predict(params=res1.params)
    (age_12_older
     .groupby('age')[['tc_01', 'tc_01_hat']]
     .mean()
     .plot
     .line()
    )
    print((res1.aic, res1.df_model))

# +
# Try different knots:
print("knots = (15, 45, 60)")
mod1 = smf.logit(
    'tc_01 ~ bs(age, knots = (15, 45, 60), degree=3)',
    data = age_12_older
)
res1 = mod1.fit()
age_12_older['tc_01_hat'] = mod1.predict(params=res1.params)
(age_12_older
 .groupby('age')[['tc_01', 'tc_01_hat']]
 .mean()
 .plot
 .line()
)
print((res1.aic, res1.df_model))

print(" ")
print("knots = (15, 35, 45, 60)")
mod1 = smf.logit(
    'tc_01 ~ bs(age, knots = (15, 35, 45, 60), degree=3)',
    data = age_12_older
)
res1 = mod1.fit()
age_12_older['tc_01_hat'] = mod1.predict(params=res1.params)
(age_12_older
 .groupby('age')[['tc_01', 'tc_01_hat']]
 .mean()
 .plot
 .line()
)
print((res1.aic, res1.df_model))

print(" ")
print("knots = (15, 25, 35, 45, 60)")
mod1 = smf.logit(
    'tc_01 ~ bs(age, knots = (15, 25, 35, 45, 60), degree=3)',
    data = age_12_older
)
res1 = mod1.fit()
age_12_older['tc_01_hat'] = mod1.predict(params=res1.params)
(age_12_older
 .groupby('age')[['tc_01', 'tc_01_hat']]
 .mean()
 .plot
 .line()
)
print((res1.aic, res1.df_model))

print(" ")
print("knots = (15, 25, 35, 45, 55, 65)")
mod1 = smf.logit(
    'tc_01 ~ bs(age, knots = (15, 25, 35, 45, 55, 65), degree=3)',
    data = age_12_older
)
res1 = mod1.fit()
age_12_older['tc_01_hat'] = mod1.predict(params=res1.params)
(age_12_older
 .groupby('age')[['tc_01', 'tc_01_hat']]
 .mean()
 .plot
 .line()
)
print((res1.aic, res1.df_model))
# -

# According to the value of AIC and the plots, we will adopt df = 7 and knots = (15, 25, 35, 45, 60).

df_AIC = {}
# for tc_01 to tc_32, we set df = 7
for i in range(0, 32):
    print(tc_vars[i])
    y = tc_vars[i]
    y_hat = y + '_hat'
    mod = smf.logit('%s ~ bs(age, df = 7, degree=3)'%y, data=age_12_older)
    res = mod.fit()
    res.summary()
    # fit age into the model and get the prediction
    age_12_older[y_hat] = mod.predict(params=res.params)
    (age_12_older
     .groupby('age')[['%s' %y, '%s' %y_hat]]
     .mean()
     .plot
     .line()
    )
    df_AIC[i] = res.aic
    print((res.aic, res.df_model))

knots_AIC = {}
# for tc_01 to tc_32, we set knots=(15, 25, 35, 45, 60)
for i in range(0, 32):
    print(tc_vars[i])
    y = tc_vars[i]
    y_hat = y + '_hat'
    mod = smf.logit('%s ~ bs(age, knots=(15, 25, 35, 45, 60), degree=3)'%y, data=age_12_older)
    res = mod.fit()
    res.summary()
    # fit age into the model and get the prediction
    age_12_older[y_hat] = mod.predict(params=res.params)
    (age_12_older
     .groupby('age')[['%s' %y, '%s' %y_hat]]
     .mean()
     .plot
     .line()
    )
    knots_AIC[i] = res.aic
    print((res.aic, res.df_model))

# Then we compare AIC under the two methods and find that there is little difference between the two methods. Knots method is much more comlicated due to adjusting parameter knots. Basically, we find that it is easier to use df as the parameter than knots. So I used df as the parameter for logistic regression models.

df = pd.concat([pd.DataFrame([ori_AIC]).transpose(), pd.DataFrame([df_AIC]).transpose(), pd.DataFrame([knots_AIC]).transpose()],axis=1)
df.columns = ["ori_AIC", "df_AIC", "knots_AIC"]
df

# b. [10 points] Fit the best model you find in part a to all other teeth in the data and create columns in your DataFrame for the fitted values.
#
# Update October 27: Leave the demographics alone, but if you are not restricting to those 12 and older you may need to modify the locations of the knots to make the models identifiable.

# We have fitted age and get the prediction in for loops above. "tc_xx_hat" is the column for the fitted values.

# c. [10 points] Create a visualization showing how the predicted probability that a permanent tooth is present varies with age for each tooth.

# for tc_01 to tc_32, we set df = 7
for i in range(0, 32):
    print(tc_vars[i])
    y = tc_vars[i]
    y_hat = y + '_hat'
    mod = smf.logit('%s ~ bs(age, df = 7, degree=3)'%y, data=age_12_older)
    res = mod.fit()
    res.summary()
    # fit age into the model and get the prediction
    age_12_older[y_hat] = mod.predict(params=res.params)
    (age_12_older
     .groupby('age')[['%s' %y, '%s' %y_hat]]
     .mean()
     .plot
     .line()
    )
    df_AIC[i] = res.aic
    print((res.aic, res.df_model))

age_12_older

# # Question 2 - Hosmer-Lemeshow Calibration Plot [30 points]

# In this question you will construct a plot often associated with the Hosmer-Lemeshow goodness-of-fit test. The plot is often used to assess the calibration of a generalized linear models across the range of predicted values. Specifically, it is used to assess if the expected and observed means are approximately equal across the range of the expected mean.

# Use the tooth you selected in question 1 part a for this question.
#
# 1. Split the data into deciles based on the fitted (aka predicted) probabilities your model assigns to each subject’s tooth. The 10 groups you create using deciles should be approximately equal in size.
#
# 2. Within each decile, compute the observed proportion of cases with a permanent tooth present and the expected proportion found by averaging the probabilities.
#
# 3. Create a scatter plot comparing the observed and expected probabilities within each decile and add a line through the origin with slope 1 as a guide. Your model is considered well-calibrated if the points in this plot fall approximately on this line.
#
# 4. Briefly comment on how-well calibrated your model is (or is not).

# First, I choose the 1st tooth data. Split the "tc_01_hat" column(fitted probabilities) into deciles.

fitted_01 = age_12_older['tc_01_hat'].values
np.percentile(fitted_01, np.arange(0, 100, 10))

# Then, we place values into deciles. Decile group named from 0 to 9.

#calculate decile of each value in data frame
age_12_older['Decile_01'] = pd.qcut(age_12_older['tc_01_hat'], 10, labels=False)

# Secondly, calculate the observed proportion of cases with a permanent tooth present and the predicted proportion found by averaging the probabilities within each decile.

col_tc_01 = ['tc_01', 'tc_01_hat']
df_tc_01 = age_12_older.groupby('Decile_01')[col_tc_01].mean().reset_index()
df_tc_01

# Thirdly, create a scatter plot. Let observed probabilities as x and let expected probabilities as y.

# +
x0 = np.linspace(-0.5, 1, 10)
y0 = x0
x = df_tc_01['tc_01'].values
y = df_tc_01['tc_01_hat'].values

plt.figure(figsize = (8, 8))
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.xlabel('observed probabilities')
plt.ylabel('expected probabilities')
plt.plot(x0, y0)
plt.scatter(x, y, color='red', s=10)
# -

# From the above scatter plot, we can see all the scattered points in this plot fall approximately on this line, thus the model can be considered well-calibrated.


