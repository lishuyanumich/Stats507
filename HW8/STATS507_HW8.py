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

# # Question 0 - Slurm [20 points]
# In this question you will write a Slurm script and use Great Lakes to compute the cross-validated MSE for one of your Random Forest or Gradient Boosted Tree models from problem set 7.
#
# Using the superconductivity data and either your best performing Random Forest or model or your best performing Gradient Boosted Tree model from problem set 7, write a Python script to compute the 10-fold cross validation error for your chosen model in parallel using 5 subprocesses. The Python script should be written to run in batch mode.
#
# If you didn’t previously divide data into test, validation, and 10 training folds based on unique materials (see unique_m.csv), redo the data splitting so that materials (rather than rows) are randomized among training folds and the validation and test sets.
#
# Write an associated Slurm shell script to run your multiple process job using 5-6 cores and run the job using the course allocation. Include this Slurm script in your submission.
#
#

import numpy as np
import pandas as pd
from sklearn import metrics
from numpy import mean
from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

filepath = "/home/lishuyan/ondemand/data/sys/myjobs/projects/default/1/STATS507/"
#filepath = "/Users/ShuyanLi/Desktop/Umich_lsy/STATS507/HW8/"
df_data = pd.read_csv(filepath+"train.csv")

# split the cases into three parts
# 80% of the cases for training
# 10% of the cases for validation
# 10% of the cases for testing
train, validate, test = np.split(df_data, [int(.8 * len(df_data)), int(.9 * len(df_data))])

# Use data in train
train_data = train.values
X_train, y_train = train_data[:, :-1], train_data[:, -1]
# Using the hyperparameter selected in the previous section
model_RF = RandomForestRegressor(n_estimators=300, max_depth=10, n_jobs=5)
# train model
model_RF.fit(X_train, y_train)
# define X and y
data = df_data.values
X, y = data[:, :-1], data[:, -1]
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
# evaluate model
CV_MSE = np.mean(absolute(cross_val_score(model_RF, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=5)))
print(CV_MSE)
