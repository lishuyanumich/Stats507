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

# # Question 0 - Data Prep [10 points]
# In this problem set you will train and tune machine learning regression models for a Superconductivty dataset. In this dataset the goal is to predict the critical temperature based on 81 features extracted from a chemical formula of a material. You should use mean squared error (or an equivalent) as the loss function.
#
# In this question you will prepare the data for training and tuning models in the following question. To do so, read the data into Python and create DataFrames or Numpy arrays for the features and dependent regression target (critical temperature).
#
# Then, split the cases into three parts: use 80% of the cases for training, hold 10% for validation and model comparison in question 2, and reserve 10% as a test dataset.

import pandas as pd
import numpy as np
import feather
from os.path import exists
from sklearn import metrics
from numpy import mean
from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from IPython.core.display import display, HTML

# read the data into Python
temp_file = "unique_m.feather"
if exists(temp_file):
    df_data = pd.read_feather(temp_file)
else:
    filepath = "/Users/ShuyanLi/Desktop/Umich_lsy/STATS507/HW7/"
    df_data = pd.read_csv(filepath+"unique_m.csv")
    #save
    df_data.to_feather(temp_file)

filepath = "/Users/ShuyanLi/Desktop/Umich_lsy/STATS507/HW7/"
df_data = pd.read_csv(filepath+"train.csv")
df_data    

# create DataFrames or Numpy arrays for the features 
# and dependent regression target (critical temperature)
column_names = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
       'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
       'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
       'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
       'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
       'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
       'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At',
       'Rn', 'critical_temp']
df_data = pd.DataFrame(df_data, columns=column_names)

# split the cases into three parts
# 80% of the cases for training
# 10% of the cases for validation
# 10% of the cases for testing
train, validate, test = np.split(df_data, [int(.8 * len(df_data)), int(.9 * len(df_data))])

df_data

# # Question 1 - Training and Tuning Models [70 points]
# In this question you should train and tune elastic-net, random forest, and gradient boosted decision tree models using the 80% training sample from question 0. Tune hyper-parameters for each model class using 10-fold cross-validation.
#
# ## part a
# Train a series of elastic net models and choose the mixing parameter l1_ratio and the amount of regularization C using 10-fold cross-validation over a grid (or grids) of values.
#
# Create a figure or table showing the cross-validated MSE at key points in your grid and identity which hyper-parameters minimize this quantity.

# +
# Use data in train
data = train.values
X, y = data[:, :-1], data[:, -1]

# generate MSE_rf to store MSE
MSE_data = []

for i in [1e-2, 1e-1, 0.0, 1.0, 10.0]: # alpha
    row = []
    for j in np.arange(0, 1, 0.25): # l1_ratio
        # define model
        model = ElasticNet(alpha=i, l1_ratio=j)
        # define model evaluation method
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        # append MSE
        row.append(mean(absolute(scores)))
    MSE_data.append(row)

MSE_data
# -

index_names = ['alpha=1e-2',
               'alpha=1e-1',
               'alpha=0.0',
               'alpha=1.0',
               'alpha=10.0']
col_names = ['l1_ratio=0', 'l1_ratio=0.25', 'l1_ratio=0.5', 'l1_ratio=0.75']
MSE_dataframe = pd.DataFrame(MSE_data, index=index_names, columns=col_names)
MSE_dataframe

# +
# construct a table, include a caption: ---------------------------------------
cap = """
<b> Table 1.</b> <em> Cross-validated MSE in elastic net models.</em>
MSE is minimized when alpha=1e-2 and l1_ratio=0.75 .
"""

t1 = MSE_dataframe.to_html(index=True)
t1 = t1.rsplit('\n')
t1.insert(1, cap)
tab1 = ''
for i, line in enumerate(t1):
    tab1 += line
    if i < (len(t1) - 1):
        tab1 += '\n'
display(HTML(tab1))
# -

# ## part b
# Train a series of random forest models and use 10-fold cross-validation for hyper-parameter selection. Focus on tuning the tree depth and number of trees. You may, but are not required, to tune other hyper-parameters as well.
#
# Create a figure or table showing the cross-validated MSE for different hyper-parameters considered and identity which hyper-parameters minimize this quantity.

# +
# Use data in train
data = train.values
X, y = data[:, :-1], data[:, -1]

# generate MSE_rf to store MSE
MSE_rf = []

for i in np.arange(start = 10, stop = 110, step = 20): # n_estimators
    row_rf = []
    for j in np.arange(10, 110, step = 20): # max_depth
        # define model
        model = RandomForestRegressor(n_estimators=i, max_depth=j)
        # define model evaluation method
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        # append MSE
        row_rf.append(mean(absolute(scores)))
    MSE_rf.append(row_rf)

MSE_rf
# -

index_names = ['n_estimators=10',
               'n_estimators=30',
               'n_estimators=50',
               'n_estimators=70',
               'n_estimators=90']
col_names = ['max_depth=10', 
             'max_depth=30', 
             'max_depth=50', 
             'max_depth=70', 
             'max_depth=90']
MSE_rfdataframe = pd.DataFrame(MSE_rf, index=index_names, columns=col_names)
MSE_rfdataframe

# +
# construct a table, include a caption: ---------------------------------------
cap = """
<b> Table 2.</b> <em> Cross-validated MSE in random forest models.</em>
MSE is minimized when n_estimators=70 and max_depth=90 .
"""

t2 = MSE_rfdataframe.to_html(index=True)
t2 = t2.rsplit('\n')
t2.insert(1, cap)
tab2 = ''
for i, line in enumerate(t2):
    tab2 += line
    if i < (len(t2) - 1):
        tab2 += '\n'
display(HTML(tab2))
# -

# ## part c
# Train a series of gradient boosted tree models and use 10-fold cross-validation for hyper-parameter selection. Focus on tuning the number of boosting rounds after selecting a suitable learning rate. You may, but are not required, to tune other hyper-parameters as well.
#
# Create a figure or table showing how the cross-validated MSE changes with the the number of boosting rounds and identity which hyper-parameters minimize this quantity.

# +
# Use data in train
data = train.values
X, y = data[:, :-1], data[:, -1]

# generate MSE_gb_lr to store MSE under different learning rate
MSE_gb_lr = []

for i in np.arange(start = 0.1, stop = 0.9, step = 0.1): # learning_rate
    # define model
    model = GradientBoostingRegressor(learning_rate=i)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # append MSE
    MSE_gb_lr.append(mean(absolute(scores)))

MSE_gb_lr
# -

# We can see when learning_rate=0.7, MSE is minimized.

# +
# generate MSE_gb_n to store MSE under different n_estimators(number of boosting rounds)
MSE_gb_n = []

for j in np.arange(start = 10, stop = 110, step = 20): # n_estimators
    # define model
    model = GradientBoostingRegressor(learning_rate=0.7)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # append MSE
    MSE_gb_n.append(mean(absolute(scores)))
MSE_gb_n
# -

index_names = ['n_estimators=10',
             'n_estimators=30',
             'n_estimators=50',
             'n_estimators=70',
             'n_estimators=90']
col_names = ['learning_rate=0.7']
MSE_gbdataframe = pd.DataFrame(MSE_gb_n, index=index_names, columns=col_names)
MSE_gbdataframe

# +
# construct a table, include a caption: ---------------------------------------
cap = """
<b> Table 3.</b> <em> Cross-validated MSE in gradient boosted tree models.</em>
MSE is minimized when learning_rate=0.7 and n_estimators=30.
"""

t3 = MSE_gbdataframe.to_html(index=True)
t3 = t3.rsplit('\n')
t3.insert(1, cap)
tab3 = ''
for i, line in enumerate(t3):
    tab3 += line
    if i < (len(t3) - 1):
        tab3 += '\n'
display(HTML(tab3))
# -

# # Question 2 - Validation and Testing [20 points]
# Using the hyperparameter selected in the previous section, train 3 models - one for each class - on the entire training sample. (You can do this in the previous question for elastic net.)
#
# Use the trained models to make predictions for each case in the validation set created in question 0. Create a nicely formatted table comparing the out-of-sample MSE on the validation dataset for these three models.
#
# Use whichever model performs best in terms of MSE on the validation dataset to make predictions on the test data and report the corresponding MSE.

# Use data in train
train_data = train.values
X_train, y_train = train_data[:, :-1], train_data[:, -1]
# Using the hyperparameter selected in the previous section
model_EN = ElasticNet(alpha=1e-2, l1_ratio=0.75)
model_RF = RandomForestRegressor(n_estimators=70, max_depth=90)
model_GB = GradientBoostingRegressor(learning_rate=0.7, n_estimators=30)
# train 3 models
model_EN.fit(X_train, y_train)
model_RF.fit(X_train, y_train)
model_GB.fit(X_train, y_train)

# +
# Use data in validation
validate_data = validate.values
X_validate, y_validate = validate_data[:, :-1], validate_data[:, -1]

val_pred_EN = model_EN.predict(X_validate)
val_pred_RF = model_RF.predict(X_validate)
val_pred_GB = model_GB.predict(X_validate)

df_val_pred = pd.DataFrame({'Actual critical_temp':y_validate,
                           'ElasticNet prediction':val_pred_EN,
                           'RandomForest prediction':val_pred_RF,
                           'GradientBoosting prediction':val_pred_GB})
df_val_pred

# +
# Calculate Mean Squared Error
df_val_pred['Forecast Error EN'] = df_val_pred['Actual critical_temp'] - df_val_pred['ElasticNet prediction']
df_val_pred['Forecast Error RF'] = df_val_pred['Actual critical_temp'] - df_val_pred['RandomForest prediction']
df_val_pred['Forecast Error GB'] = df_val_pred['Actual critical_temp'] - df_val_pred['GradientBoosting prediction']
MSE_val_EN = [sum([(x**2)*1/len(df_val_pred) for x in df_val_pred['Forecast Error EN']])]
MSE_val_RF = [sum([(x**2)*1/len(df_val_pred) for x in df_val_pred['Forecast Error RF']])]
MSE_val_GB = [sum([(x**2)*1/len(df_val_pred) for x in df_val_pred['Forecast Error GB']])]

# Create a dataframe consisting MSE of 3 models
MSE_val = pd.DataFrame({'ElasticNet MSE':MSE_val_EN,
                        'RandomForest MSE':MSE_val_RF,
                        'GradientBoosting MSE':MSE_val_GB})
MSE_val

# +
# construct a table, include a caption: ---------------------------------------
cap = """
<b> Table 4.</b> <em> Comparing the out-of-sample MSE on the validation dataset for 
ElasticNet, RandomForest and GradientBoosting tree models.</em>
MSE is minimized when using RandomForest Model.
"""

t4 = MSE_val.to_html(index=False)
t4 = t4.rsplit('\n')
t4.insert(1, cap)
tab4 = ''
for i, line in enumerate(t4):
    tab4 += line
    if i < (len(t4) - 1):
        tab4 += '\n'
display(HTML(tab4))

# +
# Use data in test
test_data = test.values
X_test, y_test = test_data[:, :-1], test_data[:, -1]

#Use RandomForest Model which performs best in terms of MSE on the validation dataset 
# to make predictions on the test data
test_pred_RF = model_RF.predict(X_test)

#Create a dataframe
df_test_pred = pd.DataFrame({'Actual critical_temp':y_test,
                           'RandomForest prediction':test_pred_RF})
df_test_pred

# +
# Calculate Mean Squared Error
df_test_pred['Forecast Error RF'] = df_test_pred['Actual critical_temp'] - df_test_pred['RandomForest prediction']
MSE_test_RF = [sum([(x**2)*1/len(df_test_pred) for x in df_test_pred['Forecast Error RF']])]

# Create a dataframe consisting MSE of RandomForest model
MSE_test = pd.DataFrame({'RandomForest MSE':MSE_test_RF})
MSE_test
# -

# The corresponding MSE = 233.403696 .
