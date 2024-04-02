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

# ## part a) 
# Revise your solution to PS2 Question 3 to also include gender (RIAGENDR) in the demographic data.
#
# Update (October 14): Include your data files in your submission and with extension .pickle, .feather or .parquet and include a code cell here that imports those files from the local directory (the same folder as your .ipynb or .py source files).

# modules: --------------------------------------------------------------------
import numpy as np
import pandas as pd
from os.path import exists
from math import floor 
from timeit import Timer
from collections import defaultdict
from IPython.core.display import display, HTML

# file location: -------------------------------------------------------------
path = '/Users/ShuyanLi/desktop/Umich_lsy/STATS507/HW4'

# column maps: ---------------------------------------------------------------
# new names for demo cols
demo_cols = {
    'SEQN': 'id',
    'RIDAGEYR': 'age',
    'RIAGENDR': 'gender',
    'RIDRETH3': 'race',
    'DMDEDUC2': 'education',
    'DMDMARTL': 'marital_status',
    'RIDSTATR': 'exam_status',
    'SDMVPSU': 'psu',
    'SDMVSTRA': 'strata',
    'WTMEC2YR': 'exam_wt',
    'WTINT2YR': 'interview_wt'
    }

# new names for ohx cols
ohx_cols = {'SEQN': 'id', 'OHDDESTS': 'dentition_status'}
tc_cols = {'OHX' + str(i).zfill(2) + 'TC':
           'tc_' + str(i).zfill(2) for i in range(1, 33)}
ctc_cols = {'OHX' + str(i).zfill(2) + 'CTC':
            'ctc_' + str(i).zfill(2) for i in range(2, 32)}
_, _ = ctc_cols.pop('OHX16CTC'), ctc_cols.pop('OHX17CTC')

ohx_cols.update(tc_cols)
ohx_cols.update(ctc_cols)

# columns to convert to integer
demo_int = ('id', 'age', 'psu', 'strata')
ohx_int = ('id', )

# levels for categorical variables
demo_cat = {
    'gender': {1: 'Male', 2: 'Female'},
    'race': {1: 'Mexican American',
             2: 'Other Hispanic',
             3: 'Non-Hispanic White',
             4: 'Non-Hispanic Black',
             6: 'Non-Hispanic Asian',
             7: 'Other/Multiracial'
             },
    'education': {1: 'Less than 9th grade',
                  2: '9-11th grade (Includes 12th grade with no diploma)',
                  3: 'High school graduate/GED or equivalent',
                  4: 'Some college or AA degree',
                  5: 'College graduate or above',
                  7: 'Refused',
                  9: "Don't know"
                  },
    'marital_status': {1: 'Married',
                       2: 'Widowed',
                       3: 'Divorced',
                       4: 'Separated',
                       5: 'Never married',
                       6: 'Living with partner',
                       77: 'Refused',
                       99: "Don't know"
                       },
    'exam_status': {1: 'Interviewed only',
                    2: 'Both interviewed and MEC examined'
                    }
    }

ohx_cat = {
    'dentition_status': {1: 'Complete', 2: 'Partial', 3: 'Not Done'}
    }

tc = {
      1: 'Primary tooth present',
      2: 'Permanent tooth present',
      3: 'Dental Implant',
      4: 'Tooth not present',
      5: 'Permanent dental root fragment present',
      9: 'Could not assess'
      }

ctc = (
 {
  'A': 'Primary tooth with a restored surface condition',
  'D': 'Sound primary tooth',
  'E': 'Missing due to dental disease',
  'F': 'Permanent tooth with a restored surface condition',
  'J':
    'Permanent root tip is present but no restorative replacement is present',
  'K': 'Primary tooth with a dental carious surface condition',
  'M': 'Missing due to other causes',
  'P':
    'Missing due to dental disease but replaced by a removable restoration',
  'Q':
    'Missing due to other causes but replaced by a removable restoration',
  'R':
    'Missing due to dental disease but replaced by a fixed restoration',
  'S': 'Sound permanent tooth',
  'T':
    'Permanent root tip is present but a restorative replacement is present',
  'U': 'Unerupted',
  'X': 'Missing due to other causes but replaced by a fixed restoration',
  'Y': 'Tooth present, condition cannot be assessed',
  'Z': 'Permanent tooth with a dental carious surface condition'
 })

# read data: -----------------------------------------------------------------
base_url = 'https://wwwn.cdc.gov/Nchs/Nhanes/'
cohorts = (
    ('2011-2012', 'G'),
    ('2013-2014', 'H'),
    ('2015-2016', 'I'),
    ('2017-2018', 'J')
    )
# demographic data
demo_file = path + '/demo.feather'

if exists(demo_file):
    demo = pd.read_feather(demo_file)
else:
    demo_cohorts = {}
    for cohort, label in cohorts:

        # read data and subset columns
        url = base_url + cohort + '/DEMO_' + label + '.XPT'
        dat = pd.read_sas(url).copy()
        dat = dat[list(demo_cols.keys())].rename(columns=demo_cols)

        # assign cohort and collect
        dat['cohort'] = cohort
        demo_cohorts.update({cohort: dat})

    # concatenate and save
    demo = pd.concat(demo_cohorts, ignore_index=True)
 
    # categorical variables
    for col, d in demo_cat.items():
        demo[col] = pd.Categorical(demo[col].replace(d))
    demo['cohort'] = pd.Categorical(demo['cohort'])

    # integer variables
    for col in demo_int:
        demo[col] = pd.to_numeric(demo[col], downcast='integer')

    demo.to_feather(demo_file)
demo.shape

# dentition data
ohx_file = path + '/ohx.feather'

if exists(ohx_file):
    ohx = pd.read_feather(ohx_file)
else:
    ohx_cohorts = {}
    for cohort, label in cohorts:

        # read data and subset columns
        url = base_url + cohort + '/OHXDEN_' + label + '.XPT'
        dat = pd.read_sas(url).copy()
        dat = dat[list(ohx_cols.keys())].rename(columns=ohx_cols)

        # assign cohort and collect
        dat['cohort'] = cohort
        ohx_cohorts.update({cohort: dat})
 
    # concatenate
    ohx = pd.concat(ohx_cohorts, ignore_index=True)

    # categorical variables
    for col, d in ohx_cat.items():
        ohx[col] = pd.Categorical(ohx[col].replace(d))
    
    for col in tc_cols.values():
        ohx[col] = pd.Categorical(ohx[col].replace(tc))

    # ctc columns get read in as bytes
    for col in ctc_cols.values():
        ohx[col] = ohx[col].apply(lambda x: x.decode('utf-8'))
        ohx[col] = pd.Categorical(ohx[col].replace(ctc))

    ohx['cohort'] = pd.Categorical(ohx['cohort'])
    # integer variables
    for col in ohx_int:
        ohx[col] = pd.to_numeric(ohx[col], downcast='integer')

    # save
    ohx.to_feather(ohx_file)
ohx.shape
# ---


