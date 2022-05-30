# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 00:27:18 2021

@author: Karuna Singh
"""

import pandas as pd
import numpy as np

df = pd.read_csv('<Path of the dataset>\\claimants.csv')

df.dtypes # checking data types

df.describe() # checking statistical description of the data

duplicate = df.duplicated() # checking for duplicate values : found none
duplicate.sum()

# Missing values : detection

df.isna().sum() # there are missing values in multiple columns


#Missing values : treatment

from sklearn.impute import SimpleImputer

mode_imputer = SimpleImputer(missing_values = np.nan,strategy ='most_frequent')

# boolean values are to be treated with mode
df['CLMSEX'] = pd.DataFrame(mode_imputer.fit_transform(df[['CLMSEX']]))
df['CLMINSUR'] = pd.DataFrame(mode_imputer.fit_transform(df[['CLMINSUR']]))
df['SEATBELT'] = pd.DataFrame(mode_imputer.fit_transform(df[['SEATBELT']]))

#
mean_imputer = SimpleImputer(missing_values=np.nan, strategy ='mean')
df['CLMAGE'] = pd.DataFrame(mean_imputer.fit_transform(df[['CLMAGE']]))

df.isna().sum()
# there are no missing values