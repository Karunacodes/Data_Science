# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# importing datasets
import numpy as np
import pandas as pd
import seaborn as sns

# reading dataset
df = pd.read_csv('<Path of the dataset>\\boston_data.csv')

#checking data types
df.dtypes

# checking nan values in the data : there are no nan values
df.isna().sum()

df.duplicated().sum() # Checking duplicate records : found none

df.shape # checking the shape of the data

df.head() # checking data
# by looking at the data : found out chas to be a boolean data and outliers cannot be found, so will have to consider differently in outlier treatment

# checking the statistical details of the data
df.describe()

# checking outlier : Visualization

sns.boxplot(data = df, orient = "V")

# Outlier treatment

#As the entire dataset has numeric values, we can find outliers without formatting the data, and can be done to entire dataset

IQR = df.quantile(0.25) - df.quantile(0.75)
lower_limit = df.quantile(0.25) - (IQR * 1.5)
upper_limit = df.quantile(0.75) + (IQR * 1.5)

# finding outliers
df_outlier = np.where(df > upper_limit, True, np.where(df < lower_limit, True, False))


# Winsorization can be applied to all the columns & is better approach as there is no data loss, also is computation time saving too
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5,variables=['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat', 'medv'])
df_t = winsor.fit_transform(df[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat', 'medv']])

df_t['chas'] = df['chas'] # replacing new 'chas' value with old values as its a boolean data and winsorization has changed the data completely by replacing all the 1s with 0s.

# can cross check the outliers status before and after winsorization

sns.boxplot(df.black) # has outliers on leftside
sns.boxplot(df_t.black) # no outliers

sns.boxplot(df.rm) # has ouliers on both sides 
sns.boxplot(df_t.rm) # has no outliers