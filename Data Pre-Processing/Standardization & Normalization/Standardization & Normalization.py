# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:49:15 2021

@author: Karuna Singh
"""
# Importing libraries
import numpy as np
import pandas as pd

seeds = pd.read_csv('Path of the dataset\\Seeds_data.csv')

seeds.dtypes # checking data types

seeds.isna().sum() # no na/null values

seeds.duplicated().sum() # no duplicate records

seeds.head() # checking first five records

seeds.describe() # checking statistical dimensions of the data

# Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

seed = scaler.fit_transform(seeds) # dataset is now in array format

data = pd.DataFrame(seed) # converting array to DataFrame

# Normalization : defining normalization function

def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return(x)

seeds_norm = norm_func(seeds) # applying the function to dataset

seeds_norm.describe() # all the values lie between 0 & 1 after normalization
