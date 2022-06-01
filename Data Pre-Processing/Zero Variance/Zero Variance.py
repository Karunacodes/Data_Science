# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 20:30:08 2021

@author: Karuna Singh
"""

import numpy as np
import pandas as pd

data = pd.read_csv('Path of the dataset\\Z_dataset.csv')

data.dtypes # checking data types in dataset

data.isna().sum() # checking na values : found none

data.describe() # checking statatistical description of data

data.duplicated().sum() # checking duplicate values : found none

data.drop(['Id'], axis = 1, inplace = True) # dropping as its nominal data and not needed for analysis

data.var()
#square.length, square.breadth, rec.length and rec.breadth have near-zero variance, hence dropping these columns

data.drop(['square.length'], axis = 1, inplace = True)
           
data.drop(['square.breadth'], axis = 1, inplace = True)

data.drop(['rec.breadth'], axis = 1, inplace = True)
 
# zero variance columns have been removed          
          