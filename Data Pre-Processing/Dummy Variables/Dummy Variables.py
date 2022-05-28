# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 02:48:51 2021

@author: Karuna Singh
"""

import numpy as np
import pandas as pd

# reading the dataset
df = pd.read_csv('<Path of the dataset>\\animal_category.csv')

df.dtypes # checking data types

df.isna().sum() # checking na values : found none

df.duplicated().sum() # checking duplicate records : there are none
# Choosing the Label Encoding technique as it creates a single column for each 
# variable and thus enhancing computational speed. Also the Column names are 
#retained, no extra steps to rename the columns.

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df['Animals'] = labelencoder.fit_transform(df['Animals'])
df['Gender'] = labelencoder.fit_transform(df['Gender'])
df['Homly'] = labelencoder.fit_transform(df['Homly'])
df['Types'] = labelencoder.fit_transform(df['Types'])
