# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 16:09:59 2021

@author: Karuna Singh
"""

import numpy as np
import pandas as pd

data = pd.read_csv('<Path of the dataset>\\OnlineRetail.csv')

data.dtypes # checking data types

#Missing Values
data.isna().sum() # Checking missing values
# There are missing values in Description column and Customer Id, we can drop customer Id column, being a nominal data
# Description is an important part of analysis, using mode imputer will create bias in the data,
#but every descrption has a unique stock code so, we can drop the Description column & use stock code for analysis
# this 

data.drop(['Description', 'CustomerID'], axis = 1, inplace = True)

# Checking duplicate values
dup = data.duplicated
data1 = data.drop_duplicates()

# Data Preprocessing : Typecasting

data1.describe()

# Quantity & UnitPrice have some negative values : assuming them a typo error, lets replace them with respective positive values
data1['Quantity'] = abs(data1.iloc[:,2])
data1['UnitPrice']=abs(data1.iloc[:,4])
data1.describe()

# Unit price is an integral part of analysis and is in float format, needs to be changed to int64
data1.UnitPrice = data1.UnitPrice.astype('int64')
data1.dtypes

# calculating total price of each transaction
data1['TotalPrice'] = data1['Quantity'] * data1['UnitPrice']
data1.dtypes

# EDA : Visualization

import seaborn as sns

sns.jointplot(x = 'Country', y ='TotalPrice', data = data1, kind = 'scatter')
plt.show()
sns.barplot(x = 'Country', y ='TotalPrice', data = data1);
