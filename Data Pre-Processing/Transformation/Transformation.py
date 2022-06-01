# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 18:18:50 2021

@author: Karuna Singh
"""

import numpy as np
import pandas as pd

data = pd.read_csv('C:\\Users\\Karuna Singh\\Study Material\\Datasets_360\\calories_consumed.csv')

data.dtypes # checkinf data types

data.isna().sum() # no na/null values

data.duplicated().sum() # no duplicates

data.describe() # checking statistical summary of the data

import seaborn as sns
sns.boxplot(data['Weight gained (grams)']) # there are no outliers but data is poitively skewed

sns.boxplot(data['Calories Consumed']) # there are no outliers but data is poitively skewed

import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed
stats.probplot(data['Weight gained (grams)'], dist="norm", plot=pylab) # graph is not linear

stats.probplot(data['Calories Consumed'], dist="norm", plot=pylab) # graph is linear : data is normally distributed


# Transformation to make workex variable normal
# 1. Log
stats.probplot(np.log(data['Weight gained (grams)']), dist="norm", plot=pylab) # data is now normally distributed


# 2. Square root
stats.probplot(np.sqrt(data['Weight gained (grams)']), dist="norm", plot=pylab) # data is now normally distributed


# 3. Cube root
stats.probplot(np.cbrt(data['Weight gained (grams)']), dist="norm", plot=pylab) # data is now normally distributed

