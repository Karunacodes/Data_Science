# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 03:27:02 2021

@author: Karuna Singh
"""

# Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# reading the data
car = pd.read_csv('C:\\Users\\Karuna Singh\\Downloads\\Q1_b.csv')

# checking nan values : no nan values detected
car.isna().sum() 

# checking the data type
car.dtypes

# 1st Moment Business Decesion

car.SP.mean()
car.SP.median()
car.SP.mode()

car.WT.mean()
car.WT.median()
car.WT.mode()

from scipy import stats
stats.mode(car.WT)

# 2nd Moment Business Decesion
car.SP.var()
car.SP.std()

car.WT.var()
car.WT.std()

#3rd Moment Business Decesion
car.SP.skew()
car.WT.skew()

#4th Moment Business Decesion
car.SP.kurt()
car.WT.kurt()

# Data Visualization

plt.hist(car.SP)
sns.boxplot(car.SP) # there are no outliers

plt.hist(car.WT)
sns.boxplot(car.WT) # outliers detected : treatment needed

