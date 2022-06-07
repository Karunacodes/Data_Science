# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:18:38 2021

@author: Karuna Singh
"""
# Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# reading the data
cars = pd.read_csv('C:\\Users\\Karuna Singh\\Downloads\\Q1_a.csv')

# checking nan values
cars.isna().sum()

# checking the data type
cars.dtypes

# 1st Moment Business Decesion

cars.speed.mean()
cars.speed.median()
cars.speed.mode()

cars.dist.mean()
cars.dist.median()
cars.dist.mode()

# 2nd Moment Business Decesion
cars.speed.var()
cars.speed.std()
cars.dist.var()
cars.dist.std()

#3rd Moment Business Decesion
cars.speed.skew()
cars.dist.skew()

#4th Moment Business Decesion
cars.speed.kurt()
cars.dist.kurt()

# Data Visualization

plt.hist(cars.speed)
sns.boxplot(cars.speed) # there are no outliers

plt.hist(cars.dist)
sns.boxplot(cars.dist) # outliers detected : treatment needed

