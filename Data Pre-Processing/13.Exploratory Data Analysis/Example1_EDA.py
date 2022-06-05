# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:18:38 2021

@author: Karuna Singh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
cars = pd.read_csv('C:\\Users\\Karuna Singh\\Downloads\\Q1_a.csv')
cars.isna().sum()
# 1st moment business decesion
cars.describe()
cars.speed.mean()
cars.speed.median()
cars.speed.mode()

cars.dist.mean()
cars.dist.median()
cars.dist.mode()

# 2nd moment business decesion
cars.speed.var()
cars.speed.std()
cars.dist.var()
cars.dist.std()

#3rd moment decesion
cars.speed.skew()
cars.dist.skew()

#4th moment decesion
cars.speed.kurt()
cars.dist.kurt()

#5th Moment business decesion

sns.boxplot(cars.speed) # there are no outliers
sns.boxplot(cars.dist) # outliers detected : treatment needed

# Outlier treatment

IQR = cars.dist.quantile(0.75) - cars.dist.quantile(0.25)
lower_limit = cars.dist.quantile(0.25) - IQR * 1.5
upper_limit = cars.dist.quantile(0.75) + IQR * 1.5

# Replacing outlier

cars['cars_replaced'] = pd.DataFrame(np.where(cars['dist'] > upper_limit, upper_limit, np.where(cars['dist'] < lower_limit, lower_limit, cars['dist'])))

sns.boxplot(cars.cars_replaced)
# there are no outliers in dist data

# checking if data is normally distributed or not

import scipy.stats as stats
import pylab

stats.probplot(cars.speed, dist = 'norm', plot = pylab)
# its anormal distibution

stats.probplot(cars.dist, dist = 'norm', plot = pylab)

stats.probplot(np.log(cars.dist), dist = 'norm', plot = pylab)

###############################################################################
###############################################################################
cars1 = pd.read_csv('C:\\Users\\Karuna Singh\\Downloads\\Q2_b.csv')

# 1st Moment Business decision

cars1.SP.mean()
cars1.SP.median()
cars1.SP.mode()

cars1.WT.mean()
cars1.WT.median()
cars1.WT.mode()

# 2nd Moment Business decesion

cars1.SP.var()
cars1.SP.std()

cars1.WT.var()
cars1.WT.std()


# 3rd Moment Business decesion

cars1.SP.skew()

cars1.WT.skew()

# 4th Moment Business decesion

cars1.SP.kurt()

cars1.WT.kurt()

# 5th Moment Business decesion

sns.boxplot(cars1.SP)

IQR = cars1.SP.quantile(0.75) - cars1.quantile(0.25)
lower_limit = cars1.SP.quantile(0.25) - (IQR * 1.5)
upper_limit = cars1.SP.quantile(0.75) + (IQR * 1.5)

# Outlier treatment : Replacing

cars1['cars1_replaced'] = pd.DataFrame(np.where(cars1['SP'] > upper_limit, upper_limit, np.where(cars1['SP'] < lower_limit, lower_limit, cars1['SP'])))
df['df_replaced'] = pd.DataFrame(np.where(df['Salaries'] > upper_limit, upper_limit, np.where(df['Salaries'] < lower_limit, lower_limit, df['Salaries'])))