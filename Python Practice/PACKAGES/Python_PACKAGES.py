# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:58:20 2021

@author: Karuna Singh
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\Karuna Singh\\OneDrive\\360\\Assignments\\Python Problem Statements\\Indian_cities.csv')

data.isna().sum()

data.duplicated().sum()

#1.
# a) Find out top 10 states in female-male sex ratio

data_FMR = data.sort_values(by='sex_ratio', ascending = False)
data_top10_st_FMR = data_FMR.head(10)
print(data_top10_st_FMR)

# b) Find out top 10 cities in total number of graduates

data_TG = data.sort_values(by='total_graduates', ascending = False)
data_top10_ct_TG = data_TG.head(10)
print(data_top10_ct_TG)

# c) Find out top 10 cities and their locations in respect of  total effective_literacy_rate.

data_TELR = data.sort_values(by='effective_literacy_rate_total', ascending = False)
data_top10_ct_TELR = data_TELR.head(10)
print(data_top10_ct_TELR)

**********************************************************************************************
# 2.
# a) Construct histogram on literates_total and comment about the inferences
plt.hist(data.literates_total) # it is not a normal data, it is positively skewed

# b) Construct scatter  plot between  male graduates and female graduates

x = data.male_graduates
y = data.female_graduates
plt.scatter(x,y)                # shows both x and y are highly positive corelation

**********************************************************************************************

# 3.
# a) Construct Boxplot on total effective literacy rate and draw inferences

plt.boxplot(data.effective_literacy_rate_total) # it has outliers on the lower fence.

# b) Find out the number of null values in each column of the dataset and delete them.

data.isnull().sum()
# There are no null values in the indian cities dataset