# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 11:18:30 2021

@author: Karuna Singh
"""

# importing the packages
import pandas as pd
import numpy as np

# reading the packages
iris = pd.read_csv("<Path of the dataset>\\iris.csv")

# checking the nan values : there are no nan values
iris.isna().sum()

iris.duplicated().sum() # checking duplicate records : found none

# checking data types
iris.dtypes

# checking the types of Species in Species column : 3 types of species, can be assigned 3 classes to it
iris['Species'].unique()

# calling Label Encoder : to assign classes to types of species

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

iris['Species'] = labelencoder.fit_transform(iris['Species'])

# Creating new file with Species column with Discretized data
iris.to_csv("Iris_Discretization.csv", index = False, encoding = "utf-8")
# Species column has three classes no. 0, 1 & 2

# Checking the path of the csv file created
import os
os.getcwd()
