# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 00:55:56 2022

@author: Karuna Singh
"""

import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv(r"D:\\Data Science Files\\Datasets_360\\SVM\\forestfires.csv")
df.head()
df.isnull().sum() # CXhecking for any null values : no null values are present
df.duplicated().sum() # not considering the results as there is no unique identity provided peraining to any forest area
df.info() # Checking data details
df.describe() # checking statistical information of the data

# Dropping the month and day columns as  we have dummies of these columns
df.drop(["month","day"],axis=1,inplace =True)

# Normalising the data as there is scale difference 
predictors = df.iloc[:,0:28]
target = df.iloc[:,28]


def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return (x)

fires = norm_func(predictors)
fires

from sklearn.svm import SVC
# import support vector classification

from sklearn.model_selection import train_test_split

# splitting the data into train and test datasets
x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.25, stratify = target)

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# ########### Kernel = linear ###########

help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)

np.mean(pred_test_linear==y_test)

confusion_matrix(y_test,pred_test_linear)
