# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:11:51 2022

@author: Karuna Singh
"""

import pandas as pd  # for data manipulation,cleaning and analysis
import matplotlib.pyplot as plt
import numpy as np # to handle numerical aspects of the data

# Importing train & test datasets
df_train=pd.read_csv(r"D:\\Data Science Files\\Datasets_360\\SVM\\SalaryData_Train (1).csv")
df_train.head()
df_test=pd.read_csv(r"D:\\Data Science Files\\Datasets_360\\SVM\\SalaryData_Test (1).csv")
df_test.head()

# ************************** Train data EDA ***********************************

df_train.isna().sum() # checking null values in the data
df_train.duplicated().sum() # not taking into consideration as there is no unique identity of each row, so chances of same information is there
df_train.describe() # checking statistical dimesions of the data
df_train.info() # checking the data details

# ************************** Test data EDA ************************************

df_test.isna().sum() # checking null values in the data
df_test.duplicated().sum() # not taking into consideration as there is no unique identity of each row, so chances of same information is there
df_test.describe() # checking statistical dimesions of the data
df_test.info() # checking the data details

string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

# Preprocessing the data. As, there are categorical variables
from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
for i in string_columns:
        df_train[i]= number.fit_transform(df_train[i])
        df_test[i]=number.fit_transform(df_test[i])
        
# Capturing the column names which can help in futher process
colnames = df_train.columns
colnames
len(colnames)

x_train = df_train[colnames[0:13]]
y_train = df_train[colnames[13]]
x_test = df_test[colnames[0:13]]
y_test = df_test[colnames[13]]

# Normalmization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
x_train = norm_func(x_train)
x_test =  norm_func(x_test)

from sklearn.svm import SVC

######## Kernel = Linear #############

model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)

np.mean(pred_test_linear==y_test) 


########### Kernel = poly ###############
# Highest accuracy is achieved with this Kernel
model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)

np.mean(pred_test_poly==y_test) 

confusion_matrix(y_test,pred_test_poly)

########### Kernel = rbf ###############

model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)

np.mean(pred_test_rbf==y_test) 


########### Sigmoid #####################
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(x_train,y_train)
pred_test_sig = model_rbf.predict(x_test)

np.mean(pred_test_sig==y_test) 

pred_test_linear