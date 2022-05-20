# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 18:54:47 2022

@author: Karuna Singh
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans

data = pd.read_excel("C:\\Users\\Karuna Singh\\Study Material\\Datasets_360\\KMeans\\Telco_customer_churn (1).xlsx")

data.info() # Checking data types
data.isnull().sum() # Checking for null values
data.describe() # Checking statistical dimensions of the data

# excluding columns# 1,2 & 3 as they have same value and would not help/impact the analysis but would hamper the computation time
df = data.iloc[:,3:]

# Applying normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
df.iloc[:,[1,2,5,9,21,22,23,24,25,26]] = norm_func(df.iloc[:,[1,2,5,9,21,22,23,24,25,26]])

# Creating dummy variables for all categorical columns
df = pd.get_dummies(df, drop_first = True)


# *********Scree Plot or Elbow curve**************

TWSS = []

k = list(range(2,9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df)
    TWSS.append(kmeans.inertia_)
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 5)
model.fit(df)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data['Clust'] = mb # creating a  new column and assigning it to new column 

data1 = data.pop('Clust') #removing the Clust column to rearrange the columns for clusterwise view of the data
data.insert(loc=0, column= 'Cluster', value = data1) # Inserting the column to view the clusterwise data

data.head()

data_mean = data.iloc[:, 4:].groupby(data.Cluster).mean().T # Calculating aggregate mean of each cluster
data.Cluster.value_counts() # checking count of each cluster

data.to_csv("Kmeans_TelcoCustomer_churn.csv", index=False, encoding = "utf-8") # Creating csv file of this dataset with clusters column

import os
os.getcwd()
