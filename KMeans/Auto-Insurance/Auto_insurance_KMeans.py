# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 01:57:30 2022

@author: Karuna Singh
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans

data = pd.read_csv("C:\\Users\\Karuna Singh\\Study Material\\Datasets_360\\KMeans\\AutoInsurance (1).csv")

data.info() # Checking data types
data.isnull().sum() # Checking for null values : there are no null values in the dataset
data.describe() # checking statistical dimensions of the data

# Applying normalization on numeric data
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

data_norm= norm_func(data.iloc[:,[2,9,12,13,14,15,16,21]]) # normalizing numerical columns
data_norm.describe() # checking statistical dimensions of the normalized data

# Applying One hot encoding to the designated columns
from sklearn.preprocessing import OneHotEncoder

# Creating instance of One Hot Encoder
enc = OneHotEncoder() # initializing method

data_enc = pd.DataFrame(enc.fit_transform(data.iloc[:,[1,3,4,5,6,7,8,10,11,17,18,19,20,22,23]]).toarray())

auto = pd.concat((data_norm,data_enc),axis=1)
auto_des = auto.describe().T # checking statistical dimensions of the data after normalizing the data

# *********Scree Plot or Elbow curve**************

TWSS = []

k = list(range(2,9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(auto)
    TWSS.append(kmeans.inertia_)
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4).fit(auto)


model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data['Cluster'] = mb # creating new column "Cluster"  and assigning it to the data

data1 = data.pop('Cluster') #removing the Clust column to rearrange the columns for clusterwise view of the data
data.insert(loc=0, column= 'Cluster', value = data1) # Inserting the column to view the clusterwise data

data.iloc[:, 1:].groupby(data.Cluster).mean() # Calculating aggregate mean of each cluster
data.Cluster.value_counts() # checking count of each cluster

data.to_csv("Kmeans_Auto_insurance.csv",index=False, encoding = "utf-8") # Creating csv file oh this dataset with clusters column

import os
os.getcwd()
