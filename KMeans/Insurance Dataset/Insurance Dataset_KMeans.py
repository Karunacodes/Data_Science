# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 18:22:03 2022

@author: Karuna Singh
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans

ins = pd.read_csv("C:\\Users\\Karuna Singh\\Study Material\\Datasets_360\\KMeans\\Insurance Dataset.csv")
ins.info() # Checking data types
ins.isnull().sum() # Checking for null values : there are no null values in the dataset
ins.describe() # checking statistical dimensions of the data

# Normalization function
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
df_norm = norm_func(ins)
df_norm.describe() # checking statistical dimensions of the normalized data

#*********Scree Plot or Elbow Plot**************

TWSS = []

k = list(range(2,9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
ins['Clust'] = mb # creating a  new column and assigning it to new column 

ins.head() # checking 'Clust' column added to 'ins' data

ins = ins.iloc[:,[5,0,1,2,3,4]] # rearranging the columns to view the clusterwise data

ins.iloc[:, 2:].groupby(ins.Clust).mean() # Calculating aggregate mean of each cluster
ins.Clust.value_counts() # checking count of each cluster

ins.to_csv("Kmeans_Insurance Dataset.csv", index=False, encoding = "utf-8") # Creating csv file of Insurance dataset with clusters column

import os
os.getcwd()
