# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 20:05:44 2022

@author: Karuna Singh
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans

data = pd.read_excel("C:\\Users\\Karuna Singh\\Study Material\\Datasets_360\\KMeans\\EastWestAirlines (1).xlsx")
data.head()
data.isnull().sum() # There are no null values in the dataset
data.describe() # checking statistical dimensions of the data

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)
df_norm = norm_func(data.iloc[:,1:]) # normalizing the data, excluding the ID# column, it has nominal data
df_norm.describe() # checking statistical dimensions of the data after normalizing and applying OneHot encoding to the data

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 5)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data['clust'] = mb # creating a  new column and assigning it to the data

data.head() # checking the data after adding the clust column

data = data.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]] # rearranging the clust column to view the clusterwise data
data.head()

data.iloc[:, 2:].groupby(data.clust).mean().T # Calculating aggregate mean of each cluster
data.clust.value_counts() # checking count of each cluster

data.to_csv("Kmeans_EastWestAirlines.csv",index=False, encoding = "utf-8") # Creating csv file of this dataset with clusters column

import os
os.getcwd()

