# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 00:27:35 2022

@author: Karuna Singh
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans

data = pd.read_csv("C:\\Users\\Karuna Singh\\Study Material\\Datasets_360\\KMeans\\crime_data (1).csv")

data.info()  # Checking data types
data.rename({'Unnamed: 0' : 'State'}, axis=1, inplace=True) # renaming the unnamed column
data.isnull().sum() # Checking for null values
data.describe() # Checking statistical dimensions of the data

# Normalization function
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

df_norm = norm_func(data.iloc[:,1:])
df_norm.describe() # Checking data after normalization


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

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data['Clust'] = mb # creating a  new column and assigning it to new column 

data.head() # checking the data for Clust column

data = data.iloc[:,[5,0,1,2,3,4]] # rearranging the columns to view the clusterwise data

data.iloc[:, 2:].groupby(data.Clust).mean() # Calculating aggregate mean of each cluster
data.Clust.value_counts() # checking count of each cluster


# Cluster wise visulaization on a stacked bar
data_clust = data.iloc[:, 2:].groupby(data.Clust).mean()

x = ['Cluster 0','Cluster 1','Cluster 2','Cluster 3']
y1 = np.array(data_clust['Murder'])
y2 = np.array(data_clust['Assault'])
y3 = np.array(data_clust['UrbanPop'])
y4 = np.array(data_clust['Rape'])

plt.figure(figsize=(8,6))
plt.bar(x, y1,color = 'tab:orange')
plt.bar(x, y2, bottom=y1, color ='tab:blue')
plt.bar(x, y3, bottom =y1+y2, color ='palevioletred')
plt.bar(x, y4,bottom =y1+y2+y3, color ='tab:red')

plt.xlabel("Clusters")
plt.ylabel("Crime")
plt.legend(["Murder", "Assault", "UrbanPop", "Rape"])
plt.title("Cluster wise Crime data", fontsize = 15)
plt.show()

data.to_csv("Kmeans_Crime_data.csv",index=False, encoding = "utf-8") # Creating csv file oh this dataset with clusters column

import os
os.getcwd()