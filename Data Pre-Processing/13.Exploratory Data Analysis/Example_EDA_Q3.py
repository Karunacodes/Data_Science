# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 04:12:44 2021

@author: Karuna Singh
"""

# importing packages
import pandas as pd
import numpy as np

Marks = 34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56 # data
df = pd.DataFrame(Marks) # converting series to dataframe

# 1st Moment Business Decesion
df.mean()
df.median()
df.mode()

# 2nd moment Business Decesion
df.std()
df.var()

# importing package for visualization
import matplotlib.pyplot as plt
plt.hist(df)
plt.boxplot(df)
