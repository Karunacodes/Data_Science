# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 02:06:45 2022

@author: Karuna Singh
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pylab as plt


mobile = pd.read_csv("D:\\Data Science Files\\Datasets_360\\Association Rules\\myphonedata.csv")
mobile = mobile.iloc[:,3:] # dropping first three columns as these have the same as 4th column onwards
mobile.head()
mobile.isna().sum() # checking for null values : there are no null values


frequent_itemsets = apriori(mobile, min_support = 0.0075, max_len = 3, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgbkymc')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=90)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)


#************************Eliminating redundancy in the rules********************************
def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules and saving in csv file
rules_no_redudancy.sort_values('lift', ascending = False).head(10)

rules_no_redudancy.to_csv("Mobile_rules.csv", index = False, encoding = "utf-8")

# Checking the path of the csv file created
import os
os.getcwd()
