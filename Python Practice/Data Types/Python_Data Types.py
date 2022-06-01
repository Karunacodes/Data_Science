# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 00:20:10 2021

@author: Karuna Singh
"""
# 1. : Construct 2 lists containing all the available data types  (integer, float, string, complex and Boolean)

list_1 = [5, 5.55, 'Dreams', (7+2j), True]
list_2 = [11, 10.01, 'Travel', (1+5j), True]

# (a) : Concatenating above two lists

list_c = [*list_1, *list_2]
list_c

# (b) : Frequency of elements

import collections
freq = collections.Counter(list_c)
freq

# (c): reverse order

list_c.reverse()
list_c

*******************************************************************************
# 2. Create 2 Sets containing integers (numbers from 1 to 10 in one set and 5 to 15 in other set)
import numpy as np
set1 = set(np.arange(1,11))
set2 = set(np.arange(5,16))

# (a) Find the common elements in above 2 Sets

set = set1 | set2

# (b) Find the elements that are not common

set_n = set1 ^ set2

# (c) Remove element 7 from both the Sets

set1.remove(7) 
set2.remove(7)

*******************************************************************************
# 3.Create a data dictionary of 5 states having state name as key and number of covid-19 cases as values.


cov_cases = {'Karnataka' : 100000, 'Maharashtra' : 6005001, 'Delhi' : 800101, 'Punjab' : 300333, 'Uttarpradesh' : 100505}

# (a) Print only state names from the dictionary

cov_cases.keys()

# (b) Update another country and it’s covid-19 cases in the dictionary

cov_cases['Germany'] = 75414

print(cov_cases)
