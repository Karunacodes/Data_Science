# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 00:11:24 2021

@author: Karuna Singh
"""

# 1.
list1 = [1, 5.5, (10+20j), 'Data Science']
list1.append(7.2) # adds an item at the end of the list
list2 = [11,100,7.9,'Python']
list1.extend(list2) # extends the list by appending specified item/list(here list)
list1.insert(2,'Python')# inserts an item at specified position.
list1.remove(5.5) # removes the specified item from the list(first is removed in case of duplicate items)
list1.pop() # removes the last item from the list(if index not specified else item at specified index is removed)
list1.clear() # removes all the items from the list.
list1.index('Data Science') # returns the index of element specified.
list1.count(1) # returns the number of elements specified.
list1.sort() # this function would not work for this list as has complex and float data type too.
list1.reverse() # reverses the elements of the list.
list = list1.copy() # returns a copy of the list1
print(len(list1)) # returns the length of the list
print(max(list1))# this function would not work because of float and complex data types in the list
print(min(list1))# this function would not work because of float and complex data types in the list

# b.
lst = [i for i in range(10,100,25)] # returns sequence of numbers between 10 and 100 and a gap of 25

# c.

welcome = input("Please enter your full name : ")

print("Hello, welcome ", welcome)

****************************************************************************************

# 2.
lst1 = [i for i in range(10)]

lst2 = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

dict = {lst2[i]: lst1[i] for i in range(0,10)}
dict

****************************************************************************************

# 3.

list_1 = [3,4,5,6,7,8]
list_2 = []
for num in list_1 :
    if num % 2 == 0 :
        list_2.append(num + 10)
    if num % 2 != 0 :
        list_2.append(num * 5)

****************************************************************************************

# 4.

name = input("Please enter your name :  ")
msg = input(" Please enter your message : ")

if msg =="":
    print(" Hello, {}.".format(name))
else :
    print("Hello, {}, {}".format(name, msg))
                       