# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 17:49:26 2021

@author: Karuna Singh
"""
# 1.

age =int( input("Please enter your age : "))
if age > 0 and age < 10 :
    print("Children")
elif age >= 10 and age < 60 :
    print("normal citizen")
elif age >= 60 :
    print("senior citizens")
else:    
    print('please try again with valid age')

************************************************************************************

# 2.

gender = str(input("Please enter your gender : "))
age = int(input("Please enter your age : "))

if age >= 60 and gender == 'male' :
    print("70% fare is applicable.")
elif age >= 60 and gender == 'female':
    print("50% fare is applicable.")
elif age <60 and gender == 'female' :
    print("70% fare is applicable.")
elif age < 60 and gender == 'male' :
    print("100% fare is applicable.")
else:
    print("Please enter the valid information. Thankyou.")
    

************************************************************************************

# 3.

n = int(input("Please enter a number : "))

if n >=0 and n%5 == 0:
    print( "Number is positive and is divisible by 5.")
else :
    print(" Number does not satisfy the given condition.")
    

