# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 00:13:24 2021

@author: Karuna Singh
"""

#1.	A. Write an equation which relates   399, 543 and 12345 

12345//543 # returns a floored quotient = 22 and hence a remainder of 399

x = 12345
y = 543
z = 399

# equation = x + 22*y + z = 0

equation = 22*y + z

if equation == x:
    print("{}, {} and {} can be related with the equation 22*y + z.". format(x,y,z))
else:
    print('{}, {} and {} cannot be related'. format(x,y,z))


# B.  “When I divide 5 with 3, I got 1. But when I divide -5 with 3, I got -2”—How would you justify it.

5//3 # returns the floored value, which is 1

-5//3 #  here the result is floored and rounded away from zero and the final result will be -2.

***************************************************************************************

# 2. What will be the output of the following:

a = 5
b = 3
c = 10

# A. a/=b 
# above equation refers to a = a/b
a/=b
a

#B. c*=5  
# above eqaution refers to c = c*5
c*=5
c

***************************************************************************************

# 3. A. How to check the presence of an alphabet ‘s’ in word “Data Science” .

's' in "Data Science" # returns false as S is in upper case in "Data Science" and not in lower case.


# B. How can you obtain 64 by using numbers 4 and 3

4**3 # Exponential calculation
