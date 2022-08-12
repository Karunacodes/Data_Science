# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 16:47:57 2022

@author: Karuna Singh
"""

import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import matplotlib.pyplot as plt # mostly used for visualization purposes 


df = pd.read_csv("D:\\Data Science Files\\Datasets_360\\Simple Linear Regression\\calories_consumed.csv")

df.info() # checking data insights - null values, data types, features

df.describe() # statistical information & distribution of data

#Graphical Representation
df.columns
df.rename(columns ={'Weight gained (grams)':'Weight','Calories Consumed':'Calories'}, inplace = True)
plt.bar(height = df.Weight, x=np.arange(len(df.Weight)))
plt.hist(df.Weight) #histogram : data is positively skewed
plt.boxplot(df.Weight) #boxplot

plt.bar(height = df.Calories, x = np.arange(len(df.Calories)))
plt.hist(df.Calories) #histogram : distribution is multi-modal & rightly skewed
plt.boxplot(df.Calories) #boxplot

# Scatter plot
plt.scatter(x = df['Weight'], y = df['Calories'], color = 'green') 

# correlation
np.corrcoef(df.Weight, df.Calories) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(df.Weight, df.Calories)[0, 1]
cov_output

df.cov() # another method to find covariance


# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Calories ~ Weight', data = df).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(df['Weight']))

# Regression Line
plt.scatter(df.Weight, df.Calories)
plt.plot(df.Weight, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = df.Calories - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(Weight); y = Calories

plt.scatter(x = np.log(df['Weight']), y = df['Calories'], color = 'brown')
np.corrcoef(np.log(df.Weight), df.Calories) #correlation

model2 = smf.ols('Calories ~ np.log(Weight)', data = df).fit()
model2.summary()  # R^2 is decreased from 897 to 878

pred2 = model2.predict(pd.DataFrame(df['Weight']))

# Regression Line
plt.scatter(np.log(df.Weight), df.Calories)
plt.plot(np.log(df.Weight), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = df.Calories - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = Weight y = log(Calories)

plt.scatter(x = df['Weight'], y = np.log(df['Calories']), color = 'orange')
np.corrcoef(df.Weight, np.log(df.Calories)) #correlation

model3 = smf.ols('np.log(Calories) ~ Weight', data = df).fit()
model3.summary() # R^2 is 808

pred3 = model3.predict(pd.DataFrame(df['Weight']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(df.Weight, np.log(df.Calories))
plt.plot(df.Weight, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = df.Calories - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation

model4 = smf.ols('np.log(Calories) ~ Weight + I(Weight*Weight)', data = df).fit()
model4.summary() # R^2 has increased to 852

pred4 = model4.predict(pd.DataFrame(df))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = df.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = df.iloc[:, 1].values


plt.scatter(df.Weight, np.log(df.Calories))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = df.Calories - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse


# from above values it can be illustrated that SLR Model is the best model

#################### The best model #####################


from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size = 0.2)

finalmodel = smf.ols('np.log(Calories) ~ Weight + I(Weight*Weight)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_Cal = np.exp(test_pred)
pred_test_Cal

# Model Evaluation on Test data
test_res = test.Calories - pred_test_Cal
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Cal = np.exp(train_pred)
pred_train_Cal

# Model Evaluation on train data
train_res = train.Calories - pred_train_Cal
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
