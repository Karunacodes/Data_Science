# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 01:45:21 2022

@author: Karuna Singh
"""

import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

time = pd.read_csv(r"D:\\Data Science Files\\Datasets_360\\Simple Linear Regression\\delivery_time.csv")

time.info() # Checking data types & Null values : No null values
time.describe() # checking statistical dimensions

time = time.rename(columns = {"Delivery Time": "DT"})
time = time.rename(columns = {"Sorting Time": "ST"})


#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.bar(height = time.ST, x = np.arange(1, 22, 1))
plt.hist(time.ST) #histogram : data is multi-modal
plt.boxplot(time.ST) #boxplot :no outliers

plt.bar(height = time.DT, x = np.arange(1, 22, 1))
plt.hist(time.DT) #histogram 
plt.boxplot(time.DT) #boxplot : no outliers

# Scatter plot
plt.scatter(x = time['DT'], y = time['ST'], color = 'green') # scatter plot suggest +ve & linear correlation

# correlation
np.corrcoef(time.DT, time.ST) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(time.DT, time.ST)[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('ST ~ DT', data = time).fit()
model.summary() # R^2 is 0.682

pred1 = model.predict(pd.DataFrame(time['DT']))

# Regression Line
plt.scatter(time.DT, time.ST)
plt.plot(time.DT, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = time.ST - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(DT); y = ST

plt.scatter(x = np.log(time['DT']), y = time['ST'], color = 'brown')
np.corrcoef(np.log(time.DT), time.ST) #correlation

model2 = smf.ols('ST ~ np.log(DT)', data = time).fit()
model2.summary() # R^2 has increased to 711

pred2 = model2.predict(pd.DataFrame(time['DT']))

# Regression Line
plt.scatter(np.log(time.DT), time.ST)
plt.plot(np.log(time.DT), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = time.ST - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = DT; y = log(ST)

plt.scatter(x = time['DT'], y = np.log(time['ST']), color = 'orange')
np.corrcoef(time.DT, np.log(time.ST)) #correlation

model3 = smf.ols('np.log(ST) ~ DT', data = time).fit()
model3.summary() # R^2 is 695

pred3 = model3.predict(pd.DataFrame(time['DT']))
pred3_ST = np.exp(pred3)
pred3_ST

# Regression Line
plt.scatter(time.DT, np.log(time.ST))
plt.plot(time.DT, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = time.ST - pred3_ST
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = DT; x^2 = DT*DT; y = log(ST)

model4 = smf.ols('np.log(ST) ~ DT + I(DT*DT)', data = time).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(time))
pred4_ST = np.exp(pred4)
pred4_ST

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = time.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(time.DT, np.log(time.ST))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = time.ST - pred4_ST
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(time, test_size = 0.2)

finalmodel = smf.ols('np.log(ST) ~ DT + I(DT*DT)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_ST = np.exp(test_pred)
pred_test_ST

# Model Evaluation on Test data
test_res = test.ST - pred_test_ST
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_ST = np.exp(train_pred)
pred_train_ST

# Model Evaluation on train data
train_res = train.ST - pred_train_ST
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
