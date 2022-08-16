# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 03:28:16 2022

@author: Karuna Singh
"""

import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import matplotlib.pyplot as plt # mostly used for visualization purposes 

sat = pd.read_csv(r"D:\\Data Science Files\\Datasets_360\\Simple Linear Regression\\SAT_GPA.csv")
sat.rename(columns = {"SAT_Scores" : "SATscores"}, inplace = True)

sat.info() # Checking data type & null values : no null values
sat.describe() # Checking statistical dimensions

#Graphical Representation

plt.bar(height = sat.GPA, x = np.arange(1, 201, 1))
plt.hist(sat.GPA) #histogram
plt.boxplot(sat.GPA) #boxplot

plt.bar(height = sat.SATscores, x = np.arange(1, 201, 1))
plt.hist(sat.SATscores) #histogram
plt.boxplot(sat.SATscores) #boxplot

# Scatter plot
plt.scatter(x = sat['SATscores'], y = sat['GPA'], color = 'green') 

# correlation
np.corrcoef(sat.SATscores, sat.GPA) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(sat.SATscores, sat.GPA)[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('GPA ~ SATscores', data = sat).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(sat['SATscores']))

# Regression Line
plt.scatter(sat.SATscores, sat.GPA)
plt.plot(sat.SATscores, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = sat.GPA - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation

plt.scatter(x = np.log(sat['SATscores']), y = sat['GPA'], color = 'brown')
np.corrcoef(np.log(sat.SATscores), sat.GPA) #correlation

model2 = smf.ols('GPA ~ np.log(SATscores)', data = sat).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(sat['SATscores']))

# Regression Line
plt.scatter(np.log(sat.SATscores), sat.GPA)
plt.plot(np.log(sat.SATscores), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = sat.GPA - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#### Exponential transformation


plt.scatter(x = sat['SATscores'], y = np.log(sat['GPA']), color = 'orange')
np.corrcoef(sat.SATscores, np.log(sat.GPA)) #correlation

model3 = smf.ols('np.log(GPA) ~ SATscores', data = sat).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(sat['SATscores']))
pred3_sat = np.exp(pred3)
pred3_sat

# Regression Line
plt.scatter(sat.SATscores, np.log(sat.GPA))
plt.plot(sat.SATscores, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = sat.GPA - pred3_sat
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation

model4 = smf.ols('np.log(GPA) ~ SATscores + I(SATscores*SATscores)', data = sat).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(sat))
pred4_sat = np.exp(pred4)
pred4_sat

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = sat.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)

plt.scatter(sat.SATscores, np.log(sat.GPA))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = sat.GPA - pred4_sat
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

#################### The best model ####################

from sklearn.model_selection import train_test_split

train, test = train_test_split(sat, test_size = 0.2)

finalmodel = smf.ols('np.log(GPA) ~ SATscores + I(SATscores*SATscores)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_sat = np.exp(test_pred)
pred_test_sat

# Model Evaluation on Test data
test_res = test.GPA - pred_test_sat
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_sat = np.exp(train_pred)
pred_train_sat

# Model Evaluation on train data
train_res = train.GPA - pred_train_sat
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse