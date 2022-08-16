# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 02:56:45 2022

@author: Karuna Singh
"""

import pandas as pd
import numpy as np

sal = pd.read_csv(r"D:\\Data Science Files\\Datasets_360\\Simple Linear Regression\\Salary_Data.csv")

sal.info() # Checking data types & Null values : No null values
sal.describe() # Checking Statistical aspects


#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.bar(height = sal.Salary, x = np.arange(1, 31, 1))
plt.hist(sal.Salary) #histogram
plt.boxplot(sal.Salary) #boxplot

plt.bar(height = sal.YearsExperience, x = np.arange(1, 31, 1))
plt.hist(sal.YearsExperience) #histogram
plt.boxplot(sal.YearsExperience) #boxplot

# Scatter plot
plt.scatter(x = sal['YearsExperience'], y = sal['Salary'], color = 'green') 

# correlation
np.corrcoef(sal.YearsExperience, sal.Salary) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(sal.YearsExperience, sal.Salary)[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Salary ~ YearsExperience', data = sal).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(sal['YearsExperience']))

# Regression Line
plt.scatter(sal.YearsExperience, sal.Salary)
plt.plot(sal.YearsExperience, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = sal.Salary - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation

plt.scatter(x = np.log(sal['YearsExperience']), y = sal['Salary'], color = 'brown')
np.corrcoef(np.log(sal.YearsExperience), sal.Salary) #correlation

model2 = smf.ols('Salary ~ np.log(YearsExperience)', data = sal).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(sal['YearsExperience']))

# Regression Line
plt.scatter(np.log(sal.YearsExperience), sal.Salary)
plt.plot(np.log(sal.YearsExperience), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = sal.Salary - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#### Exponential transformation

plt.scatter(x = sal['YearsExperience'], y = np.log(sal['Salary']), color = 'orange')
np.corrcoef(sal.YearsExperience, np.log(sal.Salary)) #correlation

model3 = smf.ols('np.log(Salary) ~ YearsExperience', data = sal).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(sal['YearsExperience']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(sal.YearsExperience, np.log(sal.Salary))
plt.plot(sal.YearsExperience, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = sal.Salary - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#### Polynomial transformation

model4 = smf.ols('np.log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience)', data = sal).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(sal))
pred4_sal = np.exp(pred4)
pred4_sal

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = sal.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)


plt.scatter(sal.YearsExperience, np.log(sal.Salary))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = sal.Salary - pred4_sal
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

train, test = train_test_split(sal, test_size = 0.2)

finalmodel = smf.ols('np.log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_sal = np.exp(test_pred)
pred_test_sal

# Model Evaluation on Test data
test_res = test.Salary - pred_test_sal
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_sal = np.exp(train_pred)
pred_train_sal

# Model Evaluation on train data
train_res = train.Salary - pred_train_sal
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse