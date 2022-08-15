# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 01:45:23 2022

@author: Karuna Singh
"""

import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

df = pd.read_csv(r"D:\\Data Science Files\\Datasets_360\\Simple Linear Regression\\emp_data.csv")

df.info() # Checking data types & Null values
df.describe() # Checking statistical aspects of the data

df.rename(columns = {"Churn_out_rate" : "COT", "Salary_hike" : "SH"}, inplace = True)


#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.bar(height = df.COT, x = np.arange(1, 11, 1))
plt.hist(df.COT) #histogram : Bimodal data
plt.boxplot(df.COT) #boxplot: no outliers

plt.bar(height = df.SH, x = np.arange(1, 11, 1))
plt.hist(df.SH) #histogram : bimodal data
plt.boxplot(df.SH) #boxplot : no outliers

# Scatter plot
plt.scatter(x = df['SH'], y = df['COT'], color = 'green') # negatively linear correlation

# correlation
np.corrcoef(df.SH, df.COT) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(df.SH, df.COT)[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('COT ~ SH', data = df).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(df['SH']))

# Regression Line
plt.scatter(df.SH, df.COT)
plt.plot(df.SH, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = df.COT - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation

plt.scatter(x = np.log(df['SH']), y = df['COT'], color = 'brown')
np.corrcoef(np.log(df.SH), df.COT) #correlation

model2 = smf.ols('COT ~ np.log(SH)', data = df).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(df['SH']))

# Regression Line
plt.scatter(np.log(df.SH), df.COT)
plt.plot(np.log(df.SH), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = df.COT - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#### Exponential transformation

plt.scatter(x = df['SH'], y = np.log(df['COT']), color = 'orange')
np.corrcoef(df.SH, np.log(df.COT)) #correlation

model3 = smf.ols('np.log(COT) ~ SH', data = df).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(df['SH']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(df.SH, np.log(df.COT))
plt.plot(df.SH, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = df.COT - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation

model4 = smf.ols('np.log(COT) ~ SH + I(SH * SH)', data = df).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(df))
pred4 = np.exp(pred4)
pred4

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = df.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)


plt.scatter(df.SH, np.log(df.COT))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = df.COT - pred4
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

train, test = train_test_split(churn, test_size = 0.2)

finalmodel = smf.ols('np.log(COT) ~ SH + I(SH * SH)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_churn = np.exp(test_pred)
pred_test_churn

# Model Evaluation on Test data
test_res = test.COT - pred_test_churn
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_churn = np.exp(train_pred)
pred_train_churn

# Model Evaluation on train data
train_res = train.COT - pred_train_churn
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse