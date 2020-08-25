#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:53:53 2020

@author: root
"""
# Machine Learning Online Class - Exercise 2: Logistic Regression

# useful libraries
import numpy as np
import pandas as pd 
#from sklearn.preprocessing import StandardScaler
import Plotting
import Sklearn
import Metrics    
def getting_data(data_file):
    
    df_data =  pd.read_csv(data_file, sep=",", header=None)
    if df_data.shape[1] == 2:
    
        X = df_data.iloc[:, 0:-1].values.reshape(-1, 1)  # values converts it into a numpy array
        y = df_data.iloc[:, -1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    else:
        X = df_data.iloc[:, 0:-1].values  # values converts it into a numpy array
        y = df_data.iloc[:, -1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
        
    return X,y 


# getting data : 
data_file = "ex2data1.txt"
X,y = getting_data(data_file)
Plotting.plot_data_classification(X,y)

#PART 1 : using logistic regression from sklearn

X_train, X_test, y_train, y_test = Sklearn.split_dataSet(X, y)

reg = Sklearn.train_logisticRegression(X_train, y_train.ravel()) # using ravel() to avoid warm message that indicates that y is not a 1d array. 

# Make predictions using the testing set
y_pred = reg.predict(X_test)
# Some metrics to evaluate the model
print("# 1) Method : logistic regression by using sklearn.linear_model")
Metrics.metrics(y_test, y_pred)
#Probability estimates
particular_student = np.array([[45,85]])
prob = reg.predict_proba(particular_student)
print(" admission probability of:",prob)