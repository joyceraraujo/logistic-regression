#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 23:42:28 2020

@author: root
"""
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 

def split_dataSet(X,y):
    # split data
    X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=0)
    return X_train, X_test, y_train, y_test
   
    
def train_linearRegression(X_train, y_train):   

    # Create linear regression object
    #linear_regressor = SGDRegressor()
    linear_regressor = LinearRegression()  
    # Train the model using the training sets
    reg = linear_regressor.fit(X_train, y_train)
    return reg 

def train_logisticRegression(X_train, y_train):   

    # Create linear regression object
    #linear_regressor = SGDRegressor()
    logistic_regressor = LogisticRegression()  
    # Train the model using the training sets
    reg = logistic_regressor.fit(X_train, y_train)
    return reg 