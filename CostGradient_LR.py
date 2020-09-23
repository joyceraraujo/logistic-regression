#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:08:00 2020

@author: root
"""
import numpy as np

#  functions to use in Gradient Descent method
def sigmoid(z):

    g = 1 /(1 + np.exp(-z))
    return g


def hypothesis(theta, x):
        
        h = (theta*x).sum(axis=1).reshape(-1,1)       
        h = sigmoid(h)
        return h

def cost(theta,X,y):
    epsilon = 1e-5 
    m = len(X)
    h_X = hypothesis(theta, X)  
    a = -y*np.log(h_X+ epsilon)  
    
    b = (1 -y) * np.log(1-h_X + epsilon)   
    J = a -b 

    J = J.sum()/m

    return J #J[0][0]

def gradient_cost(theta,X,y):
    m = len(X)
        
    h = hypothesis(theta, X)
    
    gradient = ((h-y)*X)/m  
    
    gradient = gradient.sum(axis=0)
      



        
    return gradient