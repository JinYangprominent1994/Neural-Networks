# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:03:09 2019

@author: Jerry Xing
"""
import numpy as np
from numpy import linalg as LA

def grdescent(func, w0, stepsize, maxiter, tolerance = 1e-6):
#% function [w]=grdescent(func,w0,stepsize,maxiter,tolerance)
#%
#% INPUT:
#% func function to minimize
#% w0 = initial weight vector 
#% stepsize = initial gradient descent stepsize 
#% tolerance = if norm(gradient)<tolerance, it quits
#%
#% OUTPUTS:
#% 
#% w = final weight vector
#%

    w = w0
    loss = float('10000')
    
    
    for i in range(maxiter):
        loss_temp = loss
        loss,gradient = func(w)
        
        if loss > loss_temp:
            stepsize=stepsize*0.5
        else:
            stepsize=stepsize*1.01;

        w=w-stepsize*gradient
         
        if LA.norm(gradient)<tolerance:
            break
 
    return w