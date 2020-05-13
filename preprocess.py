# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 14:05:03 2019

@author: Jerry Xing
"""
import numpy as np
import numpy.matlib
def preprocess(xTr,xTe):
# function [xTr,xTe,u,m]=preprocess(xTr,xTe);
#
# Preproces the data to make the training features have zero-mean and
# standard-deviation 1
# input:
# xTr - raw training data as d by n_train numpy ndarray 
# xTe - raw test data as d by n_test numpy ndarray
    
# output:
# xTr - pre-processed training data 
# xTe - pre-processed testing data
#
# u,m - any other data should be pre-processed by x-> u*(x-m)
#       where u is d by d ndnumpy array and m is d by 1 numpy ndarray
    
    dTr, nTr = np.shape(xTr)
    dTe, nTe = np.shape(xTe)
    
    temp = np.mean(xTr,axis=1)
    m = temp.reshape(dTr,1)
    temp = np.transpose(np.std(xTr,ddof=0,axis=1))
    u = np.diag(1./temp)

    xTr=np.matmul(u,(xTr-np.matlib.repmat(m,1,nTr)))
    xTe=np.matmul(u,(xTe-np.matlib.repmat(m,1,nTe)))

    return xTr, xTe, u, m
