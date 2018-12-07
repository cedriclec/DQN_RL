#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:56:19 2017

@author: ced
"""

import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114]) #Y' = 0.299 R + 0.587 G + 0.114 B 

def convert3DTo1D(array3D):
    array2Dgrey = rgb2gray(array3D)
    total_size = array2Dgrey.size
    return array2Dgrey.reshape(total_size)
    #array1D = np.zeros(array2Dgrey.shape[0]*array2Dgrey.shape[1])
#    print(array2Dgrey)
#    print(array2Dgrey.shape)
##    print(array2Dgrey.shape[0])
#    print(array1D.shape)
    #long = int(array2Dgrey.shape[0])
#    larg = int(array2Dgrey.shape[1])
#    for i in range(long):
#        for j in range(larg):
##            print("i ", i, " j ", j ," i * larg ", i * larg)
#            array1D[i*larg + j]
#    return array1D
            
    
def convertBoxToArray(box):
    #Don't think it is usefull
    totalSize = 1
    for i in range(len(box.shape)):
        totalSize *= box.shape[i]
    return np.reshape(np.zeros(totalSize), box)