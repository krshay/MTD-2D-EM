# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 22:10:47 2019

@author: Shay Kreymer
"""

import numpy as np

def M2_2d(A, shift1):
    """ Calculate second-order autocorrelation of A for shift1.

    Args:
        A: the image
        shift1: a tuple containing the shift
    Returns:
        second-order autocorrelation of A for shift1
    """
    dim1, dim2, _ = np.shape(A)
    
    shift1y = -shift1[0]
    shift1x = -shift1[1]

    valsy1 = [0, -shift1y]
    valsx1 = [0, -shift1x]

    rangey = list(range(max(valsy1), dim1+min(valsy1)))
    rangey1 = [x + shift1y for x in rangey]
    rangex = list(range(max(valsx1), dim2+min(valsx1)))
    rangex1 = [x + shift1x for x in rangex]

    return np.sum(A[min(rangey):max(rangey)+1, min(rangex):max(rangex)+1, :] * A[min(rangey1):max(rangey1)+1, min(rangex1):max(rangex1)+1, :], axis=(0,1))/(dim1*dim2)

def M3_2d(A, shift1, shift2):
    """ Calculate third-order autocorrelation of A for shift1, shift2.

    Args:
        A: the 2-d signal
        shift1, shift2: tuples containing the shifts
    
    Returns:
        third-order autocorrelation of A for shift1, shift2
    """
    dim1, dim2, _ = np.shape(A)
    
    shift1y = -shift1[0]
    shift1x = -shift1[1]

    shift2y = -shift2[0]
    shift2x = -shift2[1]

    valsy1 = [0, -shift1y, -shift2y]
    valsx1 = [0, -shift1x, -shift2x]

    rangey = list(range(max(valsy1), dim1+min(valsy1)))
    if rangey == []: return 0
    rangey1 = [x + shift1y for x in rangey]
    rangey2 = [x + shift2y for x in rangey]

    rangex = list(range(max(valsx1), dim2+min(valsx1)))
    if rangex == []: return 0
    rangex1 = [x + shift1x for x in rangex]
    rangex2 = [x + shift2x for x in rangex]
    
    return np.sum(A[min(rangey):max(rangey)+1, min(rangex):max(rangex)+1, :] * A[min(rangey1):max(rangey1)+1, min(rangex1):max(rangex1)+1, :] * A[min(rangey2):max(rangey2)+1, min(rangex2):max(rangex2)+1, :], axis=(0,1))/(dim1*dim2)
    