# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 20:17:45 2021

@author: Shay Kreymer
"""
import numpy as np
from Utils.funcs_calc_moments import M2_2d, M3_2d
from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots

def calc_M3_for_micrograph(L, c, kvals, Bk, W, N, gamma, T, sigma2, sd):
    """ Calculate first three autocorrelations of a micrograph.

    Args:
        L: diameter of the target image
        c: real representation of the expansion coefficients of the target image
        kvals: vector of frequencies
        Bk: matrix that maps from the expansion coefficients to the approximated image, in the freuency domain
        W: separation between images; L for arbitrary spacing distribution, 2*L-1 for the well-separated case
        N: the width and height of each micrograph
        gamma: the density
        T: matrix that maps from the real representation to the complex representation of the expansion coefficients
        sigma2: the variance of the noise
        sd: a seed

    Returns:
        M1_ys: list of first-order autocorrelations
        M2_ys: list of second-order autocorrelations
        M3_ys: list of third-order autocorrelations
    """
    y_clean, _, _ = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma*(N/L)**2, T, seed=sd)
    y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
    yy = np.zeros((N, N, 1))
    yy[ :, :, 0] = y
    
    M1_y = np.mean(y)
    
    M2_y = np.zeros((L, L))
    for i1 in range(L):
        for j1 in range(L):
            M2_y[i1, j1] = M2_2d(yy, (i1, j1))
            
    M3_y = np.zeros((L, L, L, L))
    for i1 in range(L):
        for j1 in range(L):
            for i2 in range(L):
                for j2 in range(L):
                    M3_y[i1, j1, i2, j2] = M3_2d(yy, (i1, j1), (i2, j2))
    
    return M1_y, M2_y, M3_y
