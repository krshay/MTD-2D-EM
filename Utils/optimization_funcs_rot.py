# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 20:23:28 2020

@author: Shay Kreymer
"""

from scipy.optimize import minimize
import Utils.c_g_funcs_rot
from Utils.funcs_calc_moments_rot import calcmap3, calck1, calcN_mat
import Utils.psf_tsf_funcs

def optimize_2d(initial_guesses, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, K, numiters=3000, gtol=1e-15):
    """ Optimization of the objective function, assuming known PSF and TSF.

    Args:
        initial guesses: vector containing initial guesses for gamma and c
        Bk: matrix that maps from the expansion coefficients to the approximated image, in the freuency domain
        T: matrix that maps from the real representation to the complex representation of the expansion coefficients
        kvals: vector of frequencies
        M1_y: the first-order autocorrelation of the measurement
        M2_y: the second-order autocorrelations of the measurement, of size L * L
        M3_y: the third-order autocorrelations of the measurement, of size L * L * L * L
        sigma2: the variance of the noise
        L: diameter of the target image
        K: number of target images' types; may be utilized for heterogeneity
        numiters: maximum number of optimization iterations (default is 3000) 
        gtol: gradient norm must be less than gtol before successful termination (default is 1e-15)
    
    Returns:
        The optimization result represented as a OptimizeResult object
    """
    k1_map = calck1(L)
    map3 = calcmap3(L)
    N_mat = calcN_mat(L)
    
    return minimize(fun=Utils.c_g_funcs_rot.cost_grad_fun_rot_notparallel, x0=initial_guesses, method='BFGS', jac=True, options={'disp':False, 'maxiter':numiters, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, K, N_mat, k1_map, map3))
