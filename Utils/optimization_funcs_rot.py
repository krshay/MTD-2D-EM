# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 20:23:28 2020

@author: Shay Kreymer
"""

import numpy as np
from scipy.optimize import minimize
import Utils.c_g_funcs_rot
from Utils.funcs_calc_moments_rot import calcmap3, calck1, calcN_mat
from Utils.psf_tsf_funcs import makeExtraMat, maketsfMat, maketsfMat_parallel
from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots
import Utils.psf_tsf_funcs

def optimize_2d_known_psf_triplets(initial_guesses, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, K, tsfMat, ExtraMat2, ExtraMat3, numiters=3000, gtol=1e-15):
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
        tsfMat: matrix containing all TSF values; used to ease computations
        ExtraMat2: matrix containing all PSF values; used to ease computations of the second-order autocorrelations
        ExtraMat3: matrix containing all PSF values; used to ease computations of the third-order autocorrelations
        numiters: maximum number of optimization iterations (default is 3000) 
        gtol: gradient norm must be less than gtol before successful termination (default is 1e-15)
    
    Returns:
        The optimization result represented as a OptimizeResult object
    """
    k1_map = calck1(L)
    map3 = calcmap3(L)
    N_mat = calcN_mat(L)
    
    return minimize(fun=Utils.c_g_funcs_rot.cost_grad_fun_rot_notparallel, x0=initial_guesses, method='BFGS', jac=True, options={'disp':False, 'maxiter':numiters, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, map3))

def optimize_rot_Algorithm1_parallel(initial_guesses, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, K, W, N, iters_till_change=150, gtol=1e-15, max_iter=2000):
    """ Optimization of the objective function, using Algorithm 1, with parallel processing.

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
        W: separation between images; L for arbitrary spacing distribution, 2*L-1 for the well-separated case
        N: size of the final synthesized measurement used to approximate the PSF and TSF
        iters_till_change: maximum number of optimization iterations for the initial optimization to estimate gamma (default is 150)
        gtol: gradient norm must be less than gtol before successful termination (default is 1e-15)
        numiters: maximum number of optimization iterations (default is 2000) 
    
    Returns:
        The optimization result represented as a OptimizeResult object
    """
    k1_map = calck1(L)
    map3 = calcmap3(L)
    N_mat = calcN_mat(L)
    
    _, _, locs_init = generate_clean_micrograph_2d_rots(initial_guesses[1: ], kvals, Bk, W, L, 10000, (initial_guesses[0]*(10000/L)**2).astype(int), T, seed=1)
    psf_init = Utils.psf_tsf_funcs.full_psf_2d(locs_init, L)
    ExtraMat2, ExtraMat3 = makeExtraMat(L, psf_init)
    tsf_init = Utils.psf_tsf_funcs.full_tsf_2d(locs_init, L)
    tsfMat = maketsfMat_parallel(L, tsf_init)
        
    first_estimates = minimize(fun=Utils.c_g_funcs_rot.cost_grad_fun_rot_parallel, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':iters_till_change, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, map3))
    first_gamma = first_estimates.x[0]
    print(f'We got to gamma of {first_gamma}')
    first_c = first_estimates.x[1: ]
    
    _, _, locs2 = generate_clean_micrograph_2d_rots(first_c, kvals, Bk, W, L, N, (first_gamma*(N/L)**2).astype(int), T, seed=1000)
    psf2 = Utils.psf_tsf_funcs.full_psf_2d(locs2, L)
    ExtraMat2, ExtraMat3 = makeExtraMat(L, psf2)
    tsf2 = Utils.psf_tsf_funcs.full_tsf_2d(locs2, L)
    tsfMat = maketsfMat_parallel(L, tsf2)
    
    new_guesses = np.concatenate((np.reshape(first_gamma, (1,)), first_c))
    
    return minimize(fun=Utils.c_g_funcs_rot.cost_grad_fun_rot_parallel, x0=new_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':max_iter, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, map3)), psf2, tsf2
   
def optimize_rot_Algorithm1_notparallel(initial_guesses, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, K, W, N, iters_till_change=150, gtol=1e-15, max_iter=2000):
    """ Optimization of the objective function, using Algorithm 1, without parallel processing.

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
        W: separation between images; L for arbitrary spacing distribution, 2*L-1 for the well-separated case
        N: size of the final synthesized measurement used to approximate the PSF and TSF
        iters_till_change: maximum number of optimization iterations for the initial optimization to estimate gamma (default is 150)
        gtol: gradient norm must be less than gtol before successful termination (default is 1e-15)
        numiters: maximum number of optimization iterations (default is 2000) 
    
    Returns:
        The optimization result represented as a OptimizeResult object
    """
    k1_map = calck1(L)
    map3 = calcmap3(L)
    N_mat = calcN_mat(L)
    
    y_init, _, locs_init = generate_clean_micrograph_2d_rots(initial_guesses[1: ], kvals, Bk, W, L, 10000, (initial_guesses[0]*(10000/L)**2).astype(int), T, seed=100)
    psf_init = Utils.psf_tsf_funcs.full_psf_2d(locs_init, L)
    ExtraMat2, ExtraMat3 = makeExtraMat(L, psf_init)
    tsf_init = Utils.psf_tsf_funcs.full_tsf_2d(locs_init, L)
    tsfMat = maketsfMat(L, tsf_init)
        
    first_estimates = minimize(fun=Utils.c_g_funcs_rot.cost_grad_fun_rot_notparallel, x0=initial_guesses, method='BFGS', jac=True, options={'disp':False ,'maxiter':iters_till_change, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, map3))
    first_gamma = first_estimates.x[0]
    first_c = first_estimates.x[1: ]
    
    y2, _, locs2 = generate_clean_micrograph_2d_rots(first_c, kvals, Bk, W, L, 10000, (first_gamma*(10000/L)**2).astype(int), T, seed=1000)
    psf2 = Utils.psf_tsf_funcs.full_psf_2d(locs2, L)
    ExtraMat2, ExtraMat3 = makeExtraMat(L, psf2)
    tsf2 = Utils.psf_tsf_funcs.full_tsf_2d(locs2, L)
    tsfMat = maketsfMat(L, tsf2)
    
    new_guesses = np.concatenate((np.reshape(first_gamma, (1,)), first_c))
    
    return minimize(fun=Utils.c_g_funcs_rot.cost_grad_fun_rot_notparallel, x0=new_guesses, method='BFGS', jac=True, options={'disp': False, 'maxiter':max_iter, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, map3)), psf2, tsf2

# for experiment B
def optimize_2d_known_psf_triplets_with_callback(initial_guesses, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, K, tsfMat, ExtraMat2, ExtraMat3, numiters=3000, gtol=1e-15):
    """ Optimization of the objective function, assuming known PSF and TSF. Here we use callback function to evaluate the value of \gamma through the iterations, for experiment B.

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
        tsfMat: matrix containing all TSF values; used to ease computations
        ExtraMat2: matrix containing all PSF values; used to ease computations of the second-order autocorrelations
        ExtraMat3: matrix containing all PSF values; used to ease computations of the third-order autocorrelations
        numiters: maximum number of optimization iterations (default is 3000) 
        gtol: gradient norm must be less than gtol before successful termination (default is 1e-15)
    
    Returns:
        The optimization result represented as a OptimizeResult object
    """
    k1_map = calck1(L)
    map3 = calcmap3(L)
    N_mat = calcN_mat(L)
    history = [initial_guesses[0]]
    def func_callback(x):
        history.append(x[0])
        
    return minimize(fun=Utils.c_g_funcs_rot.cost_grad_fun_rot_notparallel, x0=initial_guesses, method='BFGS', jac=True, callback=func_callback, options={'disp': True, 'maxiter':numiters, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, map3)), history
