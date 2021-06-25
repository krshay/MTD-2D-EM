# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 18:38:45 2021

@author: Shay Kreymer
"""

import numpy as np
from Utils.fb_funcs import expand_fb, min_err_coeffs, calcT
from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots
from Utils.funcs_calc_moments import M2_2d, M3_2d
from Utils.psf_tsf_funcs import full_psf_2d, full_tsf_2d, makeExtraMat, maketsfMat
import Utils.optimization_funcs_rot

def calc_err_size_both(L, ne, sizes, sd):
    """ Calculate estimation error in estimating a specific target image, multiple micrograph sizes. For the case of known PSF and TSF.

    Args:
        L: diameter of the target image
        ne: number of expansion coefficients
        sizes: an array containing the desired values of N, the size of the micrographs to be generated
        sd: a seed

    Returns:
        errs: an array containing the estimation errors for each size (3 initial guesses)
        errs: an array containing the objective function values for each size (3 initial guesses)
    """
    # %% preliminary definitions
    np.random.seed(sd)
    NumGuesses = 10
    errs_known = np.zeros((len(sizes), NumGuesses))
    costs_known = np.zeros((len(sizes), NumGuesses))
    errs_Algorithm1 = np.zeros((len(sizes), NumGuesses))
    costs_Algorithm1 = np.zeros((len(sizes), NumGuesses))
    
    X = np.random.rand(L, L)
    X = X / np.linalg.norm(X)
    
    W = L # L for arbitrary spacing distribution, 2*L-1 for well-separated

    B, z, roots, kvals, nu = expand_fb(X, ne)
    T = calcT(nu, kvals)
    c = np.real(T @ z)
    z = T.H@c
    Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
    for ii in range(nu):
        Bk[ :, :, ii] = np.fft.fft2(np.pad(np.reshape(B[ :, ii], (L, L)), L//2))

    # %% initial guesses
    gamma_initial = 0.09

    cs = np.zeros((NumGuesses, ne))
    for jj in range(NumGuesses):
        X_initial = np.random.rand(L, L)
        X_initial = X_initial / np.linalg.norm(X_initial)
        _, z_initial, _, _, _ = expand_fb(X_initial, ne)
        c_initial = np.real(T @ z_initial)
        cs[jj, :] = c_initial

    # %% calculations
    for (idx, sz) in enumerate(sizes):

        y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, sz, 0.1*(sz/L)**2, T, seed=sd)
        sigma2 = 0
        y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
        del y_clean
        psf = full_psf_2d(locs, L)
        tsf = full_tsf_2d(locs, L)
        
        ExtraMat2, ExtraMat3 = makeExtraMat(L, psf)
        tsfMat = maketsfMat(L, tsf)
        yy = np.zeros((sz, sz, 1))
        yy[ :, :, 0] = y
        del y
        M1_y = np.mean(yy)
        
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
        del yy
        
        for jj in range(NumGuesses):
            X_est_known = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets(np.concatenate((np.reshape(gamma_initial, (1,)), cs[jj, :])), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat, ExtraMat2, ExtraMat3) 
            c_est_known = X_est_known.x[1:]
            z_est_known = T.H @ c_est_known
            est_err_coeffs_known = min_err_coeffs(z, z_est_known, kvals)
            errs_known[idx, jj] = est_err_coeffs_known[0]
            costs_known[idx, jj] = X_est_known.fun
            
            X_est_Algorithm1, _, _ = Utils.optimization_funcs_rot.optimize_rot_Algorithm1_notparallel(np.concatenate((np.reshape(gamma_initial, (1,)), cs[jj, :])), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, W, 10000) 
            c_est_Algorithm1 = X_est_Algorithm1.x[1:]
            z_est_Algorithm1 = T.H @ c_est_Algorithm1
            est_err_coeffs_Algorithm1 = min_err_coeffs(z, z_est_Algorithm1, kvals)
            errs_Algorithm1[idx, jj] = est_err_coeffs_Algorithm1[0]
            costs_Algorithm1[idx, jj] = X_est_Algorithm1.fun

    return errs_known, costs_known, errs_Algorithm1, costs_Algorithm1


def calc_err_size_nopsftsf(L, ne, sizes, sd):
    """ Calculate estimation error in estimating a specific target image, multiple micrograph sizes. For the case of assuming the well-separated model.

    Args:
        L: diameter of the target image
        ne: number of expansion coefficients
        sizes: an array containing the desired values of N, the size of the micrographs to be generated
        sd: a seed

    Returns:
        errs: an array containing the estimation errors for each size (3 initial guesses)
        errs: an array containing the objective function values for each size (3 initial guesses)
    """
    # %% preliminary definitions
    np.random.seed(sd)
    NumGuesses = 10
    errs = np.zeros((len(sizes), NumGuesses))
    costs = np.zeros((len(sizes), NumGuesses))
    X = np.random.rand(L, L)
    X = X / np.linalg.norm(X)
    
    W = L # L for arbitrary spacing distribution, 2*L-1 for well-separated

    B, z, roots, kvals, nu = expand_fb(X, ne)
    T = calcT(nu, kvals)
    c = np.real(T @ z)
    z = T.H@c
    Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
    for ii in range(nu):
        Bk[ :, :, ii] = np.fft.fft2(np.pad(np.reshape(B[ :, ii], (L, L)), L//2))

    # %% initial guesses
    gamma_initial = 0.09
    # y_init, s_init, locs_init = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma*(N/L)**2, T)
   
    cs = np.zeros((NumGuesses, ne))
    for jj in range(NumGuesses):
        X_initial = np.random.rand(L, L)
        X_initial = X_initial / np.linalg.norm(X_initial)
        _, z_initial, _, _, _ = expand_fb(X_initial, ne)
        c_initial = np.real(T @ z_initial)
        cs[jj, :] = c_initial

    # %% calculations
    for (idx, sz) in enumerate(sizes):
        y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, sz, 0.100*(sz/L)**2, T, seed=sd)
        # gamma = s[0]*(L/N)**2
        sigma2 = 0
        y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
        
        psf = np.zeros((4*L - 3, 4*L - 3))
        tsf = np.zeros((4*L - 3, 4*L - 3, 4*L - 3, 4*L - 3))
        
        ExtraMat2, ExtraMat3 = makeExtraMat(L, psf)
        tsfMat = maketsfMat(L, tsf)
        yy = np.zeros((sz, sz, 1))
        yy[ :, :, 0] = y
        
        M1_y = np.mean(yy)
        
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
    
        for jj in range(NumGuesses):
            X_est_known = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets(np.concatenate((np.reshape(gamma_initial, (1,)), cs[jj, :])), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat, ExtraMat2, ExtraMat3) 
            c_est_known = X_est_known.x[1:]
            z_est_known = T.H @ c_est_known
            est_err_coeffs_known = min_err_coeffs(z, z_est_known, kvals)
            errs[idx, jj] = est_err_coeffs_known[0]
            costs[idx, jj] = X_est_known.fun
        
    return errs, costs
    
