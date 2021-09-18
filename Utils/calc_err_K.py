# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 15:36:14 2021

@author: Shay Kreymer
"""

import numpy as np
from Utils.fb_funcs import expand_fb, min_err_coeffs, calcT
from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots
from Utils.funcs_calc_moments import M2_2d, M3_2d
import Utils.optimization_funcs_rot
from Utils.EM_funcs import EM

import time

def calc_err_K_both(L, ne, Ks, SNR, N, gamma, sd):
    """ Calculate estimation error in estimating a specific target image, multiple micrograph Ks. For the case of EM PSF and TSF.

    Args:
        L: diameter of the target image
        ne: number of expansion coefficients
        Ks: an array containing the desired values of N, the size of the micrographs to be generated
        sd: a seed

    Returns:
        errs_EM: an array containing the estimation errors for each size, EM
        costs_EM: an array containing the objective function values for each size, EM
        errs_ac: an array containing the estimation errors for each size, Algorithm 1
        costs_ac: an array containing the objective function values for each size, Algorithm 1
    """
    # %% preliminary definitions
    np.random.seed(sd)
    N = (N // L) * L
    NumGuesses = 1
    
    errs_EM = np.zeros((len(Ks), NumGuesses))
    costs_EM = np.zeros((len(Ks), NumGuesses))
    times_EM = np.zeros((len(Ks), NumGuesses))
    errs_ac = np.zeros((len(Ks), NumGuesses))
    costs_ac = np.zeros((len(Ks), NumGuesses))
    times_ac = np.zeros((len(Ks), NumGuesses))

    X = np.random.rand(L, L)
    X = 10 * X / np.linalg.norm(X)
    
    sigma2 = np.linalg.norm(X)**2 / (L**2 *SNR)
    
    W = 2*L - 1 # L for arbitrary spacing distribution, 2*L-1 for well-separated

    B, z, roots, kvals, nu = expand_fb(X, ne)
    T = calcT(nu, kvals)
    c = np.real(T @ z)
    z = T.H@c
    Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
    for ii in range(nu):
        Bk[ :, :, ii] = np.fft.fft2(np.pad(np.reshape(B[ :, ii], (L, L)), L//2))

    # %% initial guesses
    gamma_initial = 0.03
    
    beta0 = 0.90
    rho_init = np.zeros((2*L, 2*L))
    for i in range(2*L):
        for j in range(2*L):
            rho_init[i, j] = (1 - beta0) / (2*L - 1)**2
            if i == L or j == L:
                rho_init[i, j] = beta0 / (4*L - 1)

    cs = np.zeros((NumGuesses, ne))
    zs = np.zeros((NumGuesses, ne), dtype=np.complex_)
    for jj in range(NumGuesses):
        X_initial = np.random.rand(L, L)
        X_initial = 10 * X_initial / np.linalg.norm(X_initial)
        _, z_initial, _, _, _ = expand_fb(X_initial, ne)
        c_initial = np.real(T @ z_initial)
        cs[jj, :] = c_initial
        zs[jj, :] = z_initial
        
    y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma*(N/L)**2, T, seed=sd)
    y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
    yy = np.zeros((N, N, 1))
    yy[ :, :, 0] = y
    
    # %% Autocorrelation Analysis
    start_ac_calc = time.time()
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
    ac_calc_time = time.time() - start_ac_calc    
    
    del yy
    
    for jj in range(NumGuesses):
        startac = time.time()
        X_est_ac = Utils.optimization_funcs_rot.optimize_2d(np.concatenate((np.reshape(gamma_initial, (1,)), cs[jj, :])), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1) 
        stopac = time.time() - startac
        c_est_ac = X_est_ac.x[1:]
        z_est_ac = T.H @ c_est_ac
        est_err_coeffs_ac = min_err_coeffs(z, z_est_ac, kvals)
        errs_ac[:, jj] = est_err_coeffs_ac[0]
        costs_ac[:, jj] = X_est_ac.fun
        times_ac[:, jj] = ac_calc_time + stopac
    # %% calculations
    for (k, K) in enumerate(Ks):
        # %% Expectation-maximization
        start_split = time.time()
        Nd = int((N / L) ** 2)
        Mss = [np.array_split(Mm, np.sqrt(Nd), axis=1) for Mm in np.array_split(y, np.sqrt(Nd), axis=0)]
        Mss = [item for sublist in Mss for item in sublist]
        Ms = np.zeros((L, L, Nd))
        for kx, Mm in enumerate(Mss):
            Ms[ :, :, kx] = Mm
        time_split = time.time() - start_split
        for jj in range(NumGuesses):
            startEM = time.time()
            z_est_EM, rho_est_EM, log_likelihood_EM = EM(Ms, zs[jj, :], rho_init, L, K, Nd, B, Bk, kvals, nu, sigma2)
            stopEM = time.time() - startEM
            est_err_coeffs_EM = min_err_coeffs(z, z_est_EM, kvals)
            errs_EM[k, jj] = est_err_coeffs_EM[0]
            costs_EM[k, jj] = log_likelihood_EM
            times_EM[k, jj] = time_split + stopEM
        np.save(f'DATA/errs_EM_sd{sd}_k{K}.npy', errs_EM)
        np.save(f'DATA/costs_EM_sd{sd}_k{K}.npy', costs_EM)
        np.save(f'DATA/times_EM_sd{sd}_k{K}.npy', times_EM)
        np.save(f'DATA/errs_ac_sd{sd}_k{K}.npy', errs_ac)
        np.save(f'DATA/costs_ac_sd{sd}_k{K}.npy', costs_ac)
        np.save(f'DATA/times_ac_sd{sd}_k{K}.npy', times_ac)
        print(f'finished iter #{sd}, SNR = {SNR}. Error for EM = {errs_EM[k, 0]}; error for autocorrelation analysis = {errs_ac[k, 0]}.')

    np.save(f'DATA/errs_EM_sd{sd}.npy', errs_EM)
    np.save(f'DATA/costs_EM_sd{sd}.npy', costs_EM)
    np.save(f'DATA/times_EM_sd{sd}.npy', times_EM)
    np.save(f'DATA/errs_ac_sd{sd}.npy', errs_ac)
    np.save(f'DATA/costs_ac_sd{sd}.npy', costs_ac)
    np.save(f'DATA/times_ac_sd{sd}.npy', times_ac)
    print(f'iter #{sd} finished')
    return errs_EM, costs_EM, times_EM, errs_ac, costs_ac, times_ac
