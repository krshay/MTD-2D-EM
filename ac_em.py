#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 16:20:00 2021

@author: shaykreymer
"""

import numpy as np
import matplotlib.pyplot as plt
from Utils.fb_funcs import expand_fb, calcT
from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots
from Utils.EM_funcs import EM_parallel, EM, rearangeB, PsiPsi, calcB_CTZs
from Utils.fb_funcs import min_err_coeffs
from Utils.funcs_calc_moments import M2_2d, M3_2d
import Utils.optimization_funcs_rot

from Utils.calcM3_parallel import calcM3_parallel_micrographs

import time

plt.close("all")

if __name__ == '__main__':
    np.random.seed(10)
    F = np.random.rand(5, 5)
    L = np.shape(F)[0]
    F = 10 * F / np.linalg.norm(F)
    W = 2*L - 1 # L for arbitrary spacing distribution, 2*L-1 for well-separated
    K = 16 # discretization of rotations
    
    ne = 10
    B, z, roots, kvals, nu = expand_fb(F, ne)
    T = calcT(nu, kvals)
    c = np.real(T @ z)
    
    Frec = np.reshape(np.real(B @ z), np.shape(F))
    Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
    for i in range(nu):
        Bk[ :, :, i] = np.fft.fft2(np.pad(np.reshape(B[ :, i], (L, L)), L//2))
    
    F_init = np.random.rand(L, L)
    F_init = np.linalg.norm(F) * F_init / np.linalg.norm(F_init)
    
    _, z_init, _, _, _ = expand_fb(F_init, ne)
    c_initial = np.real(T @ z_init)
    
    SNR = 5
    sigma2 = np.linalg.norm(F)**2 / (L**2 *SNR)
    
    gamma = 0.04
    N = 1000
    N = (N // L) * L
    M_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma*(N/L)**2, T, seed=100)

    M = M_clean + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(M_clean))

    Nd = int((N / L) ** 2)
    Mss = [np.array_split(Mm, np.sqrt(Nd), axis=1) for Mm in np.array_split(M, np.sqrt(Nd), axis=0)]
    Mss = [item for sublist in Mss for item in sublist]
    Ms = np.zeros((L, L, Nd))
    for idx, Mm in enumerate(Mss):
        Ms[ :, :, idx] = Mm
       
    gamma_initial = 0.036
    M_initial, _, _ = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma_initial*(N/L)**2, T, seed=1000)

    Mss_clean = [np.array_split(Mm, np.sqrt(Nd), axis=1) for Mm in np.array_split(M_initial, np.sqrt(Nd), axis=0)]
    Mss_clean = [item for sublist in Mss_clean for item in sublist]
    Ms_clean = np.zeros((L, L, Nd))
    for idx, Mm in enumerate(Mss_clean):
        Ms_clean[ :, :, idx] = Mm
        
    M_empty = np.sum(Ms_clean, axis=(0,1))
    beta0 = np.sum(M_empty == 0) / Nd
    
    rho_init = np.zeros((2*L, 2*L))
    for i in range(2*L):
        for j in range(2*L):
            rho_init[i, j] = (1 - beta0) / (2*L - 1)**2
            if i == L or j == L:
                rho_init[i, j] = beta0 / (4*L - 1)
    
    # %% Autocorrelation Analysis
    yy = np.zeros((N, N, 1))
    yy[ :, :, 0] = M

    M1_y = np.mean(yy)
    
    M2_y = np.zeros((L, L))
    for i1 in range(L):
        for j1 in range(L):
            M2_y[i1, j1] = M2_2d(yy, (i1, j1))
    
    M3_y = np.zeros((L, L, L, L))
    for i1 in range(L):
        print(i1)
        for j1 in range(L):
            for i2 in range(L):
                for j2 in range(L):
                    M3_y[i1, j1, i2, j2] = M3_2d(yy, (i1, j1), (i2, j2))
    
    numGuesses = 1
    costs_ac = np.zeros((numGuesses, ))
    z_ests_ac = np.zeros((numGuesses, nu), dtype=np.complex_)
    gamma_ests_ac = np.zeros((numGuesses, ))
    errs_ac = np.zeros((numGuesses, ))
    for g in range(numGuesses):

        st = time.time()
        X_est_ac = Utils.optimization_funcs_rot.optimize_2d(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1) 
        time_ac = time.time() - st
        c_est_ac = X_est_ac.x[1:]
        z_est_ac = T.H @ c_est_ac
        gamma_est_ac = X_est_ac.x[0]
        est_err_coeffs_ac = min_err_coeffs(z, z_est_ac, kvals)
        err_ac = est_err_coeffs_ac[0]
        errs_ac[g] = err_ac
        print(f'done with autocorrelation analysis. N = {N}. error of {err_ac}, objective function of {X_est_ac.fun}')
        costs_ac[g] = X_est_ac.fun
        z_ests_ac[g, :] = z_est_ac
        gamma_ests_ac[g] = gamma_est_ac
    
    # # %% Approximate EM
    # Bs = rearangeB(B)
    # PsiPsi_vals = PsiPsi(Bs, L, K, nu, kvals)
    # BCTZs = calcB_CTZs(B, K, L, kvals)
    # start = time.time()
    # z_est_parallel, rho_est, log_likelihood, numiters = EM_parallel(Ms, z_init, rho_init, L, K, Nd, B, Bk, kvals, nu, sigma2, BCTZs, PsiPsi_vals, z)
    # print(time.time() - start)
    

