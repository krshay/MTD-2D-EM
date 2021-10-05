# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 15:46:42 2020

@author: Shay Kreymer
"""

import numpy as np
from Utils.funcs_calc_moments_rot import calcS3_x_grad, calcS2_x_grad_notparallel

def calc_acs_grads_rot_notparallel(Bk, z, kvals, L, k1_map, map3):
    """ Calclulate of all needed autocorrelations and gradients.

    Args:
        Bk: matrix that maps from the expansion coefficients to the approximated image, in the freuency domain
        z: vector of the expansion coefficients of the target image
        kvals: vector of frequencies
        L: diameter of the target image
        k1_map: list of frequencies
        map3: array of triplets of frequencies

    Returns:
        autocorrelations and gradients, second- and third-order autocorrelations, autocorrelations with a neighbor, and autocorrelations of a triplet of target images
    """
    kmax = np.max(kvals)
    Nmax = 6*kmax
    S3_x, gS3_x = calcS3_x_grad(L, Nmax, Bk, z, kvals, map3)

    Nmax = 4*kmax
    S2_x, gS2_x = calcS2_x_grad_notparallel(L, Nmax, Bk, z, kvals, k1_map)

    return S2_x, gS2_x, S3_x, gS3_x

def cost_grad_fun_rot_notparallel(Z, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, K, N_mat, k1_map, map3):
    """ Calculate cost and gradient of the optimization problem (40).

    Args:
        Z: vector containing \gamma and c (the real representation of the expansion coefficients of the target image)
        Bk: matrix that maps from the expansion coefficients to the approximated image, in the freuency domain
        T: matrix that maps from the real representation to the complex representation of the expansion coefficients
        kvals: vector of frequencies
        M1_y: the first-order autocorrelation of the measurement
        M2_y: the second-order autocorrelations of the measurement, of size L * L
        M3_y: the third-order autocorrelations of the measurement, of size L * L * L * L
        sigma2: the variance of the noise
        L: diameter of the target image
        K: number of target images' types; may be utilized for heterogeneity
        N_mat: sparse matrix of the influence of the noise
        k1_map: list of frequencies
        map3: array of triplets of frequencies

    Returns:
        cost value
        gradient w.r.t. \gamma and c
    """
    gamma = Z[:K]
    c = Z[K:]
    z = T.H@c

    S2_x, gS2_x, S3_x, gS3_x = calc_acs_grads_rot_notparallel(Bk, z, kvals, L, k1_map, map3)
    w1 = 1/2 
    w2 = 1/(2*L**2)
    w3 = 1/(2*L**4)

    # %% First-order moment
    S1 = np.real(np.sum(np.fft.ifftn(Bk @ z), axis=(0, 1))/(L**2))
    gS1 = np.real(np.sum(np.fft.ifftn(Bk, axes=(0,1)), axis=(0, 1))/(L**2) @ T.H)
    R1 = gamma*S1 - M1_y
    
    # %% Second-order moment
    S2 = np.real(S2_x[ :L, :L])
    gS2 = np.real(np.reshape(gS2_x[ :L, :L, :], (L**2, len(z))) @ T.H)
    R2 = gamma*S2 - M2_y

    # Noise
    R2[0, 0] += sigma2

    # %% Third-order moment
    S3 = np.real(S3_x[ :L, :L, :L, :L])
    gS3 = np.real(np.reshape(gS3_x[ :L, :L, :L, :L, :], (L**4, len(z))) @ T.H)
    
    # Noise
    S3 += S1 * sigma2 * np.reshape(N_mat.toarray(), np.shape(S3))
    gS3 += N_mat @ np.reshape(gS1, (1, len(z))) * sigma2

    R3 = gamma*S3 - M3_y
    
    # %% cost and grad functions calculation
    f1 = w1*R1**2 
    f2 = w2*np.sum(R2**2) 
    f3 = w3*np.sum(R3**2)
    f = f1 + f2 + f3
    
    g_c1 = 2*w1*gamma*gS1*R1
    g_c2 = 2*w2*gamma*R2.flatten()@gS2
    g_c3 = 2*w3*gamma*R3.flatten()@gS3
    g_c = g_c1 + g_c2 + g_c3

    g_gamma1 = 2*w1*S1*R1
    g_gamma2 = 2*w2*np.sum(S2*R2)
    g_gamma3 = 2*w3*np.sum(S3*R3)
    g_gamma = g_gamma1 + g_gamma2 + g_gamma3
    
    return f, np.concatenate((np.reshape(g_gamma, (K,)), g_c))
