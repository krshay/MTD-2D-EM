# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 14:56:46 2021

@author: Shay Kreymer
"""

import numpy as np
import itertools
from Utils.fb_funcs import rot_img_freq
import multiprocessing as mp

from Utils.fb_funcs import min_err_coeffs

import time

def EM(Ms, z_init, rho_init, L, K, Nd, B, Bk, kvals, nu, sigma2):
    """ Approximate EM algorithm (Algorithm 1 in the paper).

    Args:
        Ms: patches
        z_init: initial guess for the vector of coefficients
        rho_init: initial guess for rho
        L: image diameter
        K: angular search space size
        Nd: number of patches
        B: matrix that maps from the expansion coefficients to the approximated image
        Bk: matrix that maps from the expansion coefficients to the approximated image, in the freuency domain
        kvals: vector of frequencies
        nu: modified number of expansion coefficients
        sigma2: noise variance

    Returns:
        z_k: estimated vector of coefficients
        rho_k: estimated rho
        log_likelihood: final log likelihood value
    """
    z_k = z_init
    rho_k = rho_init
    Phi = calc_Phi(K)
    Ls = calc_shifts(L)
    pM_k = np.zeros((K, 2*L, 2*L, Nd))
    S = np.zeros((K, 2*L, 2*L, Nd))
    B = rearangeB(B)
    PsiPsi_vals = PsiPsi(B, L, K, nu, kvals)
    log_likelihood_prev = 0
    count = 1
    while True:
        for (iPhi, phi) in enumerate(Phi):
            for l in Ls:
                S[iPhi, l[0], l[1], :] = np.real(pMm_l_phi_z(Ms, l, phi, z_k, kvals, Bk, L))
        S_normalized = S - np.min(S, axis=(0, 1, 2))
        pM_k = np.exp(-S_normalized / (2 * sigma2))
        pM_k_likelihood = np.exp(-S / (2 * sigma2))
        pM_k = pM_k / np.sum(pM_k, axis=(0, 1, 2))
        likelihood_func_l_phi = np.einsum("Pijm,ij->Pijm", pM_k, rho_k)
        pl_phi_k = likelihood_func_l_phi / np.sum(likelihood_func_l_phi, axis=(0, 1, 2))
        with np.errstate(divide='ignore'):
            log_likelihood = np.sum(np.log(np.sum(np.einsum("kijm,ij->kijm", pM_k_likelihood, rho_k), axis=(0, 1, 2))))
        z_updated = z_step(pl_phi_k, Ms, B, L, K, nu, kvals, sigma2, PsiPsi_vals)
        rho_updated = rho_step(pl_phi_k, Nd)
        z_k = z_updated
        rho_k = rho_updated
        if (not np.isinf(log_likelihood) and count != 1 and log_likelihood - log_likelihood_prev < 0.5) or count > 9:
            break
        log_likelihood_prev = log_likelihood
        count += 1
    return z_k, rho_k, log_likelihood
        
def z_step(pl_phi_k, Ms, B, L, K, nu, kvals, sigma2, PsiPsi_vals):
    """ Updating the vector of coefficients.

    Args:
        pl_phi_k: the function p(l, phi) in iteration k
        Ms: patches
        B: maps from the expansion coefficients to the approximated image
        L: image diameter
        K: angular search space size
        nu: modified number of expansion coefficients
        kvals: vector of frequencies
        sigma2: noise variance
        PsiPsi_vals: data for the multiplication of different Dirichlet-Laplacian eigenfunctions

    Returns:
        updated vector of coefficients
    """
    Phi = calc_Phi(K)
    Ls = calc_shifts(L)
    y = np.zeros((nu, ), dtype=np.complex_)
    A = np.zeros((nu, nu), dtype=np.complex_)
    for (iPhi, phi) in enumerate(Phi):
        Bphi = np.einsum("ijk,kl->ijk", B, np.diag(np.exp(1j * kvals * phi)))
        for l in Ls:
            y += (1/sigma2) * np.einsum("k,ki->i", pl_phi_k[iPhi, l[0], l[1], :], \
                np.einsum("ijm,ijn->mn", Ms, CTZB(Bphi, l, L)))
            A += (1/sigma2) * np.sum(pl_phi_k[iPhi, l[0], l[1], :]) * PsiPsi_vals[iPhi, l[0], l[1], :, :]
    return np.linalg.inv(A) @ y

def rho_step(pl_phi_k, Nd):
    """ Updating rho.

    Args:
        pl_phi_k: the function p(l, phi) in iteration k
        Nd: number of patches
        
    Returns:
        updated rho
    """
    return np.sum(pl_phi_k, axis=(0, 3)) / Nd

def pMm_l_phi_z(Ms, l, phi, z, kvals, Bk, L):
    """ Calculating part of the function p(M|l, phi, alpha).

    Args:
        Ms: patches
        l: a shift
        phi: a rotation
        z: vector of coefficients
        kvals: vector of frequencies
        Bk: maps from the expansion coefficients to the approximated image, in the freuency domain
        L: image diameter
        
    Returns:
        part of the function p(M|l, phi, alpha)
    """
    F_phi = rot_img_freq(phi, z, kvals, Bk, L)
    CTZF_phi = CTZ(F_phi, l, L)
    S = np.sum((Ms - np.expand_dims(CTZF_phi, 2)) ** 2, axis=(0, 1))
    return S

def CTZ(F, l, L):
    """ Shifts, pads and crops an image.

    Args:
        F: an image
        l: a shift
        L: image diameter
        
    Returns:
        shifted, padded and cropped an image
    """
    ZF = np.zeros((2*L, 2*L), dtype=np.complex_)
    ZF[ :L, :L] = F
    TZF = np.roll(ZF, l, axis=(0, 1))
    CTZF = TZF[ :L, :L]
    return CTZF

def CTZB(B, l, L):
    """ Shifts, pads and crops the basis functions.

    Args:
        B: the basis functions
        l: a shift
        L: image diameter
        
    Returns:
        shifted, padded and cropped basis functions
    """
    ZB = np.zeros((2*L, 2*L, np.shape(B)[2]), dtype=np.complex_)
    ZB[ :L, :L, :] = B
    TZB = np.roll(ZB, l, axis=(0, 1))
    CTZB = TZB[ :L, :L, :]
    return CTZB

def calc_Phi(K):
    """ Calculates the angular search space for EM.

    Args:
        K: size of the angular search space
        
    Returns:
        the angular search space for EM
    """
    return np.linspace(0, 2 * np.pi, K, endpoint=False)

def calc_shifts(L):
    """ Calculates the set of shifts for EM.

    Args:
        L: image diameter
        
    Returns:
        the set of shifts for EM
    """
    return list(itertools.product(np.arange(2*L), np.arange(2*L)))

def rearangeB(B):
    """ Rearranges the basis functions.

    Args:
        B: the basis functions
        
    Returns:
        the basis functions, rearranged
    """
    return np.reshape(B, (int(np.sqrt(np.shape(B)[0])), int(np.sqrt(np.shape(B)[0])), np.shape(B)[1]))

def PsiPsi(B, L, K, nu, kvals):
    """ Multiplication of the Dirichlet-Laplacian eigenfunctions.

    Args:
        B: the basis functions
        L: image diamter
        K: size of the angular search space
        nu: modified number of expansion coefficients
        kvals: vector of frequencies

    Returns:
        multiplication of the basis functions
    """
    # Here B is from rearangeB
    PsiPsi = np.zeros((K, 2*L, 2*L, nu, nu), dtype=np.complex_)
    Ls = calc_shifts(L)
    Phi = calc_Phi(K)
    for (iPhi, phi) in enumerate(Phi):
        Bphi = np.einsum("ijk,kl->ijk", B, np.diag(np.exp(1j * kvals * phi)))
        for l in Ls:
            B_CTZ = CTZB(Bphi, l, L)
            for i in range(nu):
                for j in range(nu):
                    PsiPsi[iPhi, l[0], l[1], i, j] = np.sum(B_CTZ[ :, :, i] * B_CTZ[ :, :, j])
    return PsiPsi

def EM_parallel(Ms, z_init, rho_init, L, K, Nd, B, Bk, kvals, nu, sigma2, BCTZs, PsiPsi_vals):
    # function EM - parallelized
    num_cpus = mp.cpu_count()
    pool = mp.Pool(num_cpus)
    z_k = z_init
    rho_k = rho_init
    Phi = calc_Phi(K)
    Ls = calc_shifts(L)
    Phi_Ls = set_Phi_Ls(Phi, Ls)
    num_splits = 80
    Phi_Ls_split = np.array_split(Phi_Ls, num_splits)
    Msplit = np.array_split(Ms, num_cpus, 2)
    pM_k = np.zeros((K, 2*L, 2*L, Nd))
    S = np.zeros((K, 2*L, 2*L, Nd))
    B = rearangeB(B)
    log_likelihood_prev = 0
    count = 1
    while True:
        Ss = pool.starmap(calc_pMm_l_phi_z, [[Ms, Phi_Ls_split[i], z_k, kvals, Bk, L] for i in range(num_splits)])
        S = np.reshape(Ss, (K, 2*L, 2*L, Nd))
        S_normalized = S - np.min(S, axis=(0, 1, 2))
        pM_k = np.exp(-S_normalized / (2 * sigma2))
        pM_k_likelihood = np.exp(-S / (2 * sigma2))
        pM_k = pM_k / np.sum(pM_k, axis=(0, 1, 2))
        likelihood_func_l_phi = np.einsum("Pijm,ij->Pijm", pM_k, rho_k)
        pl_phi_k = likelihood_func_l_phi / np.sum(likelihood_func_l_phi, axis=(0, 1, 2))
        with np.errstate(divide='ignore'):
            log_likelihood = np.sum(np.log(np.sum(np.einsum("kijm,ij->kijm", pM_k_likelihood, rho_k), axis=(0, 1, 2))))
        pl_phi_ks = np.array_split(pl_phi_k, num_cpus, 3)
        z_updated = z_step_parallel(pl_phi_ks, Msplit, BCTZs, Phi, Ls, L, K, nu, kvals, sigma2, PsiPsi_vals, pool)
        rho_updated = rho_step(pl_phi_k, Nd)
        z_k = z_updated
        rho_k = rho_updated
        # print(log_likelihood)
        if (not np.isinf(log_likelihood) and count != 1 and log_likelihood - log_likelihood_prev < 0.5) or count > 9:
            break

        log_likelihood_prev = log_likelihood
        count += 1

    return z_k, rho_k, log_likelihood, count

def z_step_parallel(pl_phi_ks, Ms, BCTZs, Phi, Ls, L, K, nu, kvals, sigma2, PsiPsi_vals, pool):
    # function z_step - parallelized
    num_cpus = mp.cpu_count()
    S = pool.starmap(partial_z_step, [[pl_phi_ks[i], Ms[i], BCTZs, Phi, Ls, K, nu, kvals, sigma2, PsiPsi_vals] for i in range(num_cpus)])
    y = np.zeros((nu, ), dtype=np.complex_)
    A = np.zeros((nu, nu), dtype=np.complex_)
    for i in range(num_cpus):
        y += S[i][0]
        A += S[i][1]
    return np.linalg.inv(A) @ y

def partial_z_step(pl_phi_k, Ms, BCTZs, Phi, Ls, K, nu, kvals, sigma2, PsiPsi_vals):
    # parallelization of z_step
    y = np.zeros((nu, ), dtype=np.complex_)
    A = np.zeros((nu, nu), dtype=np.complex_)
    for (iPhi, phi) in enumerate(Phi):
            for l in Ls:
                y += (1/sigma2) * np.einsum("k,ki->i", pl_phi_k[iPhi, l[0], l[1], :], \
                    np.einsum("ijm,ijn->mn", Ms, BCTZs[iPhi, l[0], l[1], :, :, :]))
                A += (1/sigma2) * np.sum(pl_phi_k[iPhi, l[0], l[1], :]) * PsiPsi_vals[iPhi, l[0], l[1], :, :]
    return y, A

def set_Phi_Ls(Phi, Ls):
    """ Product of the set Phi and Ls.

    Args:
        Phi: the set Phi
        Ls: the set Ls

    Returns:
        product of the sets
    """
    return list(itertools.product(Phi, Ls))

def calc_pMm_l_phi_z(Ms, Phi_Ls_split, z_k, kvals, Bk, L):
    # parallelization of calculations of p(Mm| l, phi, alpha)
    S = []
    for i in range(np.shape(Phi_Ls_split)[0]):
        S.append(np.real(pMm_l_phi_z(Ms, Phi_Ls_split[i][1], Phi_Ls_split[i][0], z_k, kvals, Bk, L)))
    return S

def calcB_CTZs(B, K, L, kvals):
    """ Calculation of all shifted and rotated versions of the Dirichlet-Laplacian eigenfunctions.

    Args:
        B: the basis functions
        K: size of the angular search space
        L: image diameter
        kvals: vector of frequencies

    Returns:
        all shifted and rotated versions of the basis functions
    """
    Phi = calc_Phi(K)
    Ls = calc_shifts(L)
    B = rearangeB(B)
    BCTZs = np.zeros((K, 2*L, 2*L, np.shape(B)[0], np.shape(B)[1], np.shape(B)[2]), dtype=np.complex_)
    for iPhi, phi in enumerate(Phi):
        Bphi = np.einsum("ijk,kl->ijk", B, np.diag(np.exp(1j * kvals * phi)))
        for l in Ls:
            BCTZs[iPhi, l[0], l[1], :, :, :] = CTZB(Bphi, l, L)
    return BCTZs
