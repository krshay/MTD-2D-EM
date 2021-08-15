# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 14:56:46 2021

@author: Shay Kreymer
"""

import numpy as np
import itertools
import scipy.special as spl
from Utils.fb_funcs import rot_img_freq

def EM(Ms, c_init, rho_init, L, K, Nd, B, Bk, roots, kvals, nu, sigma2, T, Gamma):
    z_init = T.H @ c_init
    c_k = c_init
    z_k = z_init
    rho_k = rho_init
    Phi = calc_Phi(K)
    Ls = calc_shifts(L)
    pM_k = np.zeros((K, 2*L, 2*L, Nd)) # np.zeros((K, 1, 1, Nd)) # 
    S = np.zeros((K, 2*L, 2*L, Nd)) # np.zeros((K, 1, 1, Nd)) # 
    B = rearangeB(B)
    PsiPsi_vals = PsiPsi(B, L, K, nu, kvals)
    log_likelihood_prev = 0
    for ii in range(5):
        for (iPhi, phi) in enumerate(Phi):
            for l in Ls:
                S[iPhi, l[0], l[1], :] = np.real(pMm_l_phi_z(Ms, l, phi, z_k, kvals, Bk, L, sigma2, Nd))
        S_normalized = S - np.min(S, axis=(0, 1, 2))
        pM_k = np.exp(-S_normalized / (2 * sigma2))
        pM_k_likelihood = np.exp(-S / (2 * sigma2))
        # pM_k_likelihood = pM_k / np.sum(pM_k, axis=(0, 1, 2))
        pM_k = pM_k / np.sum(pM_k, axis=(0, 1, 2))
        likelihood_func_l_phi = np.einsum("Pijm,ij->Pijm", pM_k, rho_k)
        likelihood_func_temp = np.einsum("kijm,ij->kijm", pM_k_likelihood, rho_k)
        pl_phi_k = likelihood_func_l_phi / np.sum(likelihood_func_l_phi, axis=(0, 1, 2))
        log_likelihood = np.sum(np.log(np.sum(likelihood_func_temp, axis=(0, 1, 2))))
        print(f'log-likelihood = {log_likelihood}')
        # rho_updated = rho_step(rho_k, pl_phi_k, Nd)
        z_updated = z_step(c_k, pl_phi_k, Ms, B, L, K, Nd, nu, roots, kvals, sigma2, PsiPsi_vals, T, Gamma)
        # c_k = c_updated
        # z_k = T.H @ c_updated
        print(z_k[0])
        z_k = z_updated
        if ii > 0:
            if log_likelihood < log_likelihood_prev:
                print("Something is wrong.")
        log_likelihood_prev = log_likelihood

        # rho_k = rho_updated
    return z_k, pM_k
        
def z_step(c_k, pl_phi_k, Ms, B, L, K, Nd, nu, roots, kvals, sigma2, PsiPsi_vals, T, Gamma):
    Phi = calc_Phi(K)
    Ls = calc_shifts(L)
    y = np.zeros((nu, ), dtype=np.complex_)
    A = np.zeros((nu, nu), dtype=np.complex_)
    for (iPhi, phi) in enumerate(Phi):
        Bphi = np.einsum("ijk,kl->ijk", B, np.diag(np.exp(1j * kvals * phi)))
        for l in Ls:
            y += (1/sigma2) * np.einsum("k,ki->i", pl_phi_k[iPhi, l[0], l[1], :], \
                np.einsum("ijm,ijn->mn", Ms, CTZB(Bphi, l, L))) #### CHECK
            A += (1/sigma2) * np.sum(pl_phi_k[iPhi, l[0], l[1], :]) * PsiPsi_vals[iPhi, l[0], l[1], :, :]
    # Afinal = A + np.linalg.inv(Gamma) @ T
    return np.linalg.inv(A) @ y

def rho_step(rho_k, pl_phi_k, Nd):
    return np.sum(pl_phi_k, axis=(0, 3)) / Nd

def pMm_l_phi_z(Ms, l, phi, z, kvals, Bk, L, sigma2, Nd):
    F_phi = rot_img_freq(phi, z, kvals, Bk, L)
    CTZF_phi = CTZ(F_phi, l, L)
    S = np.sum((Ms - np.expand_dims(CTZF_phi, 2)) ** 2, axis=(0, 1))
    return S

def CTZ(F, l, L):
    ZF = np.zeros((2*L, 2*L), dtype=np.complex_)
    ZF[ :L, :L] = F
    TZF = np.roll(ZF, l, axis=(0, 1))
    CTZF = TZF[ :L, :L]
    return CTZF

def CTZB(B, l, L):
    ZB = np.zeros((2*L, 2*L, np.shape(B)[2]), dtype=np.complex_)
    ZB[ :L, :L, :] = B
    TZB = np.roll(ZB, l, axis=(0, 1))
    CTZB = TZB[ :L, :L, :]
    return CTZB

def calc_Phi(K):
    return np.linspace(0, 2 * np.pi, K, endpoint=False)

def calc_shifts(L):
    return list(itertools.product(np.arange(2*L), np.arange(2*L)))

def rearangeB(B):
    return np.reshape(B, (int(np.sqrt(np.shape(B)[0])), int(np.sqrt(np.shape(B)[0])), np.shape(B)[1]))

# def calcBphi(B, phi, kvals):
#     Bphi = np.zeros(np.shape(B), dtype=np.complex_)
#     for i in range(np.shape(B)[2]):
#         Bphi[ :, :, i] = B[ :, :, i] * np.exp(-1j * kvals[i] * phi)
#     return Bphi

def PsiPsi(B, L, K, nu, kvals):
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
