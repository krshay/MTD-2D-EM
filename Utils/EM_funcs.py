# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 14:56:46 2021

@author: Shay Kreymer
"""

import numpy as np
import itertools
import scipy.special as spl
from Utils.fb_funcs import rot_img_freq

def EM(Ms, z_init, rho_init, L, K, Nd, B, Bk, roots, kvals, nu, sigma2):
    z_k = z_init
    rho_k = rho_init
    Phi = calc_Phi(K)
    Ls = calc_shifts(L)
    pM_k = np.zeros((K, 2*L, 2*L, Nd))
    B = rearangeB(B)
    PsiPsi_vals = PsiPsi(B, L, nu)
    for _ in range(5):
        for (iPhi, phi) in enumerate(Phi):
            for l in Ls:
                pM_k[iPhi, l[0], l[1], :] = np.real(pMm_l_phi_z(Ms, l, phi, z_k, kvals, Bk, L, sigma2, Nd))
        pM_k = pM_k / np.sum(pM_k, axis=(0, 1, 2))

        likelihood_func_l_phi = np.einsum("kijm,ij->kijm", pM_k, rho_k)
        pl_phi_k = likelihood_func_l_phi / np.sum(likelihood_func_l_phi, axis=(0, 1, 2))
        # pl_phi_k[np.isnan(pl_phi_k)] = 0 ## CHECK!!!
        log_likelihood = np.sum(np.log10(np.sum(likelihood_func_l_phi, axis=(0, 1, 2))))
        print(f'log-likelihood = {log_likelihood}')
        # rho_updated = rho_step(rho_k, pl_phi_k, Nd)
        z_updated = z_step(z_k, pl_phi_k, Ms, B, L, K, Nd, nu, roots, kvals, PsiPsi_vals)
        z_k = z_updated
        print(z_k)
        # rho_k = rho_updated
    return z_k, rho_k
        
def z_step(z_k, pl_phi_k, Ms, B, L, K, Nd, nu, roots, kvals, PsiPsi_vals):
    Phi = calc_Phi(K)
    Ls = calc_shifts(L)
    y = np.zeros((nu, ), dtype=np.complex_)
    A = np.zeros((nu, nu), dtype=np.complex_)
    for (iPhi, phi) in enumerate(Phi):
        for l in Ls:
            y += np.diag(np.exp(-1j * kvals * phi)) @ (pl_phi_k[iPhi, l[0], l[1], :] @ \
                np.einsum("ijm,ijn->mn", Ms, CTZB(B, l, L)))
            A += np.diag(np.exp(-1j * kvals * phi)) @ (np.sum(pl_phi_k[iPhi, l[0], l[1], :]) * PsiPsi_vals[l[0], l[0], :, :])
    return np.linalg.inv(A) @ y

def rho_step(rho_k, pl_phi_k, Nd):
    return np.sum(pl_phi_k, axis=(0, 3)) / Nd

def pMm_l_phi_z(Ms, l, phi, z, kvals, Bk, L, sigma2, Nd):
    F_phi = rot_img_freq(phi, z, kvals, Bk, L)
    CTZF_phi = CTZ(F_phi, l, L)
    S = np.sum((Ms - np.expand_dims(CTZF_phi, 2)) ** 2, axis=(0, 1)) / (2 * sigma2)
    return np.exp(- S)

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

def PsiPsi(B, L, nu):
    # Here B is from rearangeB
    PsiPsi = np.zeros((2*L, 2*L, nu, nu), dtype=np.complex_)
    Ls = calc_shifts(L)
    for l in Ls:
        B_CTZ = CTZB(B, l, L)
        for i in range(nu):
            for j in range(nu):
                PsiPsi[l[0], l[1], i, j] = np.sum(B_CTZ[ :, :, i] * B_CTZ[ :, :, j])
    return PsiPsi
        