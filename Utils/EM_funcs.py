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
            print(iPhi)
            for l in Ls:
                pM_k[iPhi, l[0], l[1], :] = np.real(pMm_l_phi_z(Ms, l, phi, z_k, kvals, Bk, L, sigma2, Nd))
        # pM_k = pM_k / np.sum(pM_k, axis=(0, 1, 2))
        pl_phi_k = pM_k * np.expand_dims(np.moveaxis(np.expand_dims(rho_k, 2),  [0,1,2], [1,2,0]), 3)
        pl_phi_k = pl_phi_k / np.sum(pl_phi_k, axis=(0, 1, 2))
        # pl_phi_k[np.isnan(pl_phi_k)] = 0 ## CHECK!!!
        
        rho_updated = rho_step(rho_k, pl_phi_k, Nd)
        z_updated = z_step(z_k, pl_phi_k, Ms, B, L, K, Nd, nu, roots, kvals, PsiPsi_vals)
        z_k = z_updated
        rho_k = rho_updated
    return z_k, rho_k
        
def z_step(z_k, pl_phi_k, Ms, B, L, K, Nd, nu, roots, kvals, PsiPsi_vals):
    Phi = calc_Phi(K)
    Ls = calc_shifts(L)
    nume = np.zeros((nu, ), dtype=np.complex_)
    deno = np.zeros((nu, nu), dtype=np.complex_)
    for (iPhi, phi) in enumerate(Phi):
        print(iPhi)
        for l in Ls:
            nume += np.diag(np.exp(-1j * kvals * phi)) @ (pl_phi_k[iPhi, l[0], l[1], :] @ \
                np.sum(np.repeat(Ms[:, :, :, np.newaxis], nu, axis=3) * np.repeat(CTZB(B, l, L)[ :, :, np.newaxis, :], np.shape(Ms)[2], axis=(2)), axis=(0, 1)))
            deno += np.diag(np.exp(-1j * kvals * phi)) * np.sum(pl_phi_k[iPhi, l[0], l[1], :]) * PsiPsi_vals[l[0], l[0], :, :]
    return np.linalg.inv(deno) @ nume

def rho_step(rho_k, pl_phi_k, Nd):
    return np.sum(pl_phi_k, axis=(0, 3)) / Nd

def pMm_l_phi_z(Ms, l, phi, z, kvals, Bk, L, sigma2, Nd):
    F_phi = rot_img_freq(phi, z, kvals, Bk, L)
    CTZF_phi = CTZ(F_phi, l, L)
    return np.exp(- np.sum((Ms - np.expand_dims(CTZF_phi, 2)) ** 2, axis=(0, 1)) / (2 * sigma2))

def CTZ(F, l, L):
    TZF = np.zeros((2*L, 2*L), dtype=np.complex_)
    TZF[tuple(np.meshgrid(np.arange(l[0], l[0] + L) % (2*L), np.arange(l[1], l[1] + L) % (2*L)))] = F
    CTZF = TZF[ :L, :L]
    return CTZF

def CTZB(B, l, L):
    TZB = np.zeros((2*L, 2*L, np.shape(B)[2]), dtype=np.complex_)
    TZB[tuple(np.meshgrid(np.arange(l[0], l[0] + L) % (2*L), np.arange(l[1], l[1] + L) % (2*L), np.arange(np.shape(B)[2])))] = B
    CTZB = TZB[ :L, :L, :]
    return CTZB

def calc_Phi(K):
    return np.linspace(0, 2 * np.pi, K, endpoint=False)

def calc_shifts(L):
    return list(itertools.product(np.arange(2*L), np.arange(2*L)))

# def calc_corresponding_kvals(kvals):
#     corresponding_kvals = np.zeros_like(kvals)
#     for ii in range(len(kvals)):
#         if kvals[ii] > 0:
#             corresponding_kvals[ii] = ii + 1
#         elif kvals[ii] < 0:
#             corresponding_kvals[ii] = ii - 1
#         else:
#             corresponding_kvals[ii] = ii
#     return corresponding_kvals

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
        
    