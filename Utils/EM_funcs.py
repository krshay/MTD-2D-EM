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
    corresponding_kvals = calc_corresponding_kvals(kvals)
    for _ in range(5):
        z_updated = z_step(z_k, Ms, rho_k, L, K, Nd, B, Bk, roots, kvals, corresponding_kvals, nu, sigma2)
        rho_updated = rho_step(rho_k, Ms, z_k, L, K, Nd, kvals, Bk, sigma2)
        z_k = z_updated
        rho_k = rho_updated
    return z_k, rho_k
        
def z_step(z_k, Ms, rho_k, L, K, Nd, B, Bk, roots, kvals, corresponding_kvals, nu, sigma2):
    z_updated = np.zeros_like(z_k)
    Phi = calc_Phi(K)
    Ls = calc_shifts(L)
    for ii in range(nu):
        print(ii)
        nume = np.zeros((Nd, ), dtype=np.complex_)
        deno = np.zeros((Nd, ), dtype=np.complex_)
        root = roots[ii]
        kval = kvals[ii]
        for phi in Phi:
            for l in Ls:
                nume += np.exp(-1j * kval * phi) * pMm_l_phi_z(Ms, l, phi, z_k, kvals, Bk, L, sigma2, Nd) * \
                    rho_k[l] * np.sum(Ms * np.expand_dims(CTZ(np.reshape(B[: , corresponding_kvals[ii]], (L, L)), l, L), 2), axis=(0,1))
                deno += np.exp(-1j * kval * phi) * pMm_l_phi_z(Ms, l, phi, z_k, kvals, Bk, L, sigma2, Nd)
        z_updated[ii] = np.sum(nume / deno) / (np.pi * (L // 2 + 1) ** 2 * spl.jv(np.abs(kval) + 1, root)**2)
    return z_updated

def rho_step(rho_k, Ms, z_k, L, K, Nd, kvals, Bk, sigma2):
    rho_updated = np.zeros_like(rho_k)
    Phi = calc_Phi(K)
    Ls = calc_shifts(L)
    for l in Ls:
        for phi in Phi:
            rho_updated[l[0], l[1]] += pMm_l_phi_z(Ms, l, phi, z_k, kvals, Bk, L, sigma2, Nd) * rho_k[l[0], l[1]] / Nd
    return rho_updated

def pMm_l_phi_z(Ms, l, phi, z, kvals, Bk, L, sigma2, Nd):
    F_phi = rot_img_freq(phi, z, kvals, Bk, L)
    CTZF_phi = CTZ(F_phi, l, L)
    return np.exp(- np.sum((Ms - np.expand_dims(CTZF_phi, 2)) ** 2, axis=(0, 1)) / (2 * sigma2))

def CTZ(F, l, L):
    TZF = np.zeros((2*L, 2*L), dtype=np.complex_)
    TZF[tuple(np.meshgrid(np.arange(l[0], l[0] + L) % (2*L), np.arange(l[1], l[1] + L) % (2*L)))] = F
    CTZF = TZF[ :L, :L]
    return CTZF

def calc_Phi(K):
    return np.linspace(0, 2 * np.pi, K, endpoint=False)

def calc_shifts(L):
    return list(itertools.product(np.arange(2*L), np.arange(2*L)))

def calc_corresponding_kvals(kvals):
    corresponding_kvals = np.zeros_like(kvals)
    for ii in range(len(kvals)):
        if kvals[ii] > 0:
            corresponding_kvals[ii] = ii + 1
        elif kvals[ii] < 0:
            corresponding_kvals[ii] = ii - 1
        else:
            corresponding_kvals[ii] = ii
    return corresponding_kvals