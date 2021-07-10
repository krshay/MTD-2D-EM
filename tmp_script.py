# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:18:47 2021

@author: Shay Kreymer
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from Utils.fb_funcs import expand_fb, calcT
from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots_discrete
from Utils.funcs_calc_moments import M2_2d, M3_2d
from Utils.psf_tsf_funcs import full_psf_2d, full_tsf_2d, makeExtraMat, maketsfMat
from Utils.EM_funcs import EM, rearangeB, PsiPsi

plt.close("all")

np.random.seed(1)
rng = np.random.default_rng()
F = np.random.rand(5, 5)
L = np.shape(F)[0]
F = 100 * F / np.linalg.norm(F)
W = 2*L - 1 # L for arbitrary spacing distribution, 2*L-1 for well-separated
K = 1 # discretization of rotations

gamma = 0.04
N = 500
N = (N // L) * L
ne = 50
B, z, roots, kvals, nu = expand_fb(F, ne)
T = calcT(nu, kvals)
BT = B @ T.H
c = np.real(T @ z)
z = T.H@c
Frec = np.reshape(np.real(B @ z), np.shape(F))
Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
for i in range(nu):
    Bk[ :, :, i] = np.fft.fft2(np.pad(np.reshape(B[ :, i], (L, L)), L//2))

M_clean, s, locs = generate_clean_micrograph_2d_rots_discrete(c, kvals, Bk, W, L, N, gamma*(N/L)**2, T, K, seed=100)

gamma = s[0]*(L/N)**2

sigma2 = 10

M = M_clean + rng.normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(M_clean))

Nd = int((N / L) ** 2)
Mss = [np.array_split(Mm, np.sqrt(Nd), axis=1) for Mm in np.array_split(M, np.sqrt(Nd), axis=0)]
Mss = [item for sublist in Mss for item in sublist]
Ms = np.zeros((L, L, Nd))
for idx, Mm in enumerate(Mss):
    Ms[ :, :, idx] = Mm
    
Mss_clean = [np.array_split(Mm, np.sqrt(Nd), axis=1) for Mm in np.array_split(M_clean, np.sqrt(Nd), axis=0)]
Mss_clean = [item for sublist in Mss_clean for item in sublist]
Ms_clean = np.zeros((L, L, Nd))
for idx, Mm in enumerate(Mss_clean):
    Ms_clean[ :, :, idx] = Mm
    
M_empty = np.sum(Ms_clean, axis=(0,1))
beta = np.sum(M_empty == 0) / Nd
    
F_init = np.random.rand(L, L)
F_init = 100 * F_init / np.linalg.norm(F_init)
_, z_init, _, _, _ = expand_fb(F_init, ne)

rho_init = np.zeros((2*L, 2*L))
for i in range(2*L):
    for j in range(2*L):
        rho_init[i, j] = (1 - beta) / (2*L - 1)**2
        if i == L or j == L:
            rho_init[i, j] = beta / (4*L - 1)

z, rho = EM(Ms, z_init, rho_init, L, K, Nd, B, Bk, roots, kvals, nu, sigma2)

