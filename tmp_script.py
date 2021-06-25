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
from Utils.EM_funcs import EM

plt.close("all")

np.random.seed(100)
rng = np.random.default_rng()
F = np.random.rand(15, 15)
L = np.shape(F)[0]
F = F / np.linalg.norm(F)
W = 2*L - 1 # L for arbitrary spacing distribution, 2*L-1 for well-separated
K = 4 # discretization of rotations

gamma = 0.04
N = 400
N = (N // L) * L
ne = 45
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
sigma2 = 0.1
M = M_clean + rng.normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(M_clean))

Nd = int((N / L) ** 2)
Mss = [np.array_split(Mm, np.sqrt(Nd), axis=1) for Mm in np.array_split(M, np.sqrt(Nd), axis=0)]
Mss = [item for sublist in Mss for item in sublist]
Ms = np.zeros((L, L, Nd))
for idx, Mm in enumerate(Mss):
    Ms[ :, :, idx] = Mm