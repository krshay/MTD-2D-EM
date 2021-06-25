# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:18:47 2021

@author: Shay Kreymer
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from Utils.fb_funcs import expand_fb, calcT
from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots
from Utils.funcs_calc_moments import M2_2d, M3_2d
from Utils.psf_tsf_funcs import full_psf_2d, full_tsf_2d, makeExtraMat, maketsfMat
import Utils.optimization_funcs_rot

plt.close("all")

np.random.seed(100)
X = np.random.rand(5, 5)
L = np.shape(X)[0]
X = X / np.linalg.norm(X)
W = 2*L - 1 # L for arbitrary spacing distribution, 2*L-1 for well-separated

gamma = 0.1
N = 25000
ne = 10
B, z, roots, kvals, nu = expand_fb(X, ne)
T = calcT(nu, kvals)
BT = B @ T.H
c = np.real(T @ z)
z = T.H@c
Xrec = np.reshape(np.real(B @ z), np.shape(X))
Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
for i in range(nu):
    Bk[ :, :, i] = np.fft.fft2(np.pad(np.reshape(B[ :, i], (L, L)), L//2))

y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma*(N/L)**2, T, seed=100)

gamma = s[0]*(L/N)**2
sigma2 = 0.1
y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))

