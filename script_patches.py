# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 12:13:50 2021

@author: Shay Kreymer
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from Utils.fb_funcs import expand_fb, calcT
from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots_discrete
from Utils.funcs_calc_moments import M2_2d, M3_2d
from Utils.psf_tsf_funcs import full_psf_2d, full_tsf_2d, makeExtraMat, maketsfMat
from Utils.EM_funcs import EM, rearangeB, PsiPsi, CTZ, calc_shifts, calc_Phi
from Utils.fb_funcs import rot_img_freq

plt.close("all")

np.random.seed(1)
F = np.random.rand(5, 5)
L = np.shape(F)[0]
F = F / np.linalg.norm(F)
W = 2*L - 1 # L for arbitrary spacing distribution, 2*L-1 for well-separated
K = 4 # discretization of rotations

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

SNR = 100

sigma2 = 1 / (L**2 *SNR)

Nd = int((N / L) ** 2)

beta_init = 0.85
rho = np.zeros((2*L, 2*L))

Ls = calc_shifts(L)
Phis = calc_Phi(K)
Ms_clean = np.zeros((L, L, Nd))

for i in range(int(Nd * (1 - beta_init))):
    l_i = Ls[np.random.randint(0, len(Ls))]
    F_phi = rot_img_freq(Phis[np.random.randint(0, K)], z, kvals, Bk, L)
    Ms_clean[ :, :, i] = np.real(CTZ(F_phi, l_i, L))
    rho[l_i[0], l_i[1]] = rho[l_i[0], l_i[1]] + 1 / Nd
    
beta = np.sum(np.sum(Ms_clean, axis=(0, 1)) == 0) / Nd
rho[ L, :] = beta / (4*L - 1)
rho[ :, L] = beta / (4*L - 1)

Ms = Ms_clean + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(Ms_clean))

z_est, rho_est = EM(Ms, z, rho, L, K, Nd, B, Bk, roots, kvals, nu, sigma2)

F_init = np.random.rand(L, L)
F_init = F_init / np.linalg.norm(F_init)
_, z_init, _, _, _ = expand_fb(F_init, ne)

z_est, rho_est = EM(Ms, z_init, rho, L, K, Nd, B, Bk, roots, kvals, nu, sigma2)

