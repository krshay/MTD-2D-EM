# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:37:57 2021

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
W = L # L for arbitrary spacing distribution, 2*L-1 for well-separated

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

kmax = np.max(kvals)

yy = np.zeros((N, N, 1))
yy[ :, :, 0] = y
M1_y = np.mean(y)

M2_y = np.zeros((L, L))
for i1 in range(L):
    for j1 in range(L):
        M2_y[i1, j1] = M2_2d(yy, (i1, j1))

M3_y = np.zeros((L, L, L, L))
for i1 in range(L):
    for j1 in range(L):
        for i2 in range(L):
            for j2 in range(L):
                M3_y[i1, j1, i2, j2] = M3_2d(yy, (i1, j1), (i2, j2))

psf_true = full_psf_2d(locs, L)
tsf_true = full_tsf_2d(locs, L)

ExtraMat2_true, ExtraMat3_true = makeExtraMat(L, psf_true)
tsfMat_true = maketsfMat(L, tsf_true)

X_initial = np.random.rand(L, L)
X_initial = X_initial / np.linalg.norm(X_initial)

_, z_initial, _, _, _ = expand_fb(X_initial, ne)
c_initial = np.real(T @ z_initial)
# %% initiate from gamma = 0.09
gamma_initial_009 = 0.09

y_initial, _, locs_initial = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma_initial_009*(N/L)**2, T)

# using known PSF and TSF
est_true_009, history_true_009 = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets_with_callback(np.concatenate((np.reshape(gamma_initial_009, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat_true, ExtraMat2_true, ExtraMat3_true, numiters=100, gtol=1e-15)
errs_true_009 = np.abs(np.array(history_true_009) - gamma) / gamma

# using approximated PSF and TSF
psf_approx_009 = full_psf_2d(locs_initial, L)
tsf_approx_009 = full_tsf_2d(locs_initial, L)

ExtraMat2_approx_009, ExtraMat3_approx_009 = makeExtraMat(L, psf_approx_009)
tsfMat_approx_009 = maketsfMat(L, tsf_approx_009)

est_approx_009, history_approx_009 = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets_with_callback(np.concatenate((np.reshape(gamma_initial_009, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat_approx_009, ExtraMat2_approx_009, ExtraMat3_approx_009, numiters=100, gtol=1e-15)
errs_approx_009 = np.abs(np.array(history_approx_009) - gamma) / gamma

# using no PSF and TSF
ExtraMat2_well_separated_009 = scipy.sparse.csr_matrix(np.zeros(np.shape(ExtraMat2_approx_009)))
ExtraMat3_well_separated_009 = scipy.sparse.csr_matrix(np.zeros(np.shape(ExtraMat3_approx_009)))
tsfMat_well_separated_009 = scipy.sparse.csr_matrix(np.zeros(np.shape(tsfMat_approx_009)))

est_well_separated_009, history_well_separated_009 = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets_with_callback(np.concatenate((np.reshape(gamma_initial_009, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat_well_separated_009, ExtraMat2_well_separated_009, ExtraMat3_well_separated_009, numiters=100, gtol=1e-15)
errs_well_separated_009 = 100 * np.abs(np.array(history_well_separated_009) - gamma) / gamma
