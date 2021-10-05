# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:27:49 2020

@author: Shay Kreymer
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from Utils.calc_err_size import calc_err_size_both

plt.close("all")

if __name__ == '__main__':
    # %% Preliminary definitions
    Niters = 40
    L = 5
    ne = 10
    Nsizes = 10
    sizes = np.logspace(np.log10(150), np.log10(1500), Nsizes).astype(int)
    
    SNR = 5
    gamma = 0.04
    K = 16
    num_cpus = mp.cpu_count()
    # %% EM and Autocorrelation Analysis
    pool = mp.Pool(num_cpus)
    S = pool.starmap(calc_err_size_both, [[L, ne, sizes, SNR, gamma, K, i] for i in range(Niters)])
    pool.close()
    pool.join() 
    
    errs_EM = np.zeros((Niters, Nsizes))
    likelihood_EM = np.zeros((Niters, Nsizes))
    times_EM = np.zeros((Niters, Nsizes))

    for j in range(Niters):
        errs_EM[j, :] = S[j][0][np.arange(Nsizes), np.argmax(S[j][1], axis=1)]
        likelihood_EM[j, :] = S[j][1][np.arange(Nsizes), np.argmax(S[j][1], axis=1)]
        times_EM[j, :] = S[j][2][np.arange(Nsizes), np.argmax(S[j][1], axis=1)]
    errs_EM_mean = np.mean(errs_EM, 0)
    times_EM_mean = np.mean(times_EM, 0)
    
    errs_ac = np.zeros((Niters, Nsizes))
    costs_ac = np.zeros((Niters, Nsizes))
    times_ac = np.zeros((Niters, Nsizes))

    for j in range(Niters):
        errs_ac[j, :] = S[j][3][np.arange(Nsizes), np.argmin(S[j][4], axis=1)]
        costs_ac[j, :] = S[j][4][np.arange(Nsizes), np.argmin(S[j][4], axis=1)]
        times_ac[j, :] = S[j][5][np.arange(Nsizes), np.argmin(S[j][4], axis=1)]
    errs_ac_mean = np.mean(errs_ac, 0)
    times_ac_mean = np.mean(times_ac, 0)
