# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 17:22:20 2021

@author: Shay Kreymer
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from Utils.calc_err_SNR import calc_err_SNR_both

plt.close("all")

if __name__ == '__main__':
    # %% Preliminary definitions
    Niters = 40
    L = 5
    ne = 10
    NSNRs = 10
    SNRs = np.logspace(-3, 2, NSNRs).astype(int)
    
    N = 1000
    gamma = 0.04
    K = 16
    num_cpus = mp.cpu_count()
    # %% EM and Autocorrelation Analysis
    pool = mp.Pool(num_cpus)
    S = pool.starmap(calc_err_SNR_both, [[L, ne, SNRs, N, gamma, K, i] for i in range(NSNRs)])
    pool.close()
    pool.join() 
    
    errs_EM = np.zeros((Niters, NSNRs))
    times_EM = np.zeros((Niters, NSNRs))

    for j in range(Niters):
        errs_EM[j, :] = S[j][0][np.arange(NSNRs), np.argmin(S[j][1], axis=1)]
        times_EM[j, :] = S[j][2][np.arange(NSNRs), np.argmin(S[j][1], axis=1)]
    errs_EM_mean = np.mean(errs_EM, 0)
    times_EM_mean = np.mean(times_EM, 0)
    
    errs_ac = np.zeros((Niters, NSNRs))
    times_ac = np.zeros((Niters, NSNRs))

    for j in range(Niters):
        errs_ac[j, :] = S[j][3][np.arange(NSNRs), np.argmin(S[j][4], axis=1)]
        times_ac[j, :] = S[j][5][np.arange(NSNRs), np.argmin(S[j][4], axis=1)]
    errs_ac_mean = np.mean(errs_ac, 0)
    times_ac_mean = np.mean(times_ac, 0)

    # %% plots
    plt.close("all")
    with plt.style.context('ieee'):
        fig = plt.figure()
        
        plt.loglog(SNRs, errs_EM_mean, '.-b', label=r'EM')
    
        plt.loglog(SNRs, errs_ac_mean, '.--r', label='Autocorrelation analysis')
        
        plt.legend(loc=(0.5, 0.62))#, fontsize=6)
        
        plt.xlabel('SNR')
        
        plt.ylabel('Mean estimation error')
        fig.tight_layout()
        plt.show()
        
        fig = plt.figure()

        plt.semilogx(SNRs, errs_EM_mean, '.-b', label=r'EM')

        plt.semilogx(SNRs, errs_ac_mean, '.--r', label='Autocorrelation analysis')
        
        plt.legend(loc=(0.5, 0.62))#, fontsize=6)
        
        plt.xlabel('SNR')
        
        plt.ylabel('Mean computation time [CPU sec]')
        fig.tight_layout()
        plt.show()
