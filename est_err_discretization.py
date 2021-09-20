# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 15:02:51 2021

@author: Shay Kreymer
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from Utils.calc_err_K import calc_err_K_both

plt.close("all")

if __name__ == '__main__':
    # %% Preliminary definitions
    Niters = 40
    L = 5
    ne = 10
    
    SNR = 5
    N = 1500
    gamma = 0.04
    Ks = [1, 2, 4, 8, 16, 32]
    NKs = np.shape(Ks)[0]
    num_cpus = mp.cpu_count()
    # %% EM and Autocorrelation Analysis
    pool = mp.Pool(num_cpus)
    S = pool.starmap(calc_err_K_both, [[L, ne, Ks, SNR, N, gamma, i] for i in range(Niters)])
    pool.close()
    pool.join() 
    
    errs_EM = np.zeros((Niters, NKs))
    times_EM = np.zeros((Niters, NKs))

    for j in range(Niters):
        errs_EM[j, :] = S[j][0][np.arange(NKs), np.argmin(S[j][1], axis=1)]
        times_EM[j, :] = S[j][2][np.arange(NKs), np.argmin(S[j][1], axis=1)]
    errs_EM_mean = np.mean(errs_EM, 0)
    times_EM_mean = np.mean(times_EM, 0)
    
    errs_ac = np.zeros((Niters, NKs))
    times_ac = np.zeros((Niters, NKs))

    for j in range(Niters):
        errs_ac[j, :] = S[j][3][np.arange(NKs), np.argmin(S[j][4], axis=1)]
        times_ac[j, :] = S[j][5][np.arange(NKs), np.argmin(S[j][4], axis=1)]
    errs_ac_mean = np.mean(errs_ac, 0)
    times_ac_mean = np.mean(times_ac, 0)

    # %% plots
    plt.close("all")
    with plt.style.context('ieee'):
        fig = plt.figure()
        # plt.loglog(Ks, errs_EM_mean[-1]*(Ks/Ks[-1])**(-1/2), 'k--', label='_nolegend_', lw=0.5)
        plt.semilogy(Ks, errs_EM_mean, '.-b', label=r'Approximate EM')
    
        plt.semilogy(Ks, errs_ac_mean, '.--r', label='Autocorrelation analysis')
        
        plt.legend(loc=1, fontsize=6)
        
        plt.xlabel('K')
        
        plt.ylabel('Mean estimation error')
        plt.xticks(Ks)
        fig.tight_layout()
        plt.show()
        plt.savefig(r'C:\Users\kreym\Google Drive\PhD\Documents\MTD-2D-EM-ICASSP\figures/experiment_K_err.pdf')

        fig = plt.figure()

        plt.plot(Ks, times_EM_mean, '.-b', label=r'Approximate EM')

        plt.plot(Ks, times_ac_mean, '.--r', label='Autocorrelation analysis')
        
        plt.legend(loc=2, fontsize=6)
        
        plt.xlabel('K')
        plt.xticks(Ks)
        
        plt.ylabel('Mean computation time [CPU sec]')
        fig.tight_layout()
        plt.show()
        plt.savefig(r'C:\Users\kreym\Google Drive\PhD\Documents\MTD-2D-EM-ICASSP\figures/experiment_K_time.pdf')
