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
    Nsizes = 5
    sizes = np.logspace(np.log10(500), np.log10(3000), Nsizes).astype(int)
    
    SNR = 1
    gamma = 0.04
    K = 16
    num_cpus = mp.cpu_count()
    num_cpus = 5
    # %% EM and Autocorrelation Analysis
    # pool = mp.Pool(num_cpus)
    # S = pool.starmap(calc_err_size_both, [[L, ne, sizes, SNR, gamma, K, i] for i in range(Niters)])
    # pool.close()
    # pool.join() 
    
    # errs_EM = np.zeros((Niters, Nsizes))
    # times_EM = np.zeros((Niters, Nsizes))

    # for j in range(Niters):
    #     errs_EM[j, :] = S[j][0][np.arange(Nsizes), np.argmin(S[j][1], axis=1)]
    #     times_EM[j, :] = S[j][2][np.arange(Nsizes), np.argmin(S[j][1], axis=1)]
    # errs_EM_mean = np.mean(errs_EM, 0)
    # times_EM_mean = np.mean(times_EM, 0)
    
    # errs_ac = np.zeros((Niters, Nsizes))
    # times_ac = np.zeros((Niters, Nsizes))

    # for j in range(Niters):
    #     errs_ac[j, :] = S[j][3][np.arange(Nsizes), np.argmin(S[j][4], axis=1)]
    #     times_ac[j, :] = S[j][5][np.arange(Nsizes), np.argmin(S[j][4], axis=1)]
    # errs_ac_mean = np.mean(errs_ac, 0)
    # times_ac_mean = np.mean(times_ac, 0)
    errs_EM = np.zeros((Niters, Nsizes, 1))
    costs_EM = np.zeros((Niters, Nsizes, 1))
    times_EM = np.zeros((Niters, Nsizes, 1))
    errs_ac = np.zeros((Niters, Nsizes, 1))
    costs_ac = np.zeros((Niters, Nsizes, 1))
    times_ac = np.zeros((Niters, Nsizes, 1))
    
    for i in range((Niters)):
        errs_EM[i, :, :], costs_EM[i, :, :], times_EM[i, :, :], errs_ac[i, :, :], costs_ac[i, :, :], times_ac[i, :, :] = calc_err_size_both(L, ne, sizes, SNR, gamma, K, i)
    
    # # %% plots
    # plt.close("all")
    # with plt.style.context('ieee'):
    #     fig = plt.figure()
        
    #     plt.loglog(sizes**2, errs_EM_mean[3]*(sizes**2/sizes[3]**2)**(-1/2), 'k--', label='_nolegend_', lw=0.5)
    #     plt.loglog(sizes**2, errs_EM_mean, '.-b', label=r'EM')
    
    #     plt.loglog(sizes**2, errs_ac_mean[3]*(sizes**2/sizes[3]**2)**(-1/2), 'k--', label='_nolegend_', lw=0.5)
    #     plt.loglog(sizes**2, errs_ac_mean, '.--r', label='Autocorrelation analysis')
        
    #     plt.legend(loc=(0.5, 0.62))#, fontsize=6)
        
    #     plt.xlabel('Measurement size [pixels]')
        
    #     plt.ylabel('Mean estimation error')
    #     fig.tight_layout()
    #     plt.show()
        
    #     fig = plt.figure()

    #     plt.semilogx(sizes**2, errs_EM_mean, '.-b', label=r'EM')

    #     plt.semilogx(sizes**2, errs_ac_mean, '.--r', label='Autocorrelation analysis')
        
    #     plt.legend(loc=(0.5, 0.62))#, fontsize=6)
        
    #     plt.xlabel('Measurement size [pixels]')
        
    #     plt.ylabel('Mean computation time [CPU sec]')
    #     fig.tight_layout()
    #     plt.show()
