# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 21:34:19 2021

@author: Shay Kreymer
"""


import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from Utils.calc_err_SNR import calc_err_SNR_comparison

plt.close("all")

if __name__ == '__main__':
    # %% Preliminary definitions
    Niters = 40
    L = 5
    ne = 10
    NSNRs = 10
    SNRs = np.logspace(0, 2, NSNRs)
    
    N = 70
    gamma = 0.04
    K = 16
    num_cpus = mp.cpu_count()
    # %% EM and Autocorrelation Analysis
    pool = mp.Pool(num_cpus)
    S = pool.starmap(calc_err_SNR_comparison, [[L, ne, SNRs, N, gamma, K, i] for i in range(Niters)])
    pool.close()
    pool.join() 
    
    errs_EM = np.zeros((Niters, NSNRs))
    likelihood_EM = np.zeros((Niters, NSNRs))
    times_EM = np.zeros((Niters, NSNRs))

    for j in range(Niters):
        errs_EM[j, :] = S[j][0][np.arange(NSNRs), np.argmin(S[j][1], axis=1)]
        likelihood_EM[j, :] = S[j][1][np.arange(NSNRs), np.argmin(S[j][1], axis=1)]
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
    
    errs_conv = np.zeros((Niters, NSNRs))
    for j in range(Niters):
        errs_conv[j, :] = S[j][6][np.arange(NSNRs), 0]
    errs_conv_mean = np.mean(errs_conv, 0)
    # %% plots
    plt.close("all")
    with plt.style.context('ieee'):
        fig = plt.figure()
        # plt.loglog(SNRs[9:15], errs_EM_mean[9]*(SNRs[9:15]/SNRs[9])**(-3/2), 'k--', label='_nolegend_', lw=0.5)
        plt.loglog(SNRs, errs_EM_mean, '.-b', label=r'Approximate EM')
        # plt.loglog(SNRs, errs_ac_mean[4]*(SNRs/SNRs[4])**(-3/2), 'k--', label='_nolegend_', lw=0.5)

        plt.loglog(SNRs, errs_ac_mean, '.--r', label='Autocorrelation analysis')
        plt.loglog(SNRs, errs_conv_mean, label='oracle-deconvolution')
        plt.legend(loc=1)#, fontsize=6)
        
        plt.xlabel('SNR')
        
        plt.ylabel('Mean estimation error')
        fig.tight_layout()
        plt.show()
        # plt.savefig(r'C:\Users\kreym\Google Drive\PhD\Documents\MTD-2D-EM-ICASSP\figures/experiment_SNR_err.pdf')

        # fig = plt.figure()

        # plt.loglog(SNRs, times_EM_mean, '.-b', label=r'Approximate EM')

        # plt.loglog(SNRs, times_ac_mean, '.--r', label='Autocorrelation analysis')
        
        # plt.legend(loc=(0.3, 0.3))#, fontsize=6)
        
        # plt.xlabel('SNR')
        
        # plt.ylabel('Mean computation time [CPU sec]')
        # fig.tight_layout()
        # plt.show()
        # plt.savefig(r'C:\Users\kreym\Google Drive\PhD\Documents\MTD-2D-EM-ICASSP\figures/experiment_SNR_time.pdf')
