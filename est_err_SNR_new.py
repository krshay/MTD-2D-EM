#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 13:41:00 2021

@author: shaykreymer
"""


import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from Utils.calc_err_SNR import calc_err_SNR_startac

plt.close("all")

if __name__ == '__main__':
    # %% Preliminary definitions
    Niters = 40
    L = 5
    ne = 10
    NSNRs = 8
    SNRs = np.logspace(0, 1.7, NSNRs)
    
    N = 3700
    gamma = 0.04
    K = 8
    errs_EM = np.zeros((Niters, NSNRs))
    errs_ac = np.zeros((Niters, NSNRs))
    for i in range(Niters):
        errors_EM, errors_ac = calc_err_SNR_startac(L, ne, SNRs, N, gamma, K, i)
        errs_EM[i, :] = errors_EM
        errs_ac[i, :] = errors_ac
        np.save(f'errors_EM_{i}.npy', errs_EM)
        np.save(f'errors_ac_{i}.npy', errs_ac)
    

    errs_EM_mean = np.mean(errs_EM, 0)

    errs_ac_mean = np.mean(errs_ac, 0)
    # %% plots
    plt.close("all")
    with plt.style.context('ieee'):
        fig = plt.figure()
        # plt.loglog(SNRs[9:15], errs_EM_mean[9]*(SNRs[9:15]/SNRs[9])**(-3/2), 'k--', label='_nolegend_', lw=0.5)
        plt.loglog(SNRs, errs_EM_mean, '.-b', label=r'Approximate EM')
        # plt.loglog(SNRs, errs_ac_mean[4]*(SNRs/SNRs[4])**(-3/2), 'k--', label='_nolegend_', lw=0.5)

        plt.loglog(SNRs, errs_ac_mean, '.--r', label='Autocorrelation analysis')
        
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
