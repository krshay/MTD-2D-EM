#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 13:41:00 2021

@author: shaykreymer
"""

import numpy as np
import matplotlib.pyplot as plt
from Utils.calc_err_SNR import calc_err_SNR_initialize_from_aca

plt.close("all")

if __name__ == '__main__':
    Niters = 40
    L = 5
    ne = 10
    NSNRs = 7
    SNRs = np.logspace(0, 1.05, NSNRs)
    
    N = 2500
    gamma = 0.04
    K = 8
    errs_EM = np.zeros((Niters, NSNRs))
    errs_ac = np.zeros((Niters, NSNRs))
    for i in range(Niters):
        errors_EM, errors_ac = calc_err_SNR_initialize_from_aca(L, ne, SNRs, N, gamma, K, i)
        errs_EM[i, :] = errors_EM
        errs_ac[i, :] = errors_ac
   
    errs_EM_mean = np.mean(errs_EM, 0)

    errs_ac_mean = np.mean(errs_ac, 0)
