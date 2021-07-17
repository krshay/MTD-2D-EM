# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 18:21:00 2021

@author: Shay Kreymer
"""

import numpy as np

def signal_prior(kvals):
    # corresponding_kvals = calc_corresponding_kvals(kvals)
    # Cr = np.diag((4 * np.exp(- np.abs(kvals) / 8)) ** 2)
    # Ci = np.diag((4 * np.exp(- np.abs(kvals) / 8)) ** 2)
    # Cr[corresponding_kvals, np.arange(len(kvals))] = np.diag(Cr)
    # Ci[corresponding_kvals, np.arange(len(kvals))] = -np.diag(Ci)
    # Ci[np.arange(len(kvals)), np.arange(len(kvals))] = np.diag(Cr)
    # Ci[kvals == 0, kvals == 0] = 0
    # return np.random.multivariate_normal(np.zeros((len(kvals), )), Cr) + 1j * np.random.multivariate_normal(np.zeros((len(kvals), )), Ci)
    Gamma = np.diag((4 * np.exp(- np.abs(kvals) / 8)) ** 2)
    # Gamma[corresponding_kvals, np.arange(len(kvals))] = np.diag(Gamma)
    return np.random.multivariate_normal(np.zeros((len(kvals), )), Gamma), Gamma

def calc_corresponding_kvals(kvals):
    corresponding_kvals = np.zeros_like(kvals)
    for ii in range(len(kvals)):
        if kvals[ii] > 0:
            corresponding_kvals[ii] = ii + 1
        elif kvals[ii] < 0:
            corresponding_kvals[ii] = ii - 1
        else:
            corresponding_kvals[ii] = ii
    return corresponding_kvals

# def calcR(C, Gamma):
#     return np.conj(C).T @ np.linalg.inv(Gamma)

# def calcP(C, Gamma, R=None):
#     if R == None:
#         R = calcR(C, Gamma)
#     return np.conj(Gamma) - R @ C
