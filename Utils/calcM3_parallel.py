# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 20:15:39 2021

@author: Shay Kreymer
"""

import numpy as np
import multiprocessing as mp
from Utils.calc_3rdorder_ac import calc_M3_for_micrograph

def calcM3_parallel_micrographs(L, sigma2, gamma, c, kvals, Bk, W, T, N, NumMicrographs):
    """ Create multiple micrographs and calculate the first-, second- 
    and third-order autocorrelations over all micrographs.
    Computation is done in parallel.

    Args:
        L: diameter of the target image
        sigma2: the variance of the noise
        gamma: the density
        c: real representation of the expansion coefficients of the target image
        kvals: vector of frequencies
        Bk: matrix that maps from the expansion coefficients to the approximated image, in the freuency domain
        W: separation between images; L for arbitrary spacing distribution, 2*L-1 for the well-separated case
        T: matrix that maps from the real representation to the complex representation of the expansion coefficients
        N: the width and height of each micrograph
        NumMicrographs: number of micrographs of size N * N with density gamma and image expanded by c

    Returns:
        M1_ys: list of first-order autocorrelations
        M2_ys: list of second-order autocorrelations
        M3_ys: list of third-order autocorrelations
    """
    print('Started calculations')
    M1_ys = np.zeros((NumMicrographs, ))
    M2_ys = np.zeros((NumMicrographs, L, L))
    M3_ys = np.zeros((NumMicrographs, L, L, L, L))

    num_cpus = mp.cpu_count()
    pool = mp.Pool(num_cpus)
    Ms_parallel = pool.starmap(calc_M3_for_micrograph, [[L, c, kvals, Bk, W, N, gamma, T, sigma2, ii] for ii in range(NumMicrographs)])
    pool.close()
    pool.join()
    
    for ii in range(NumMicrographs):
        M1_ys[ii] = Ms_parallel[ii][0]
        M2_ys[ii, :, :] = Ms_parallel[ii][1]
        M3_ys[ii, :, :, :, :] = Ms_parallel[ii][2]
    
    return M1_ys, M2_ys, M3_ys
