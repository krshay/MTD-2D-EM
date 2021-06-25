# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 16:31:53 2021

@author: Shay Kreymer
"""

import numpy as np
import scipy.spatial as spatial
import scipy
import itertools
import multiprocessing as mp

def full_psf_2d(locations, L):
    """ Calculate the pair separation function from the locations of the target images in the measurement.

    Args:
        locations: array of 2-D locations of the target images in a measurement
        L: diameter of the target image

    Returns:
        the pair separation function as an array of size 4L-3 * 4L-3
    """
    M = len(locations)
    r_max = np.sqrt(2)*(2*L-2)
    psf = np.zeros((4*L-3, 4*L-3))
    locations_tree = spatial.cKDTree(locations)
    
    for loc in locations:
        close_locs = [locations[j] for j in locations_tree.query_ball_point(loc, r_max)]
        close_locs = [close_loc for close_loc in close_locs if not ((np.abs(loc[0] - close_loc[0]) >= 2*L-1)
                                                                    or (np.abs(loc[1] - close_loc[1]) >= 2*L-1))]
        for close_loc in close_locs:
            dif = np.array(loc) - np.array(close_loc)
            psf[dif[0]+2*L-2, dif[1]+2*L-2] += 1/(M)
    psf[2*L-2, 2*L-2] = 0

    return psf

def full_tsf_2d(locations, L):
    """ Calculate the triplet separation function from the locations of the target images in the measurement.

    Args:
        locations: array of 2-D locations of the target images in a measurement
        L: diameter of the target image

    Returns:
        the triplet separation function as an array of size 4L-3 * 4L-3 * 4L-3 * 4L-3
    """
    M = len(locations)
    r_max = np.sqrt(2)*(2*L-2)
    tsf = np.zeros((4*L-3, 4*L-3, 4*L-3, 4*L-3))
    locations_tree = spatial.cKDTree(locations)
    
    for loc in locations:
        close_locs = [locations[j] for j in locations_tree.query_ball_point(loc, r_max)]
        close_locs = [close_loc for close_loc in close_locs if not ((np.abs(loc[0] - close_loc[0]) >= 2*L-1)
                                                                    or (np.abs(loc[1] - close_loc[1]) >= 2*L-1)
                                                                    or (loc[0] - close_loc[0] == 0 and loc[1] - close_loc[1] == 0))]
        
        for close_loc1 in close_locs:
            close_locs_reduced = [close_loc for close_loc in close_locs if close_loc[0] != close_loc1[0] and close_loc[1] != close_loc1[1]]
            for close_loc2 in close_locs_reduced:
                dif1 = np.array(loc) - np.array(close_loc1)
                dif2 = np.array(loc) - np.array(close_loc2)
                tsf[dif1[0]+2*L-2, dif1[1]+2*L-2, dif2[0]+2*L-2, dif2[1]+2*L-2] += 1/(M**2)
    tsf[2*L-2, 2*L-2, :, :] = 0
    
    return tsf

def makeExtraMat(L, psf):
    """ Rearranging the pair separation function to a matrix-form, to ease calculations.

    Args:
        L: diameter of the target image
        psf: the pair separation function
        
    Returns:
        ExtraMat2: matrix containing all PSF values; used to ease computations of the second-order autocorrelations
        ExtraMat3: matrix containing all PSF values; used to ease computations of the third-order autocorrelations
    """
    Mat3 = scipy.sparse.lil_matrix((L**4, (2*L-1)**4))
    for i1 in range(L):
        for j1 in range(L):
            for i2 in range(L):
                for j2 in range(L):
                    shift1y = j1
                    shift1x = i1
                    shift2y = j2
                    shift2x = i2
                            
                    row = np.ravel_multi_index([i1, j1, i2, j2], (L, L, L, L))
                    
                    for j in range(max(shift1y, shift2y) - (L-1), L + min(shift1y, shift2y)):
                        for i in range(max(shift1x, shift2x) - (L-1), L + min(shift1x, shift2x)):
                            if not (np.abs(i) < L and np.abs(j) < L):
                                Mat3[row, np.ravel_multi_index([(shift2x-shift1x)%(2*L-1), (shift2y-shift1y)%(2*L-1), (i-shift1x)%(2*L-1), (j-shift1y)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1))] += psf[i + 2*L-2, j + 2*L-2]
                    
                    for j in range(shift1y - (L-1), L + shift1y - shift2y):
                        for i in range(shift1x - (L-1), L + shift1x - shift2x):
                            if not (np.abs(i) < L and np.abs(j) < L):
                                Mat3[row, np.ravel_multi_index([shift2x%(2*L-1), shift2y%(2*L-1), (shift1x-i)%(2*L-1), (shift1y-j)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1))] += psf[i + 2*L-2, j + 2*L-2]
                    
                    for j in range(shift2y - (L-1), L + shift2y - shift1y):
                        for i in range(shift2x - (L-1), L + shift2x - shift1x):
                            if not (np.abs(i) < L and np.abs(j) < L):
                                Mat3[row, np.ravel_multi_index([shift1x%(2*L-1), shift1y%(2*L-1), (shift2x-i)%(2*L-1), (shift2y-j)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1))] += psf[i + 2*L-2, j + 2*L-2]
    
    Mat2 = scipy.sparse.lil_matrix((L**2, (2*L-1)**2))
    for i1 in range(L):
        for j1 in range(L):
            shift1y = j1
            shift1x = i1
            
            row = np.ravel_multi_index([i1, j1], (L, L))
            
            for j in range(shift1y-(L-1), L + shift1y):
                for i in range(shift1x-(L-1), L + shift1x):
                    if not (np.abs(i) < L and np.abs(j) < L):
                        Mat2[row, np.ravel_multi_index([(i-shift1x)%(2*L-1), (j-shift1y)%(2*L-1)], (2*L-1, 2*L-1))] += psf[i + 2*L-2, j + 2*L-2]
                        
    return scipy.sparse.csr_matrix(Mat2), scipy.sparse.csr_matrix(Mat3)

def maketsfMat(L, tsf):
    """ Rearranging the triplet separation function to a matrix-form, to ease calculations.

    Args:
        L: diameter of the target image
        tsf: the triplet separation function
        
    Returns:
        a sparse matrix that is utilized to efficiently calculate the contribution of the pairs of neighbors to the autocorrelations
    """    
    Mat3 = scipy.sparse.lil_matrix((L**4, (2*L-1)**4))
    for ii1 in range(L):
        for jj1 in range(L):
            for ii2 in range(L):
                for jj2 in range(L):
                    shift1y = jj1
                    shift1x = ii1
                    shift2y = jj2
                    shift2x = ii2
                            
                    row = np.ravel_multi_index([ii1, jj1, ii2, jj2], (L, L, L, L))
                    
                    for j1 in range(shift1y - (L-1), L + shift1y):
                        for i1 in range(shift1x - (L-1), L + shift1x):
                            if not (np.abs(i1) < L and np.abs(j1) < L):
                                for j2 in range(max(j1, shift1y) - (L-1) - shift2y, L + min(j1, shift1y) - shift2y):
                                    for i2 in range(max(i1, shift1x) - (L-1) - shift2x, L + min(i1, shift1x) - shift2x):
                                        if not (np.abs(i2) < L and np.abs(j2) < L):
                                            if not (np.abs(i2 - i1) < L and np.abs(j2 - j1) < L):
                                                Mat3[row, np.ravel_multi_index([(i1-shift1x)%(2*L-1), (j1-shift1y)%(2*L-1), (i2+shift2x-shift1x)%(2*L-1), (j2+shift2y-shift1y)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1))] += tsf[i1 + 2*L-2, j1 + 2*L-2, i2 + 2*L-2, j2 + 2*L-2]
                    
                    for j1 in range(shift2y - (L-1), L + shift2y):
                        for i1 in range(shift2x - (L-1), L + shift2x):
                            if not (np.abs(i1) < L and np.abs(j1) < L):
                                for j2 in range(max(j1, shift2y) - (L-1) - shift1y, L + min(j1, shift2y) - shift1y):
                                    for i2 in range(max(i1, shift2x) - (L-1) - shift1x, L + min(i1, shift2x) - shift1x):
                                        if not (np.abs(i2) < L and np.abs(j2) < L):
                                            if not (np.abs(i2 - i1) < L and np.abs(j2 - j1) < L):
                                                Mat3[row, np.ravel_multi_index([(i1-shift2x)%(2*L-1), (j1-shift2y)%(2*L-1), (i2+shift1x-shift2x)%(2*L-1), (j2+shift1y-shift2y)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1))] += tsf[i1 + 2*L-2, j1 + 2*L-2, i2 + 2*L-2, j2 + 2*L-2]
                    
    return scipy.sparse.csr_matrix(Mat3)

def maketsfMat_parallel(L, tsf):
    """ Rearranging the triplet separation function to a matrix-form, to ease calculations. Parallel computing.

    Args:
        L: diameter of the target image
        tsf: the triplet separation function
        
    Returns:
        a sparse matrix that is utilized to efficiently calculate the contribution of the pairs of neighbors to the autocorrelations
    """  
    # Rearranging the triplet separation function to a matrix-form, to ease calculations. Parallel processing.
    shifts = list(itertools.product(np.arange(L), np.arange(L), np.arange(L), np.arange(L)))
    num_cpus = mp.cpu_count()
    divided_shifts = np.array_split(shifts, num_cpus)
    pool = mp.Pool(num_cpus)
    tsfMats = pool.starmap(maketsfMat_partial, [[L, tsf, shift_divided] for shift_divided in divided_shifts])
    pool.close()
    pool.join()
    
    return np.sum(tsfMats)

def maketsfMat_partial(L, tsf, shifts):
    """ Rearranging the triplet separation function to a matrix-form, to ease calculations, for specific shifts.

    Args:
        L: diameter of the target image
        tsf: the triplet separation function
        
    Returns:
        a sparse matrix that is utilized to efficiently calculate the contribution of the pairs of neighbors to the autocorrelations. For specific shifts, merged in maketsfMat_parallel.
    """  
    Mat3 = scipy.sparse.lil_matrix((L**4, (2*L-1)**4))
    for shift in shifts:
        shift1y, shift1x, shift2y, shift2x = shift
                
        row = np.ravel_multi_index([shift1y, shift1x, shift2y, shift2x], (L, L, L, L))
        
        for j1 in range(shift1y - (L-1), L + shift1y):
            for i1 in range(shift1x - (L-1), L + shift1x):
                if not (np.abs(i1) < L and np.abs(j1) < L):
                    for j2 in range(max(j1, shift1y) - (L-1) - shift2y, L + min(j1, shift1y) - shift2y):
                        for i2 in range(max(i1, shift1x) - (L-1) - shift2x, L + min(i1, shift1x) - shift2x):
                            if not (np.abs(i2) < L and np.abs(j2) < L):
                                if not (np.abs(i2 - i1) < L and np.abs(j2 - j1) < L):
                                    Mat3[row, np.ravel_multi_index([(i1-shift1x)%(2*L-1), (j1-shift1y)%(2*L-1), (i2+shift2x-shift1x)%(2*L-1), (j2+shift2y-shift1y)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1))] += tsf[i1 + 2*L-2, j1 + 2*L-2, i2 + 2*L-2, j2 + 2*L-2]
        
        for j1 in range(shift2y - (L-1), L + shift2y):
            for i1 in range(shift2x - (L-1), L + shift2x):
                if not (np.abs(i1) < L and np.abs(j1) < L):
                    for j2 in range(max(j1, shift2y) - (L-1) - shift1y, L + min(j1, shift2y) - shift1y):
                        for i2 in range(max(i1, shift2x) - (L-1) - shift1x, L + min(i1, shift2x) - shift1x):
                            if not (np.abs(i2) < L and np.abs(j2) < L):
                                if not (np.abs(i2 - i1) < L and np.abs(j2 - j1) < L):
                                    Mat3[row, np.ravel_multi_index([(i1-shift2x)%(2*L-1), (j1-shift2y)%(2*L-1), (i2+shift1x-shift2x)%(2*L-1), (j2+shift1y-shift2y)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1))] += tsf[i1 + 2*L-2, j1 + 2*L-2, i2 + 2*L-2, j2 + 2*L-2]
        
    return scipy.sparse.csr_matrix(Mat3)
