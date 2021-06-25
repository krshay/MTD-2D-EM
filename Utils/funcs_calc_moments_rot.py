# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 12:23:14 2020

@author: Shay Kreymer
"""

import numpy as np
import itertools
import scipy.sparse as sp
import multiprocessing as mp

def calck1k2k3(L):
    """ Calculate triplets of freuencies k1, k2 and -k1-k2
    Args:
        L: diameter of the target image

    Returns:
        list of triplets of frequencies
    """
    kmap = list(itertools.product(np.arange(2*L-1), np.arange(2*L-1)))
    k2map = list(itertools.product(kmap, kmap))
    k1k2k3_map = []
    for k2 in k2map:
        shift1 = k2[0]
        shift2 = k2[1]
        idx3x = (-shift1[0] - shift2[0]) % (2*L - 1)
        idx3y = (-shift1[1] - shift2[1]) % (2*L - 1)
        shift3 = (idx3x, idx3y)
        k1k2k3_map.append((shift1, shift2, shift3))
    
    return k1k2k3_map

def calcmap3(L):
    """ Calculate triplets of freuencies k1, k2 and -k1-k2, in an array form
    Args:
        L: diameter of the target image

    Returns:
        array of triplets of frequencies
    """
    k1k2k3_map = calck1k2k3(L)
    map3 = np.zeros((np.shape(k1k2k3_map)[0], 6))

    for i in range(np.shape(map3)[0]):
        map3[i, :] = [k1k2k3_map[i][0][0], k1k2k3_map[i][0][1], k1k2k3_map[i][1][0], k1k2k3_map[i][1][1], k1k2k3_map[i][2][0], k1k2k3_map[i][2][1]]
        
    return map3.astype(int)

def calck1(L):
    """ Calculate a list of frequencies k
    Args:
        L: diameter of the target image

    Returns:
        list of frequencies
    """
    k1_map = list(itertools.product(np.arange(2*L-1), np.arange(2*L-1)))
    
    return k1_map

def calcN_mat(L):
    """ Calculate matrix of the influence of the noise (1s and 0s according to the deltas in the proposition in the paper)
    Args:
        L: diameter of the target image

    Returns:
        sparse matrix of the influence of the noise
    """
    N_mat = np.zeros((L, L, L, L))
    for ii in range(L):
        for jj in range(L):
            N_mat[ii, jj, ii, jj] += 1
    N_mat[ 0, 0, :, :] += 1
    N_mat[ :, :, 0, 0] += 1
    
    return sp.csr_matrix(np.reshape(N_mat, (L**4, 1)))

def calcS3_x_grad(L, Nmax, Bk, z, kvals, map3):
    """ Calculate the third-order rotationally-averaged autocorrelations of an image expanded using z, and the gradients w.r.t. z

    Args:
        L: diameter of the target image
        Nmax: maximum number of angles for calculation (6 \nu_max in the paper)
        Bk: matrix that maps from the expansion coefficients to the approximated image, in the freuency domain
        z: Fourier-Bessel expansion coefficients in complex format
        kvals: vector of frequencies
        map3: array of triplets of frequencies

    Returns:
        S3_x: the third-order rotationally-averaged autocorrelations
        gS3_x: the corresponding gradients
    """
    kvals = np.atleast_2d(kvals)

    k = np.atleast_2d(2*np.pi * np.arange(Nmax) / Nmax)
    omega_phi_nu = np.moveaxis(np.repeat(Bk[ :, :, :, np.newaxis], Nmax, axis=3)*np.exp(1j*kvals.T*k), [2, 3], [3, 2])
    Fk = np.dot(omega_phi_nu, z)
    Fk0 = Fk[map3[ :, 0], map3[ :, 1], :]
    Fk1 = Fk[map3[ :, 2], map3[ :, 3], :]
    Fk2 = Fk[map3[ :, 4], map3[ :, 5], :]
    
    S3_k = np.reshape(np.sum(Fk0 * Fk1 * Fk2, axis=1), (2*L-1, 2*L-1, 2*L-1, 2*L-1))
    
    omega_phi_nu0 = omega_phi_nu[map3[ :, 0], map3[ :, 1], :, :]
    omega_phi_nu1 = omega_phi_nu[map3[ :, 2], map3[ :, 3], :, :]
    omega_phi_nu2 = omega_phi_nu[map3[ :, 4], map3[ :, 5], :, :]
    
    Fk0 = np.repeat(Fk0[ :, :, np.newaxis], len(z), axis=2)
    Fk1 = np.repeat(Fk1[ :, :, np.newaxis], len(z), axis=2)
    Fk2 = np.repeat(Fk2[ :, :, np.newaxis], len(z), axis=2)
    
    gS3_k = np.reshape(np.sum(Fk0 * Fk1 * omega_phi_nu2 + Fk0 * Fk2 * omega_phi_nu1 + Fk1 * Fk2 * omega_phi_nu0, axis=1), (2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)))

    S3_x = np.fft.ifftn(S3_k) / (Nmax * L**2)
    gS3_x = np.fft.ifftn(gS3_k, axes=(0,1,2,3)) / (Nmax * L**2)
    
    return S3_x, gS3_x

def calcS3_x_neigh_grad(L, Nmax, Bk, z, kvals, map3):
    """ Calculate the third-order rotationally-averaged autocorrelations with a neighbor of an image expanded using z, and the gradients w.r.t. z

    Args:
        L: diameter of the target image
        Nmax: maximum number of angles for calculation (6 \nu_max in the paper)
        Bk: matrix that maps from the expansion coefficients to the approximated image, in the freuency domain
        z: Fourier-Bessel expansion coefficients in complex format
        kvals: vector of frequencies
        map3: array of triplets of frequencies

    Returns:
        S3_x_neigh: the third-order rotationally-averaged autocorrelations with a neighbor
        gS3_x_neigh: the corresponding gradients
    """
    kvals = np.atleast_2d(kvals)
    
    Nmax0 = int((2/3)*Nmax)
    omega0 = Bk * (kvals == 0)
    omega0_expanded = np.moveaxis(np.repeat(omega0[:, :, :, np.newaxis], Nmax0, axis=3), [2, 3], [3, 2])
    F0 = np.dot(omega0, z)
    F0_expanded = np.repeat(F0[:, :, np.newaxis], Nmax0, axis=2)
    k = np.atleast_2d(2*np.pi * np.arange(Nmax0) / Nmax0)
    omega_phi_nu = np.moveaxis(np.repeat(Bk[ :, :, :, np.newaxis], Nmax0, axis=3)*np.exp(1j*kvals.T*k), [2, 3], [3, 2])
    Fk = np.dot(omega_phi_nu, z)
    Fk0 = Fk[map3[ :, 0], map3[ :, 1], :]
    Fk1 = F0_expanded[map3[ :, 2], map3[ :, 3], :]
    Fk2 = Fk[map3[ :, 4], map3[ :, 5], :]
    
    S3_k_neigh = np.reshape(np.sum(Fk0 * Fk1 * Fk2, axis=1), (2*L-1, 2*L-1, 2*L-1, 2*L-1))
    
    omega_phi_nu0 = omega_phi_nu[map3[ :, 0], map3[ :, 1], :, :]
    omega_phi_nu1 = omega0_expanded[map3[ :, 2], map3[ :, 3], :, :]
    omega_phi_nu2 = omega_phi_nu[map3[ :, 4], map3[ :, 5], :, :]
    
    Fk0 = np.repeat(Fk0[ :, :, np.newaxis], len(z), axis=2)
    Fk1 = np.repeat(Fk1[ :, :, np.newaxis], len(z), axis=2)
    Fk2 = np.repeat(Fk2[ :, :, np.newaxis], len(z), axis=2)
    
    gS3_k_neigh = np.reshape(np.sum(Fk0 * Fk1 * omega_phi_nu2 + Fk0 * Fk2 * omega_phi_nu1 + Fk1 * Fk2 * omega_phi_nu0, axis=1), (2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)))
    
    S3_x_neigh = np.fft.ifftn(S3_k_neigh) / (Nmax0 * L**2)
    gS3_x_neigh = np.fft.ifftn(gS3_k_neigh, axes=(0,1,2,3)) / (Nmax0 * L**2)
    
    return S3_x_neigh, gS3_x_neigh

def calcS3_x_triplets_grad(L, Nmax, Bk, z, kvals, map3):
    """ Calculate the third-order rotationally-averaged autocorrelations with a pair of neighbors of an image expanded using z, and the gradients w.r.t. z

    Args:
        L: diameter of the target image
        Nmax: maximum number of angles for calculation (6 \nu_max in the paper)
        Bk: matrix that maps from the expansion coefficients to the approximated image, in the freuency domain
        z: Fourier-Bessel expansion coefficients in complex format
        kvals: vector of frequencies
        map3: array of triplets of frequencies

    Returns:
        S3_x_triplets: the third-order rotationally-averaged autocorrelations of a triplet
        gS3_x_triplets: the corresponding gradients
    """
    kvals = np.atleast_2d(kvals)
    
    omega0 = Bk * (kvals == 0)
    F0 = np.dot(omega0, z)
    # %% Rotationally-averaged third-order autocorrelation with two neighbors
    Fk0 = np.atleast_2d(F0[map3[ :, 0], map3[ :, 1]])
    Fk1 = np.atleast_2d(F0[map3[ :, 2], map3[ :, 3]])
    Fk2 = np.atleast_2d(F0[map3[ :, 4], map3[ :, 5]])
    
    S3_k_triplets = np.reshape(np.squeeze(Fk0 * Fk1 * Fk2), (2*L-1, 2*L-1, 2*L-1, 2*L-1))
    
    omega_phi_nu0 = omega0[map3[ :, 0], map3[ :, 1], :]
    omega_phi_nu1 = omega0[map3[ :, 2], map3[ :, 3], :]
    omega_phi_nu2 = omega0[map3[ :, 4], map3[ :, 5], :]
    
    Fk0 = np.repeat(Fk0[ :, :, np.newaxis], len(z), axis=2)
    Fk1 = np.repeat(Fk1[ :, :, np.newaxis], len(z), axis=2)
    Fk2 = np.repeat(Fk2[ :, :, np.newaxis], len(z), axis=2)
    
    gS3_k_triplets = np.reshape(np.squeeze(Fk0 * Fk1 * omega_phi_nu2 + Fk0 * Fk2 * omega_phi_nu1 + Fk1 * Fk2 * omega_phi_nu0), (2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)))
    
    S3_x_triplets = np.fft.ifftn(S3_k_triplets) / (L**2)
    gS3_x_triplets = np.fft.ifftn(gS3_k_triplets, axes=(0,1,2,3)) / (L**2)
    
    return S3_x_triplets, gS3_x_triplets

def calcS2_x_grad_notparallel(L, Nmax, Bk, z, kvals, k1_map):
    """ Calculate the second-order rotationally-averaged autocorrelations of an image expanded using z, and the gradients w.r.t. z

    Args:
        L: diameter of the target image
        Nmax: maximum number of angles for calculation (6 \nu_max in the paper)
        Bk: matrix that maps from the expansion coefficients to the approximated image, in the freuency domain
        z: Fourier-Bessel expansion coefficients in complex format
        kvals: vector of frequencies
        k1_map: list of frequencies

    Returns:
        S2_x: the second-order rotationally-averaged autocorrelations
        gS2_x: the corresponding gradients
    """
    S2_k = np.zeros((2*L-1, 2*L-1), dtype=np.complex_)
    gS2_k = np.zeros((2*L-1, 2*L-1, len(z)), dtype=np.complex_)
    
    for k in range(Nmax):
        phi_nu = 2*np.pi * k / Nmax
        omega_phi_nu = Bk*np.exp(1j*kvals*phi_nu)
        Fk = omega_phi_nu @ z
        for k1 in k1_map:
            S2_k[k1[0], k1[1]] += np.abs(Fk[k1[0], k1[1]])**2
            gS2_k[k1[0], k1[1], :] += 2 * Fk[k1[0], k1[1]] * omega_phi_nu[(-k1[0]) % (2*L-1), (-k1[1]) % (2*L-1), :]          
    
    S2_x = np.fft.ifftn(S2_k) / (Nmax * L**2)
    gS2_x = np.fft.ifftn(gS2_k, axes=(0,1)) / (Nmax * L**2)
    
    return S2_x, gS2_x

def calcS2_x_neigh_grad_notparallel(L, Bk, z, kvals, k1_map):
    """ Calculate the second-order rotationally-averaged autocorrelations with a neighbor of an image expanded using z, and the gradients w.r.t. z

    Args:
        L: diameter of the target image
        Nmax: maximum number of angles for calculation (6 \nu_max in the paper)
        Bk: matrix that maps from the expansion coefficients to the approximated image, in the freuency domain
        z: Fourier-Bessel expansion coefficients in complex format
        kvals: vector of frequencies
        k1_map: list of frequencies

    Returns:
        S2_x_neigh: the second-order rotationally-averaged autocorrelations with a neighbor
        gS2_x_neigh: the corresponding gradients
    """
    S2_k_neigh = np.zeros((2*L-1, 2*L-1), dtype=np.complex_)
    gS2_k_neigh = np.zeros((2*L-1, 2*L-1, len(z)), dtype=np.complex_)
    
    omega0 = Bk * (kvals == 0)
    F0 = omega0 @ z
    for k1 in k1_map:
        S2_k_neigh[k1[0], k1[1]] += F0[k1[0], k1[1]] * F0[(-k1[0]) % (2*L-1), (-k1[1]) % (2*L-1)]
        gS2_k_neigh[k1[0], k1[1], :] += F0[k1[0], k1[1]] * omega0[(-k1[0]) % (2*L-1), (-k1[1]) % (2*L-1), :] + F0[(-k1[0]) % (2*L-1), (-k1[1]) % (2*L-1)] * omega0[k1[0], k1[1], :]
    
    S2_x_neigh = np.fft.ifftn(S2_k_neigh) / (L**2)
    gS2_x_neigh = np.fft.ifftn(gS2_k_neigh, axes=(0,1)) / (L**2)
    
    return S2_x_neigh, gS2_x_neigh

def calcS3_x_grad_neigh_triplets_parallel(L, Nmax, Bk, z, kvals, map3):
    """ Calculate parallely the third-order rotationally-averaged autocorrelations, 
    the third-order rotationally-averaged autocorrelations with a neighbor, 
    and the third-order rotationally-averaged autocorrelations of a triplet,
    of an image expanded using z, and the gradients w.r.t. z. 
    The computation is done in calcS3_x_grad_neigh_triplets_partial below. 
    Here we divide the computations to calculate efficiently in parallel, 
    and then merge.
    
    Args:
        L: diameter of the target image
        Nmax: maximum number of angles for calculation (6 \nu_max in the paper)
        Bk: matrix that maps from the expansion coefficients to the approximated image, in the freuency domain
        z: Fourier-Bessel expansion coefficients in complex format
        kvals: vector of frequencies
        map3: array of triplets of frequencies

    Returns:
        S3_x: the third-order rotationally-averaged autocorrelations
        gS3_x: the corresponding gradients
        S3_x_neigh: the third-order rotationally-averaged autocorrelations with a neighbor
        gS3_x_neigh: the corresponding gradients
        S3_x_triplets: the third-order rotationally-averaged autocorrelations of a triplet
        gS3_x_triplets: the corresponding gradients
    """
    num_cpus = mp.cpu_count()
    divided_k1k2k3map = np.array_split(map3, num_cpus) # dividing the array of triplets of frequencies for parallel processing
    
    pool = mp.Pool(num_cpus)
    S = pool.starmap(calcS3_x_grad_neigh_triplets_partial, [[L, Nmax, Bk, z, kvals, div_k1k2k3_map] for div_k1k2k3_map in divided_k1k2k3map])
    pool.close()
    pool.join()
    
    S3_k = np.reshape(np.concatenate([S3 for S3, gS3, S3_neigh, gS3_neigh, S3_triplets, gS3_triplets in S]), (2*L-1, 2*L-1, 2*L-1, 2*L-1))
    
    gS3_k = np.reshape(np.concatenate([gS3 for S3, gS3, S3_neigh, gS3_neigh, S3_triplets, gS3_triplets in S]), (2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)))

    S3_k_neigh = np.reshape(np.concatenate([S3_neigh for S3, gS3, S3_neigh, gS3_neigh, S3_triplets, gS3_triplets in S]), (2*L-1, 2*L-1, 2*L-1, 2*L-1))
    
    gS3_k_neigh = np.reshape(np.concatenate([gS3_neigh for S3, gS3, S3_neigh, gS3_neigh, S3_triplets, gS3_triplets in S]), (2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)))
    
    S3_k_triplets = np.reshape(np.concatenate([S3_triplets for S3, gS3, S3_neigh, gS3_neigh, S3_triplets, gS3_triplets in S]), (2*L-1, 2*L-1, 2*L-1, 2*L-1))
    
    gS3_k_triplets = np.reshape(np.concatenate([gS3_triplets for S3, gS3, S3_neigh, gS3_neigh, S3_triplets, gS3_triplets in S]), (2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)))
    
    S3_x = np.fft.ifftn(S3_k) / (Nmax * L**2)
    gS3_x = np.fft.ifftn(gS3_k, axes=(0,1,2,3)) / (Nmax * L**2)
    S3_x_neigh = np.fft.ifftn(S3_k_neigh) / ((2/3)*Nmax * L**2)
    gS3_x_neigh = np.fft.ifftn(gS3_k_neigh, axes=(0,1,2,3)) / ((2/3)*Nmax * L**2)
    S3_x_triplets = np.fft.ifftn(S3_k_triplets) / (L**2)
    gS3_x_triplets = np.fft.ifftn(gS3_k_triplets, axes=(0,1,2,3)) / (L**2)
    
    return S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, S3_x_triplets, gS3_x_triplets

def calcS3_x_grad_neigh_triplets_partial(L, Nmax, Bk, z, kvals, map3):
    """ Calculate parallely the Fourier transorm of the third-order 
    rotationally-averaged autocorrelations, the third-order 
    rotationally-averaged autocorrelations with a neighbor, and the third-order
    rotationally-averaged autocorrelations of a triplet, of an image expanded
    using z, and the gradients w.r.t. z. 
    
    Args:
        L: diameter of the target image
        Nmax: maximum number of angles for calculation (6 \nu_max in the paper)
        Bk: matrix that maps from the expansion coefficients to the approximated image, in the freuency domain
        z: Fourier-Bessel expansion coefficients in complex format
        kvals: vector of frequencies
        map3: array of triplets of frequencies

    Returns:
        S3_k: the Fourier transform of the third-order rotationally-averaged autocorrelations
        gS3_k: the corresponding gradients
        S3_k_neigh: the Fourier transform of the third-order rotationally-averaged autocorrelations with a neighbor
        gS3_k_neigh: the corresponding gradients
        S3_k_triplets: the Fourier transform of the third-order rotationally-averaged autocorrelations of a triplet
        gS3_k_triplets: the corresponding gradients
    """
    kvals = np.atleast_2d(kvals)
    # %% Rotationally-averaged third-order autocorrelation
    k = np.atleast_2d(2*np.pi * np.arange(Nmax) / Nmax)
    omega_phi_nu = np.moveaxis(np.repeat(Bk[ :, :, :, np.newaxis], Nmax, axis=3)*np.exp(1j*kvals.T*k), [2, 3], [3, 2])
    Fk = np.dot(omega_phi_nu, z)
    Fk0 = Fk[map3[ :, 0], map3[ :, 1], :]
    Fk1 = Fk[map3[ :, 2], map3[ :, 3], :]
    Fk2 = Fk[map3[ :, 4], map3[ :, 5], :]
    
    S3_k = np.sum(Fk0 * Fk1 * Fk2, axis=1)
    
    omega_phi_nu0 = omega_phi_nu[map3[ :, 0], map3[ :, 1], :, :]
    omega_phi_nu1 = omega_phi_nu[map3[ :, 2], map3[ :, 3], :, :]
    omega_phi_nu2 = omega_phi_nu[map3[ :, 4], map3[ :, 5], :, :]
    
    Fk0 = np.repeat(Fk0[ :, :, np.newaxis], len(z), axis=2)
    Fk1 = np.repeat(Fk1[ :, :, np.newaxis], len(z), axis=2)
    Fk2 = np.repeat(Fk2[ :, :, np.newaxis], len(z), axis=2)
    
    gS3_k = np.sum(Fk0 * Fk1 * omega_phi_nu2 + Fk0 * Fk2 * omega_phi_nu1 + Fk1 * Fk2 * omega_phi_nu0, axis=1)

    # %% Rotationally-averaged third-order autocorrelation with a neighbor
    Nmax0 = int((2/3)*Nmax)
    omega0 = Bk * (kvals == 0)
    omega0_expanded = np.moveaxis(np.repeat(omega0[:, :, :, np.newaxis], Nmax0, axis=3), [2, 3], [3, 2])
    F0 = np.dot(omega0, z)
    F0_expanded = np.repeat(F0[:, :, np.newaxis], Nmax0, axis=2)
    k = np.atleast_2d(2*np.pi * np.arange(Nmax0) / Nmax0)
    omega_phi_nu = np.moveaxis(np.repeat(Bk[ :, :, :, np.newaxis], Nmax0, axis=3)*np.exp(1j*kvals.T*k), [2, 3], [3, 2])
    Fk = np.dot(omega_phi_nu, z)
    Fk0 = Fk[map3[ :, 0], map3[ :, 1], :]
    Fk1 = F0_expanded[map3[ :, 2], map3[ :, 3], :]
    Fk2 = Fk[map3[ :, 4], map3[ :, 5], :]
    
    S3_k_neigh = np.sum(Fk0 * Fk1 * Fk2, axis=1)
    
    omega_phi_nu0 = omega_phi_nu[map3[ :, 0], map3[ :, 1], :, :]
    omega_phi_nu1 = omega0_expanded[map3[ :, 2], map3[ :, 3], :, :]
    omega_phi_nu2 = omega_phi_nu[map3[ :, 4], map3[ :, 5], :, :]
    
    Fk0 = np.repeat(Fk0[ :, :, np.newaxis], len(z), axis=2)
    Fk1 = np.repeat(Fk1[ :, :, np.newaxis], len(z), axis=2)
    Fk2 = np.repeat(Fk2[ :, :, np.newaxis], len(z), axis=2)
    
    gS3_k_neigh = np.sum(Fk0 * Fk1 * omega_phi_nu2 + Fk0 * Fk2 * omega_phi_nu1 + Fk1 * Fk2 * omega_phi_nu0, axis=1)
    
    # %% Rotationally-averaged third-order autocorrelation with two neighbors
    Fk0 = np.atleast_2d(F0[map3[ :, 0], map3[ :, 1]])
    Fk1 = np.atleast_2d(F0[map3[ :, 2], map3[ :, 3]])
    Fk2 = np.atleast_2d(F0[map3[ :, 4], map3[ :, 5]])
    
    S3_k_triplets = np.squeeze(Fk0 * Fk1 * Fk2)
    
    omega_phi_nu0 = omega0[map3[ :, 0], map3[ :, 1], :]
    omega_phi_nu1 = omega0[map3[ :, 2], map3[ :, 3], :]
    omega_phi_nu2 = omega0[map3[ :, 4], map3[ :, 5], :]
    
    Fk0 = np.repeat(Fk0[ :, :, np.newaxis], len(z), axis=2)
    Fk1 = np.repeat(Fk1[ :, :, np.newaxis], len(z), axis=2)
    Fk2 = np.repeat(Fk2[ :, :, np.newaxis], len(z), axis=2)
    
    gS3_k_triplets = np.squeeze(Fk0 * Fk1 * omega_phi_nu2 + Fk0 * Fk2 * omega_phi_nu1 + Fk1 * Fk2 * omega_phi_nu0)
    
    return S3_k, gS3_k, S3_k_neigh, gS3_k_neigh, S3_k_triplets, gS3_k_triplets
