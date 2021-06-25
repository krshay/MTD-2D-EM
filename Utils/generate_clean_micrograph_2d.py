# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:40:17 2019

@author: Shay Kreymer
"""

import numpy as np
from Utils.fb_funcs import rot_img_freqT
from numpy.random import default_rng

def generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, m, T, p=np.array([1]), seed=None):
    """ Form an N*N matrix containing target images at random locations and rotations.

    Args:
        c: real representation of the expansion coefficients of the target image
        kvals: vector of frequencies
        Bk: matrix that maps from the expansion coefficients to the approximated image, in the freuency domain
        W: separation between images; L for arbitrary spacing distribution, 2*L-1 for the well-separated case
        L: diameter of the target image
        N: the width and height of the required micrograph
        m: wanted number of images to place
        T: matrix that maps from the real representation to the complex representation of the expansion coefficients
        p: choosing probability of images (default is np.array([1]); may be utilized for heterogeneity)
        seed: random seed (default is None)
    
    Returns:
        Y: N*N matrix containing target images at random locations and rotations
        placed_list: number of placed target images
        locations: list of 2-D locations of the target images in Y
    """
    if seed != None:
        np.random.seed(seed)
    m = round(m)
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(m, ))
    mask = np.zeros((N, N))
    # The locations table records the chosen signal locations.
    locations = [[] for i in range(m)]
    # This counter records how many signals we successfully placed.
    placed = 0
    placed_list = [0 for i in range(1)]
    max_trials = 5*m
    Y = np.zeros((N, N))
    for _ in range(max_trials):
        # Pick a candidate location for the upper-left corner of the signal
        candidate = np.random.randint(N-W, size=2)
        # Check if there is enough room, taking the separation rule into
        # account. That is, a square of size WxW with upper-left corner
        # specified by the candidate must be entirely free.
        if not((mask[candidate[0]:candidate[0]+2*W-1, candidate[1]:candidate[1]+2*W-1] == 1).any()):
            # Record the successful candidate
            locations[placed] = candidate
            # Mark the area as reserved
            mask[candidate[0]:candidate[0]+W, candidate[1]:candidate[1]+W] = 1
            index_rand = np.random.choice(1, p=p)
            placed_list[index_rand] = placed_list[index_rand] + 1
            theta = thetas[placed]
            placed = placed + 1
            X_theta = rot_img_freqT(theta, c, kvals, Bk, L, T)
            Y[candidate[0] : candidate[0] + L, candidate[1] : candidate[1] + L] = X_theta
            # Stop if we placed sufficiently many signals successfully.
            if placed >= m:
                break
    locations = locations[0:placed]
    return Y, placed_list, locations

def generate_clean_micrograph_2d_rots_discrete(c, kvals, Bk, W, L, N, m, T, K, seed=None):
    """ Form an N*N matrix containing target images at random locations and rotations.

    Args:
        c: real representation of the expansion coefficients of the target image
        kvals: vector of frequencies
        Bk: matrix that maps from the expansion coefficients to the approximated image, in the freuency domain
        W: separation between images; L for arbitrary spacing distribution, 2*L-1 for the well-separated case
        L: diameter of the target image
        N: the width and height of the required micrograph
        m: wanted number of images to place
        T: matrix that maps from the real representation to the complex representation of the expansion coefficients
    
    Returns:
        Y: N*N matrix containing target images at random locations and rotations
        placed_list: number of placed target images
        locations: list of 2-D locations of the target images in Y
    """
    if seed != None:
        np.random.seed(seed)
    m = round(m)
    possible_thetas = np.linspace(0, 360, K, endpoint=False)
    rng = default_rng()
    thetas = possible_thetas[rng.integers(low=0, high=K, size=(m, ))]

    mask = np.zeros((N, N))
    # The locations table records the chosen signal locations.
    locations = [[] for i in range(m)]
    # This counter records how many signals we successfully placed.
    placed = 0
    placed_list = [0 for i in range(1)]
    max_trials = 5*m
    Y = np.zeros((N, N))
    for _ in range(max_trials):
        # Pick a candidate location for the upper-left corner of the signal
        candidate = np.random.randint(N-W, size=2)
        # Check if there is enough room, taking the separation rule into
        # account. That is, a square of size WxW with upper-left corner
        # specified by the candidate must be entirely free.
        if not((mask[candidate[0]:candidate[0]+2*W-1, candidate[1]:candidate[1]+2*W-1] == 1).any()):
            # Record the successful candidate
            locations[placed] = candidate
            # Mark the area as reserved
            mask[candidate[0]:candidate[0]+W, candidate[1]:candidate[1]+W] = 1
            index_rand = 0
            placed_list[index_rand] = placed_list[index_rand] + 1
            theta = thetas[placed]
            placed = placed + 1
            X_theta = rot_img_freqT(theta, c, kvals, Bk, L, T)
            Y[candidate[0] : candidate[0] + L, candidate[1] : candidate[1] + L] = X_theta
            # Stop if we placed sufficiently many signals successfully.
            if placed >= m:
                break
    locations = locations[0:placed]
    return Y, placed_list, locations
