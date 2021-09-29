# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 18:28:03 2021

@author: Shay Kreymer
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


from Utils.fb_funcs import expand_fb, rot_img_freq, calcT, rot_img_freqT
from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots

# plt.close("all")

if __name__ == '__main__':
    X = plt.imread("./images/molecule9.png")
    X = np.random.rand(5, 5)
    L = np.shape(X)[0]
    X = X
    W = 2*L-1 # L for arbitrary spacing distribution, 2*L-1 for well-separated
    np.random.seed(1)
    N = 55
    ne = 100
    B, z, roots, kvals, nu = expand_fb(X, ne)
    T = calcT(nu, kvals)
    BT = B @ T.H
    c = np.real(T @ z)
    z = T.H@c
    Xrec = np.reshape(np.real(B @ z), np.shape(X))
    XrecT = np.reshape(np.real(BT @ c), np.shape(X))
    theta = np.random.uniform(0, 2*np.pi)
    Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
    for i in range(nu):
        Bk[ :, :, i] = np.fft.fft2(np.pad(np.reshape(B[ :, i], (L, L)), L//2))
    
    Xrot_freq = rot_img_freq(theta, z, kvals, Bk, L)
    Xrot_freqT = rot_img_freqT(theta, c, kvals, Bk, L, T)
    y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, 0.30*(N/L)**2, T, seed=100)

    gamma = s[0]*(L/N)**2
    SNR = 2
    sigma2 = np.linalg.norm(Xrec)**2 / (SNR * np.pi * (L//2)**2)
    y1 = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
    
    SNR = 0
    sigma2 = 0
    y2 = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
    
    SNR = 50
    sigma2 = np.linalg.norm(Xrec)**2 / (SNR * np.pi * (L//2)**2)
    y3 = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
    
    width = 3.487
    height = width / 1.618
    # plt.close("all")
    ttls = ['SNR = 10', 'SNR = 0.1']
    ys = [y3, y1, y2]
    for n in range(3):
        fig = plt.figure()#subplots(1, 2)
    
        ax = plt.axes()
        im = ax.imshow(ys[n], cmap="gray")   
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=20)

        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

        fig.set_size_inches(4*width, 4*height)
        if n == 0:
            fig.savefig(r'C:\Users\kreym\Google Drive\PhD\Documents\MTD-2D-EM-ICASSP\figures\Micrographs_noise_a.pdf', bbox_inches='tight')
        else:
            if n == 1:
                fig.savefig(r'C:\Users\kreym\Google Drive\PhD\Documents\MTD-2D-EM-ICASSP\figures\Micrographs_noise_b.pdf', bbox_inches='tight')
            else:
                fig.savefig(r'C:\Users\kreym\Google Drive\PhD\Documents\MTD-2D-EM-ICASSP\figures\Micrographs_noise_clean.pdf', bbox_inches='tight')

