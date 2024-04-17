# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:27:30 2024

@author: Michal Trojanowski
"""

from dff import *
import numpy as np
from alpha_parser import *
from matplotlib import pyplot as plt

def indicator(omega_min, omega_max):
    chi = lambda x: 1 if x >= omega_min and x <= omega_max else 0
    return chi

def gauss(x):
    omega_mid = (omega_min+omega_max)/2
    return np.exp(-(x-omega_mid)*(x-omega_mid))


def alpha_equidistant(omega_start, omega_end, omega_min, omega_max, target_generator, T, tau, K, method=""):
    L = int(T/tau)
    target = target_generator(omega_min, omega_max)
    mesh = np.linspace(omega_start, omega_end, num=K)
    beta_vec = np.array(list(map(target, mesh)))                # vector [target(omega_0), ..., target(omega_K-1)] - rhs of equation
    A = np.zeros((K, L))
    for k in range(K):
        A[k,:] = tau * q_omega(mesh[k], tau, L)
    if L == K:          # linear system of equations
        alpha = np.linalg.solve(A, beta_vec)
    elif method == "QR":               
        # linear minimalisation problem:
        # A = Q  [ R ]
        #        [ 0 ]
        # => 
        # Q* b = [ c ]
        #        [ d ]
        # =>
        # solve R alpha = c
        Q, R = np.linalg.qr(A, mode='complete')
        R = R[:L,:]     # upper triangle square matrix
        c = (Q.transpose()@beta_vec)[:L]     # upper part of Q* @ beta_vec
        alpha = np.linalg.solve(R, c)
    else:
        alpha = np.linalg.solve(A.transpose()@A, A.transpose()@beta_vec)
    return alpha


if __name__ == "__main__":
    omega_start, omega_end = 0, 360
    tau = 1/omega_end           
    
    Ts = [1, 5]
    
    for T in Ts:
        fig1 = plt.figure()
        ax1 = plt.subplot()
        fig2 = plt.figure()
        ax2 = plt.subplot()
        L = int(T/tau)
        for K in [2*L, 5*L, 10*L]:
            for omega_min, omega_max in [(2,4), (100, 110)]: #[(2, 4), (6,8), (20, 22), (100, 110)]:
                # GAUSS NORMAL EQUATION:
                alpha = alpha_equidistant(omega_start, omega_end, omega_min, omega_max, indicator, T, tau, K)
                # write_to_file(f"Collocation/linear minimalization, equidistant mesh, ({omega_min}, {omega_max}), T = {T}, L = {L}", f"K = {K}, solve ATAx=ATb", alpha, tau, L)
                # SOLVE WITH QR DECOMPOSITION:
                # alpha = alpha_equidistant(omega_start, omega_end, omega_min, omega_max, indicator, T, tau, K, method="QR")
                # write_to_file(f"Collocation/linear minimalization, equidistant mesh, ({omega_min}, {omega_max}), T = {T}, L = {L}", f"K = {K}", alpha, tau, L)
                
                plot_beta(0, omega_end, alpha, tau, L, ax1, label=f"({omega_min}, {omega_max}): K = {K}")
                plot_beta(omega_end-0.1, omega_end, alpha, tau, L, ax2, label=f"({omega_min}, {omega_max}): K = {K}")
            print(f"T = {T}, K={K} done")
        plt.sca(ax1)
        ax1.legend()
        ax1.grid()
        plt.xlim(0, omega_end)
        plt.ylim(-1, 2)
        plt.title(f"Collocation/linear minimalization, equidistant mesh, T={T}")
        plt.sca(ax2)
        ax2.legend()
        ax2.grid()
        plt.xlim(omega_end-0.1, omega_end)
        plt.ylim(-1, 2)
        plt.title(f"Collocation/linear minimalization, equidistant mesh, T={T}")
        print(f"T={T} done")
    plt.show()
    
    
