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


def alpha_chebyshev(omega_start, omega_end, omega_min, omega_max, target_generator, T, tau, K, method=""):
    L = int(T/tau)
    target = target_generator(omega_min, omega_max)
    if K%3 != 0:
        K = K + 3 - K%3
    s = int(K/3)
    Chebyshev = np.array(list(map(lambda j: np.cos((2*j+1)/(2*s)*np.pi), range(s))))
    mesh = np.zeros(K)
    mesh[0:s] = 1/2 * ((omega_min - omega_start) * Chebyshev + (omega_start+omega_min)*np.ones(s))
    mesh[s:2*s] = 1/2 * ((omega_max - omega_min) * Chebyshev + (omega_max+omega_min)*np.ones(s))
    mesh[2*s:] = 1/2 * ((omega_end - omega_max) * Chebyshev + (omega_max+omega_end)*np.ones(s))
    beta_vec = np.array(list(map(target, mesh)))                # vector [target(omega_0), ..., target(omega_K)] - rhs of equation
    A = np.zeros((K, L))
    for k in range(K):
        A[k,:] = tau * q_omega(mesh[k], tau, L)
    if L == K:
        alpha = np.linalg.solve(A, beta_vec)    
    elif method == "QR":
        Q, R = np.linalg.qr(A, mode='complete')
        R = R[:L,:]     # upper triangle square matrix
        c = (Q.transpose()@beta_vec)[:L]     # upper part of Q* @ beta_vec
        alpha = np.linalg.solve(R, c)
    else:
        alpha = np.linalg.solve(A.transpose()@A, A.transpose()@beta_vec)
    return alpha, K


if __name__ == "__main__":
    omega_start, omega_end = 0, 360
    tau = 1/omega_end           
    
    Ts = [1, 2.5, 5, 10]
    
    for T in Ts:
        fig1 = plt.figure()
        ax1 = plt.subplot()
        fig2 = plt.figure()
        ax2 = plt.subplot()
        L = int(T/tau)
        for K in [L+1, 2*L, 5*L, 10*L]:
            for omega_min, omega_max in [(2, 4), (6,8), (20, 22), (100, 110)]: 
                # EQUIDISTANT MESH:
                alpha = alpha_equidistant(omega_start, omega_end, omega_min, omega_max, indicator, T, tau, K)
                write_to_file(f"Collocation/linear minimalization, equidistant mesh, ({omega_min}, {omega_max}), T = {T}, L = {L}", f"K = {K}, solve ATAx=ATb", alpha, tau, L)
                # write_to_file(f"Collocation/linear minimalization, equidistant mesh, ({omega_min}, {omega_max}), T = {T}, L = {L}", f"K = {K}", alpha, tau, L)
                
                # CHEBYSHEV MESH:
                alpha, K = alpha_chebyshev(omega_start, omega_end, omega_min, omega_max, indicator, T, tau, K)
                write_to_file(f"Collocation/linear minimalization, Chebyshev mesh, ({omega_min}, {omega_max}), T = {T}, L = {L}", f"K = {K}, solve ATAx=ATb", alpha, tau, L)
                # write_to_file(f"Collocation/linear minimalization, Chebyshev mesh, ({omega_min}, {omega_max}), T = {T}, L = {L}", f"K = {K}", alpha, tau, L)
                
                plot_beta(0, omega_end, alpha, tau, L, ax1, label=f"({omega_min}, {omega_max}): K = {K}")
                plot_beta(omega_end-0.1, omega_end, alpha, tau, L, ax2, label=f"({omega_min}, {omega_max}): K = {K}")
            print(f"T = {T}, K={K} done")
        plt.sca(ax1)
        ax1.legend()
        ax1.grid()
        plt.xlim(0, omega_end)
        plt.ylim(-1, 2)
        # plt.title(f"Collocation/linear minimalization, equidistant mesh, T={T}")
        plt.title(f"Collocation/linear minimalization, Chebyshev mesh, T={T}")
        plt.sca(ax2)
        ax2.legend()
        ax2.grid()
        plt.xlim(omega_end-0.1, omega_end)
        plt.ylim(-1, 2)
        # plt.title(f"Collocation/linear minimalization, equidistant mesh, T={T}")
        plt.title(f"Collocation/linear minimalization, Chebyshev mesh, T={T}")
        print(f"T={T} done")
    plt.show()
    
    
