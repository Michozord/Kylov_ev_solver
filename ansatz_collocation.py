# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:27:30 2024

@author: Michal Trojanowski
"""

from dff import *
from matplotlib import pyplot as plt
import numpy as np

omega_min, omega_max = 2, 4

def indicator(x):
    return 1 if x >= omega_min and x <= omega_max else 0

def gauss(x):
    omega_mid = (omega_min+omega_max)/2
    return np.exp(-(x-omega_mid)*(x-omega_mid))


def alpha_equidistant(omega_start, omega_end, target, T, tau, K):
    L = int(T/tau)
    mesh = np.linspace(omega_start, omega_end, num=K)
    beta_vec = np.array(list(map(target, mesh)))                # vector [target(omega_0), ..., target(omega_K-1)] - rhs of equation
    A = np.zeros((K, L))
    for k in range(K):
        A[k,:] = tau * q_omega(mesh[k], tau, L)
    if L == K:          # linear system of equations
        alpha = np.linalg.solve(A, beta_vec)
    else:               # linear minimalisation problem
        alpha = np.linalg.solve(A.transpose()@A, A.transpose()@beta_vec)
    return alpha


def alpha_chebyshev(omega_start, omega_end, target, T, tau, K):
    L = int(T/tau)
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
    else:
        alpha = np.linalg.solve(A.transpose()@A, A.transpose()@beta_vec)
    return alpha, K


if __name__ == "__main__":
    omega_start, omega_end = 0, 360
    tau = 2/omega_end           # approx. 0.0056
    omega_min, omega_max = 2, 4
    Ts = [1, 2.5, 5, 10]
    target_name = "$\chi$"
    target = indicator
    # target_name = "Gauss"
    # target = gauss
    
    for T in Ts:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot()
        fig3 = plt.figure()
        ax3 = fig3.add_subplot()
        axes = [ax1, ax3]
        
        L = int(T/tau)
        Ks = [L, 4*L, 10*L]
        for K in Ks:
            alpha = alpha_equidistant(omega_start, omega_end, target, T, tau, K)
            plot_beta(omega_start, omega_end, alpha, tau, L, ax1, label="K="+str(K))
            if K > 2*L+3:
                plot_beta(omega_start, omega_end, alpha, tau, L, ax3, label="K="+str(K))

        for ax in axes:
            plt.sca(ax)
            plt.xlim(omega_start, omega_end)
            plt.title(f"Equidistant mesh, target function {target_name}, T = {T}, L = {L}")
            ax.legend()
            ax.grid()
            plt.ylim(-1, 10)
        
    
    for T in Ts:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot()
        fig3 = plt.figure()
        ax3 = fig3.add_subplot()
        axes = [ax1, ax3]

        L = int(T/tau)
        Ks = [L, 4*L, 10*L]
        for K in Ks:
            alpha, K = alpha_chebyshev(omega_start, omega_end, target, T, tau, K)
            plot_beta(omega_start, omega_end, alpha, tau, L, ax1, label="K="+str(K))
            if K > 2*L+3:
                plot_beta(omega_start, omega_end, alpha, tau, L, ax3, label="K="+str(K))

        for ax in axes:
            plt.sca(ax)
            plt.xlim(omega_start, omega_end)
            plt.title(f"Chebyshev mesh, target function {target_name}, T = {T}, L = {L}")
            ax.legend()
            ax.grid()
            plt.ylim(-1, 10)
    
    plt.show()
        
    
    
    
