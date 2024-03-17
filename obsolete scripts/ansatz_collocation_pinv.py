# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:27:30 2024

@author: Michal Trojanowski
"""

from dff import *
from matplotlib import pyplot as plt
import numpy as np


omega_min = 2
omega_max = 4
omega_start = 0
omega_end = 20


def indicator(x):
    return 1 if x >= omega_min and x <= omega_max else 0

def gauss(x):
    omega_mid = (omega_min+omega_max)/2
    return np.exp(-(x-omega_mid)*(x-omega_mid))


def alpha_equidistant(omega_start, omega_end, target, T, tau, K):
    L = int(T/tau)
    mesh = np.linspace(omega_start, omega_end, num=K+1)
    beta_vec = np.array(list(map(target, mesh)))                # vector [target(omega_0), ..., target(omega_K)] - rhs of equation
    A = np.zeros((K+1, L+1))
    for k in range(K+1):
        A[k,:] = tau * q_omega(mesh[k], tau, L)
    A_pinv = np.linalg.pinv(A)
    alpha = A_pinv @ beta_vec
    # print(np.linalg.norm(A @ alpha - beta_vec, ord = np.Inf))
    print(f"alpha(0) = {alpha[0]}, alpha(tau) = {alpha[1]}")
    return alpha


def alpha_chebyshev(omega_start, omega_end, target, T, tau, K):
    L = int(T/tau)
    if (K+1)%3 != 0:
        K = K + 2 - K%3
    s = int((K+1)/3)
    Chebyshev = np.array(list(map(lambda j: np.cos((2*j+1)/(2*s)*np.pi), range(s))))
    mesh = np.zeros(K+1)
    mesh[0:s] = 1/2 * ((omega_min - omega_start) * Chebyshev + (omega_start+omega_min)*np.ones(s))
    mesh[s:2*s] = 1/2 * ((omega_max - omega_min) * Chebyshev + (omega_max+omega_min)*np.ones(s))
    mesh[2*s:] = 1/2 * ((omega_end - omega_max) * Chebyshev + (omega_max+omega_end)*np.ones(s))
    beta_vec = np.array(list(map(target, mesh)))                # vector [target(omega_0), ..., target(omega_K)] - rhs of equation
    A = np.zeros((K+1, L+1))
    for k in range(K+1):
        A[k,:] = tau * q_omega(mesh[k], tau, L)
    A_pinv = np.linalg.pinv(A)
    alpha = A_pinv @ beta_vec
    # print(np.linalg.norm(A @ alpha - beta_vec, ord = np.Inf))
    print(f"alpha(0) = {alpha[0]}, alpha(tau) = {alpha[1]}")
    return alpha, K


if __name__ == "__main__":
    tau = 0.025
    Ts = [1, 2.5, 5, 10]
    # target_name = "$\chi$"
    # target = indicator
    target_name = "Gauss"
    target = gauss
    
    for T in Ts:
        fig, ax = plt.subplots()
        L = int(T/tau)
        Ks = [L, 2*L, 4*L]
        for K in Ks:
            alpha = alpha_equidistant(omega_start, omega_end, target, T, tau, K)
            plot_beta(omega_start, omega_end, alpha, tau, L, ax, label="K="+str(K))
        ax.grid()
        plt.xlim(omega_start, omega_end)
        plt.xticks([2*n for n in range(11)])
        plt.legend()
        plt.title(f"Equidistant mesh, target function {target_name}, pseudoinverse, T = {T}, L = {L}")
           
    for T in Ts:
        fig, ax = plt.subplots()
        L = int(T/tau)
        Ks = [L, 2*L, 4*L]
        for K in Ks:
            alpha, K = alpha_chebyshev(omega_start, omega_end, target, T, tau, K)
            plot_beta(omega_start, omega_end, alpha, tau, L, ax, label="K="+str(K))
            
        ax.grid()
        plt.xlim(omega_start, omega_end)
        plt.xticks([2*n for n in range(11)])
        plt.legend()
        plt.title(f"Chebyshev mesh, target function {target_name}, pseudoinverse, T = {T}, L = {L}")
    plt.show()
        
    
    
    
