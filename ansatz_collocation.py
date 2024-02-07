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

if __name__ == "main":
    tau = 0.025
    Ts = [1, 2.5, 5, 10]
    target = indicator
    
    for T in Ts:
        fig, ax = plt.subplots()
        L = int(T/tau)
        Ks = [L, 2*L, 4*L]
        for K in Ks:
            mesh = np.linspace(omega_start, omega_end, num=K+1)
            beta_vec = np.array(list(map(target, mesh)))                # vector [target(omega_0), ..., target(omega_K)] - rhs of equation
            A = np.zeros((K+1, L+1))
            for k in range(K+1):
                A[k,:] = tau * q_omega(mesh[k], tau, L)
            A_pinv = np.linalg.pinv(A)
            alpha = A_pinv @ beta_vec
            print(np.linalg.norm(A @ alpha - beta_vec, ord = np.Inf))
            plot_beta(omega_start, omega_end, alpha, tau, L, ax, label="K="+str(K))
        ax.grid()
        plt.xlim(omega_start, omega_end)
        plt.xticks([2*n for n in range(11)])
        plt.legend()
        plt.title(f"Equidistant mesh, target function $\chi$, T = {T}, L = {L}")
        plt.show()
        
        
    for T in Ts:
        fig, ax = plt.subplots()
        L = int(T/tau)
        Ks = [L, 2*L, 4*L]
        for K in Ks:
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
            plot_beta(omega_start, omega_end, alpha, tau, L, ax, label="K="+str(K))
            # plt.plot(mesh, beta_vec, "x", color="black")
            
        ax.grid()
        plt.xlim(omega_start, omega_end)
        plt.xticks([2*n for n in range(11)])
        plt.legend()
        plt.title(f"Chebyshev mesh, target function $\chi$, T = {T}, L = {L}")
        plt.show()
        
        
        
    
    # target = indicator
    # a = 0
    # b = 6
    # Ts = [(2.5, 0.025), (5, 0.025), (5, 0.1), (10, 0.1)]
    # for T, tau in Ts:
    #     fig, ax = plt.subplots()
    #     L = int(T/tau)
    #     Ks = [L, 2*L, 4*L]
    #     for K in Ks:
    #         mesh = np.array(list(map(lambda j : np.cos((2*j+1)/(2*(K+1)) * np.pi), range(K+1))))     # K+1 Chebyshev roots in [-1, 1]
    #         mesh = 1/2*((a+b)*mesh + (b-a)*np.ones(len(mesh)))         # Chebyshev nodes in [a, b]
    #         print(sorted(mesh))
    #         beta_vec = np.array(list(map(target, mesh)))                # vector [target(omega_0), ..., target(omega_K)] - rhs of equation
    #         A = np.zeros((K+1, L+1))
    #         for k in range(K+1):
    #             A[k,:] = tau * q_omega(mesh[k], tau, L)
    #         A_pinv = np.linalg.pinv(A)
    #         alpha = A_pinv @ beta_vec
    #         print(np.linalg.norm(A @ alpha - beta_vec))
    #         plot_beta(omega_start, omega_end, alpha, tau, L, ax, label="K="+str(K))
    #     ax.grid()
    #     plt.xlim(omega_start, omega_end)
    #     plt.xticks([2*n for n in range(11)])
    #     plt.legend()
    #     plt.title(f"Chebyshev mesh, target function $\chi$, T = {T}, tau = {tau}")
    #     plt.show()
    
    # target = gauss    
    # for T in Ts:
    #     fig, ax = plt.subplots()
    #     L = int(T/tau)
    #     Ks = [L, 2*L, 4*L]
    #     for K in Ks:
    #         mesh = np.linspace(omega_start, omega_end, num=K+1)
    #         beta_vec = np.array(list(map(target, mesh)))                # vector [target(omega_0), ..., target(omega_K)] - rhs of equation
    #         A = np.zeros((K+1, L+1))
    #         for k in range(K+1):
    #             A[k,:] = tau * q_omega(mesh[k], tau, L)
    #         A_pinv = np.linalg.pinv(A)
    #         alpha = A_pinv @ beta_vec
    #         print(np.linalg.norm(A @ alpha - beta_vec, ord=np.Inf))
    #         plot_beta(omega_start, omega_end, alpha, tau, L, ax, label="K="+str(K))
    #         plt.plot(mesh, list(map(target, mesh)), ":", color="black")
    #     ax.grid()
    #     plt.xlim(omega_start, omega_end)
    #     plt.xticks([2*n for n in range(11)])
    #     plt.legend()
    #     plt.title(f"Equidistant mesh, target function Gauss, T = {T}, L = {L}")
    #     plt.show()
    
    
    
