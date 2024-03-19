# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:48:09 2024

@author: Michal Trojanowski
"""

import numpy as np
from matplotlib import pyplot as plt
from dff import *
from ansatz_collocation import indicator
from alpha_parser import read_file
from dff_test import alpha_fourier_indicator


# COLLOCATION APROACH: TARGET INDICATOR AND EQUIDISTANT MESH
def plot_col_ind_eq():
    omega_start, omega_end = 0, 360
    tau = 1/omega_end           
    omega_min, omega_max = 2, 4
    Ts = [1, 2.5, 5, 10]
    target_name = r"$\chi$"
    target = indicator
    for T in Ts:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot()
        fig3 = plt.figure()
        ax3 = fig3.add_subplot()
        axes = [ax1, ax3]
        
        L = int(T/tau)
        Ks = [L, L+1, 4*L, 10*L]
        for K in Ks:
            alpha, tau, L = read_file(f"Equidistant mesh, target function {target_name}, T = {T}, L = {L}", f"K = {K}")
            plot_beta(omega_start, omega_end+10, alpha, tau, L, ax1, label="K="+str(K))
            if K > 2*L:
                plot_beta(omega_start, omega_end+10, alpha, tau, L, ax3, label="K="+str(K))
            
        for ax in axes:
            plt.sca(ax)
            plt.xlim(omega_start, omega_end)
            plt.title(f"Equidistant mesh, target function {target_name}, T = {T}, L = {L}")
            ax.legend()
            ax.grid()
            plt.ylim(-1, 3)
    plt.show()
        
        
        
# COLLOCATION APROACH: TARGET INDICATOR AND CHEBYSHEV MESH
def plt_col_ind_cheb():
    omega_start, omega_end = 0, 360
    tau = 1/omega_end           
    omega_min, omega_max = 2, 4
    Ts = [1, 2.5, 5, 10]
    target_name = r"$\chi$"
    target = indicator
    for T in Ts:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot()
        fig3 = plt.figure()
        ax3 = fig3.add_subplot()
        axes = [ax1, ax3]
        
        L = int(T/tau)
        Ks = [L, L+3, 4*L, 10*L]
        for K in Ks:
            alpha, tau, L = read_file(f"Chebyshev mesh, target function {target_name}, T = {T}, L = {L}", f"K = {K}")
            plot_beta(omega_start, omega_end+10, alpha, tau, L, ax1, label="K="+str(K))
            if K > 2*L:
                plot_beta(omega_start, omega_end+10, alpha, tau, L, ax3, label="K="+str(K))
            
        for ax in axes:
            plt.sca(ax)
            plt.xlim(omega_start, omega_end)
            plt.title(f"Chebyshev mesh, target function {target_name}, T = {T}, L = {L}")
            ax.legend()
            ax.grid()
            plt.ylim(-1, 3)
    plt.show()


# L2 MINIMALISATION APPROACH: TARGET INDICATOR  
def plt_l2_ind():
    omega_start, omega_end = 0, 360
    tau = 1/omega_end           
    omega_min, omega_max = 2, 4
    Ts = [1, 2.5, 5, 10]
    target_name = r"$\chi$"
    
    for T in Ts:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot()
        L = int(T/tau)
        for M in [2*omega_end, 10*omega_end, 20*omega_end]:
            alpha, tau, L = read_file(f"L2 minimalisation, target function {target_name}, T = {T}, L = {L}", f"M = {M} quadrature points")
            plot_beta(0, omega_end + 10, alpha, tau, L, ax1, label=f"M = {M} quadrature points")
        plt.xlim(omega_start, omega_end)
        plt.title(f"L2 minimalisation, target function {target_name}, T = {T}, L = {L}")
        ax1.legend()
        ax1.grid()
        plt.ylim(-0.5, 3)
    plt.show()
    
def comparison():
    for T in [1, 2.5, 5, 10]:
        start, end = 0, 360
        omega_min, omega_max = 2, 4
        target_name = "$\\chi$"
        target = indicator
        
        for tau in [1/end]:
            fig, ax = plt.subplots()
            L = int(T/tau)
            alpha, _, __ = read_file(f"L2 minimalisation, target function $\chi$, T = {T}, L = {L}", f"M = {2*end} quadrature points")
            plot_beta(start, end+10, alpha, tau, L, ax, label=f"L2-minimalisation; {2*end} quadrature points")
            alpha, _, __ = read_file(f"L2 minimalisation, target function $\chi$, T = {T}, L = {L}", f"M = {20*end} quadrature points")
            plot_beta(start, end+10, alpha, tau, L, ax, label=f"L2-minimalisation; {20*end} quadrature points")
            
            alpha, _, __ = read_file(f"Equidistant mesh, target function $\chi$, T = {T}, L = {L}", f"K = {4*L}")
            plot_beta(start, end+10, alpha, tau, L, ax, label=f"Collocation; equidistant mesh with {4*L} points")
            alpha, _, __ = read_file(f"Equidistant mesh, target function $\chi$, T = {T}, L = {L}", f"K = {10*L}")
            plot_beta(start, end+10, alpha, tau, L, ax, label=f"Collocation; equidistant mesh with {10*L} points")

            alpha, _, __ = read_file(f"Chebyshev mesh, target function $\chi$, T = {T}, L = {L}", f"K = {4*L}")
            plot_beta(start, end+10, alpha, tau, L, ax, label=f"Collocation; Chebyshev mesh with {4*L} points")
            alpha, _, __ = read_file(f"Chebyshev mesh, target function $\chi$, T = {T}, L = {L}", f"K = {10*L}")
            plot_beta(start, end+10, alpha, tau, L, ax, label=f"Collocation; Chebyshev mesh with {10*L} points")

            alpha = alpha_fourier_indicator(T, omega_min, omega_max)
            plot_beta(start, end+10, alpha, tau, L, ax, label="Fourier")
            
            ax.grid()
            plt.xlim(start, end)
            plt.ylim(-1, 3)
            plt.legend()
            plt.title(f"Different methods: target function {target_name}, T = {T}")
    plt.show()
    



if __name__ == "__main__":
    # COLLOCATION APROACH: TARGET INDICATOR AND EQUIDISTANT MESH
    # plot_col_ind_eq()
    # COLLOCATION APROACH: TARGET INDICATOR AND CHEBYSHEV MESH
    # plt_col_ind_cheb()
    # L2 MINIMALISATION APPROACH: TARGET INDICATOR  
    # plt_l2_ind()
    # COMPARE DIFFERENT METHODS
    comparison()