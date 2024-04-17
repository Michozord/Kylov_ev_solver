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
        ax1 = plt.subplot()
        fig2 = plt.figure()
        ax2 = plt.subplot()
        for omega_min, omega_max in [(2, 4), (6,8), (20, 22), (100, 110)]:
        # for omega_min, omega_max in [(2, 4)]:
            L = int(T/tau)
            for K in [2*L, 5*L, 10*L]:
                # alpha, tau, L = read_file(f"Collocation/linear minimalization, equidistant mesh, ({omega_min}, {omega_max}), T = {T}, L = {L}", f"K = {K}")
                # plot_beta(omega_start, omega_end+10, alpha, tau, L, ax1, label=f"({omega_min}, {omega_max}): K = {K}, QR")
                # plot_beta(omega_end-0.1, omega_end, alpha, tau, L, ax2, label=f"({omega_min}, {omega_max}): K = {K}, QR")
                alpha, tau, L = read_file(f"Collocation/linear minimalization, equidistant mesh, ({omega_min}, {omega_max}), T = {T}, L = {L}", f"K = {K}, solve ATAx=ATb")
                plot_beta(omega_start, omega_end+10, alpha, tau, L, ax1, label=f"({omega_min}, {omega_max}): K = {K}, Gauss")
                # plot_beta(omega_end-0.1, omega_end, alpha, tau, L, ax2, label=f"({omega_min}, {omega_max}): K = {K}, Gauss")
            
            
            plt.sca(ax1)
            plt.xlim(omega_start, omega_end)
            plt.sca(ax2)
            plt.xlim(omega_end-0.1, omega_end)
            for ax in [ax1, ax2]:
                plt.sca(ax)
                plt.title(f"Equidistant mesh, T = {T}, L = {L}")
                ax.legend()
                plt.grid()
                plt.ylim(-1, 2)
            print(f"T = {T}, ({omega_min}, {omega_max}) done")
    plt.show()
        

# L2 MINIMALISATION APPROACH: TARGET INDICATOR  
def plt_l2_ind():
    omega_start, omega_end = 0, 360
    tau = 1/omega_end      

    Ts = [1, 2.5, 5, 10]
    
    for T in Ts:
        fig1 = plt.figure()
        ax1 = plt.subplot()
        fig2 = plt.figure()
        ax2 = plt.subplot()
        L = int(T/tau)
        for omega_min, omega_max in [(2, 4), (6,8), (20, 22), (100, 110)]:
            for M in [2*omega_end, 10*omega_end]:
                alpha, tau, L = read_file(f"L2 minimalisation, target ({omega_min}, {omega_max}), T = {T}, L = {L}", f"M = {M} quadrature points")
                plot_beta(0, omega_end + 10, alpha, tau, L, ax1, label=f"({omega_min}, {omega_max}): M = {M}")
                plot_beta(omega_end - 0.1, omega_end, alpha, tau, L, ax2, label=f"({omega_min}, {omega_max}): M = {M}")
            print(f"T = {T}, ({omega_min}, {omega_max}): done")
        plt.sca(ax1)
        plt.xlim(omega_start, omega_end)
        plt.title(f"L2 minimalisation, T = {T}, L = {L}")
        ax1.legend()
        ax1.grid()
        plt.ylim(-0.5, 2)
        plt.sca(ax2)
        plt.xlim(omega_end-0.1, omega_end)
        plt.title(f"L2 minimalisation, T = {T}, L = {L}")
        ax2.legend()
        ax2.grid()
        plt.ylim(-0.5, 2)
    plt.show()


def comparison():
    Ts = [1, 2.5, 5, 10]
    omega_end = 360
    tau = 1/omega_end
    for T in Ts:
        L = int(T/tau)
        for omega_min, omega_max in [(2, 4), (6,8), (20, 22), (100, 110)]:
            fig, ax = plt.subplots()
            alpha, tau, L = read_file(f"L2 minimalisation, target ({omega_min}, {omega_max}), T = {T}, L = {L}", f"M = {10*omega_end} quadrature points")
            plot_beta(0, omega_end+10, alpha, tau, L, ax, label="L2 minimalization")
            for K in [2*L, 5*L, 10*L]:
                alpha, tau, L = read_file(f"Collocation/linear minimalization, equidistant mesh, ({omega_min}, {omega_max}), T = {T}, L = {L}", f"K = {K}, solve ATAx=ATb")
                plot_beta(0, omega_end+10, alpha, tau, L, ax, label=f"linear minimalization; equidistant mesh; K = {K}; Gauss eq.")
            
            alpha = alpha_fourier_indicator(T, omega_min, omega_max)
            plot_beta(0, omega_end+10, alpha, tau, L, ax, label="Fourier")
            
            plt.grid()
            plt.legend()
            plt.xlim(0, omega_end)
            plt.ylim(-1, 2)
            plt.title(f"Different methods: target indicator ({omega_min}, {omega_max}), T = {T}")
            print(f"T = {T}, ({omega_min}, {omega_max}): done")
    plt.show()




if __name__ == "__main__":
    # COLLOCATION APROACH: TARGET INDICATOR AND EQUIDISTANT MESH
    plot_col_ind_eq()
    # L2 MINIMALISATION APPROACH: TARGET INDICATOR  
    # plt_l2_ind()
    # COMPARE DIFFERENT METHODS
    # comparison()