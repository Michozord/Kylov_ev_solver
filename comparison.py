# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:54:33 2024

@author: Michal Trojanowski
"""

from dff import *
from matplotlib import pyplot as plt
import numpy as np
from ansatz_collocation import alpha_equidistant, alpha_chebyshev, indicator, gauss
from dff_test import alpha_fourier_indicator, alpha_fourier_gauss
from ansatz_minimalisation import alpha_l2


def compare(t):
    for T in [1, 2.5, 5, 10]:
        start, end = 0, 360
        omega_min, omega_max = 2, 4
        if t == "Gauss":
            target_name = "Gauss"
            target = gauss
        else:
            target_name = "$\\chi$"
            target = indicator
        
        for tau in [1/end]:
            fig, ax = plt.subplots()
            L = int(T/tau)
            alpha = alpha_l2(L, omega_min, omega_max, end, 10*end, tau)
            plot_beta(start, end, alpha, tau, L, ax, label=f"L2-minimalisation; {10*end} quadrature points", multiply_with_tau=False)
            alpha = alpha_l2(L, omega_min, omega_max, end, 20*end, tau)
            plot_beta(start, end, alpha, tau, L, ax, label=f"L2-minimalisation; {20*end} quadrature points", multiply_with_tau=False)
            alpha = alpha_equidistant(start, end, target, T, tau, 2*L)
            plot_beta(start, end, alpha, tau, L, ax, label=f"Collocation; equidistant mesh with {2*L} points")
            alpha, K = alpha_chebyshev(start, end, target, T, tau, 2*L)
            plot_beta(start, end, alpha, tau, L, ax, label=f"Collocation; Chebyshev mesh with {K} points")
            alpha = alpha_equidistant(start, end, target, T, tau, 4*L)
            plot_beta(start, end, alpha, tau, L, ax, label=f"Collocation; equidistant mesh with {4*L} points")
            alpha, K = alpha_chebyshev(start, end, target, T, tau, 4*L)
            plot_beta(start, end, alpha, tau, L, ax, label=f"Collocation; Chebyshev mesh with {K} points")
            if t == "Gauss":
                alpha = alpha_fourier_gauss(T, omega_min, omega_max)
                plot_beta(start, end, alpha, tau, L, ax, label="Fourier")
            else:
                alpha = alpha_fourier_indicator(T, omega_min, omega_max)
                plot_beta(start, end, alpha, tau, L, ax, label="Fourier")
            
            ax.grid()
            plt.xlim(start, end)
            plt.ylim(-1, 3)
            plt.legend()
            plt.title(f"Different methods: target function {target_name}, T = {T}")
    plt.show()


if __name__ == "__main__":
    # compare("Gauss")
    compare("indicator")