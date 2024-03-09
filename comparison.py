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


def compare(t):
    for T in [1, 2.5, 5, 10]:
        start, end = 0, 20
        omega_min, omega_max = 2, 4
        if t == "Gauss":
            target_name = "Gauss"
            target = gauss
        else:
            target_name = "$\\chi$"
            target = indicator
        
        for tau in [0.025, 0.01]:
            fig, ax = plt.subplots()
            L = int(T/tau)
            alpha = alpha_equidistant(start, end, target, T, tau, L)
            plot_beta(start, end, alpha, tau, L, ax, label=f"Collocation normaleq.; equidistant mesh with {L} points")
            alpha = alpha_equidistant(start, end, target, T, tau, 2*L)
            plot_beta(start, end, alpha, tau, L, ax, label=f"Collocation normaleq.; equidistant mesh with {2*L} points")
            alpha, K = alpha_chebyshev(start, end, target, T, tau, L)
            plot_beta(start, end, alpha, tau, L, ax, label=f"Collocation normaleq.; Chebyshev mesh with {K} points")
            alpha, K = alpha_chebyshev(start, end, target, T, tau, 2*L)
            plot_beta(start, end, alpha, tau, L, ax, label=f"Collocation normaleq.; Chebyshev mesh with {K} points")
            if t == "Gauss":
                alpha = alpha_fourier_gauss(T, omega_min, omega_max)
                plot_beta(start, end, alpha, tau, L, ax, label="Fourier")
            else:
                alpha = alpha_fourier_indicator(T, omega_min, omega_max)
                plot_beta(start, end, alpha, tau, L, ax, label="Fourier")
            
            ax.grid()
            plt.xlim(start, end)
            plt.xticks([2*n for n in range(11)])
            plt.legend()
            plt.title(f"Different methods: target function {target_name}, T = {T}, $\\tau$ = {tau}")
    plt.show()


if __name__ == "__main__":
    compare("Gauss")
    # compare("indicator")