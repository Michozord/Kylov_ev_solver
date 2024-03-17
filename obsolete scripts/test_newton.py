# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:11:03 2024

@author: Michal Trojanowski
"""


import numpy as np
from matplotlib import pyplot as plt
from ansatz_collocation import alpha_equidistant, alpha_chebyshev, indicator, gauss
from ansatz_minim import functional
from dff import plot_beta


def newton(x, target, T, tit=None, lam_min = 1e-8, q = 0.5):
    S = 400
    tau = 0.025
    F = functional(T, S, target)
    L = int(T/tau)
    f = F.gradient
    df = F.hessian
    i = 0
    lam = 1
    tol = 1e-5
    fig, ax = plt.subplots()
    while np.linalg.norm(f(x)) > tol:
        if i > 50:
            print("Newton did not converge!")
            return x
        if i%5 == 0:
            plot_beta(0, 20, x, tau, L, ax, label=f"i={i}")
            
        incr = np.linalg.solve(df(x), -f(x))
        fx_norm = np.linalg.norm(f(x))
        print(f"i = {i}, ||DF(x)|| = {fx_norm}")
        while lam >= lam_min and np.linalg.norm(f(x+lam*incr)) >= fx_norm:
            lam *= q
        if lam < lam_min:
            raise RuntimeError("Lambda too small!")
        else:
            x = x + lam*incr
            lam = min(1, lam/q)
            i += 1
        if tit:
            plt.title(tit)
    
    plot_beta(0, 20, x, tau, L, ax, label=f"i={i}")
    ax.grid()
    plt.xlim(0, 20)
    plt.xticks([2*n for n in range(11)])
    plt.legend()
    return x


if __name__ == "__main__":
    target = gauss 
    tau = 0.025
    for T in [1, 2.5, 5]:
        K = 2*int(T/tau)
        x_start = alpha_equidistant(0, 20, target, T, tau, K)
        newton(x_start, target, T, tit=f"Target indicator; T = {T}; start: Chebyshev collocation with K = {K}")
        print("T = {T} done")
        plt.show()
    