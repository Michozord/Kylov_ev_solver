# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 20:41:38 2024

@author: Michal Trojanowski
"""

from dff import *
from math import floor, ceil
from equidistant_collocation import compute_alpha as alpha_eq

def compute_alpha(om_end: float, L: int, tau: float, om_min: float, om_max: float, K: Optional[int] = 20):
    quad_mesh = np.linspace(0, om_end - 1/K, num = int(K*om_end)) + 1/(2*K)
    Q = q_eval_mat(quad_mesh, L, tau)
    rhs = np.zeros(L)
    for l in range(ceil(K*om_min), floor(K*om_max)+1):
        rhs += Q[l,:]
    rhs /= K
    X = 1/K * Q.transpose() @ Q 
    print(f"det(X) = {np.linalg.det(X)}, cond(X) = {np.linalg.cond(X)}")
    alpha = np.linalg.solve(X, rhs)/tau
    return alpha


if __name__ == "__main__":
    om_min, om_max = 212, 214
    tau = 0.0056
    om_end= 2/tau
    L = 200
    title = r"L2 minimization, T = " + str(L * tau)[0:7] + r", $\omega_{end}$ = " + str(om_end)[0:7] + ", L = " + str(L) + f", target intervall ({om_min}, {om_max})"
    ax = prepare_plots(0, om_end, title=title)
    ax2 = prepare_plots(om_end-0.2, om_end, title=title)
    
    alpha = compute_alpha(om_end, L, tau, om_min, om_max, K=1000)
    Q1 = plot_beta(alpha, L, tau, 0, om_end, ax, label=f"quad. step {1/1000}", cheb=True)
    Q2 = plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label=f"quad. step {1/1000}", cheb=True)
    
    alpha = compute_alpha(om_end, L, tau, om_min, om_max)
    plot_beta(alpha, L, tau, 0, om_end, ax, label=f"quad. step {1/20}", cheb=True, Q=Q1)
    plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label=f"quad. step {1/20}", cheb=True, Q=Q2)
    
    alpha = fourier_indicator(om_min, om_max, L*tau)
    plot_beta(alpha, L, tau, 0, om_end, ax, label="Fourier")
    plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label="Fourier")
    
    ax.legend()
    ax2.legend()
    plt.show()
    
    