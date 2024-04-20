# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 20:41:38 2024

@author: Michal Trojanowski
"""

from dff import *
from math import floor, ceil

def compute_alpha(om_end: float, L: int, tau: float, om_min: float, om_max: float, K: Optional[int] = 20):
    quad_mesh = np.linspace(0, om_end, num = int(K*om_end)+1)
    Q = q_eval_mat(quad_mesh, L, tau)
    rhs = (Q[ceil(K*om_min),:] + Q[floor(K*om_max),:])/2 * np.ones(L)
    for l in range(ceil(K*om_min)+1, floor(K*om_max)):
        rhs += Q[l,:]
    rhs /= K
    Q[0,:] *= 1/np.sqrt(2)
    Q[-1,:] *= 1/np.sqrt(2)
    X = 1/K * Q.transpose() @ Q 
    alpha = np.linalg.solve(X, rhs)/tau
    return alpha


if __name__ == "__main__":
    om_min, om_max = 12, 14
    tau = 0.0056
    om_end= 2/tau
    L = 200
    title = r"L2 minimization, T = " + str(L * tau)[0:7] + r", $\omega_{end}$ = " + str(om_end)[0:7] + ", L = " + str(L) + f", target intervall ({om_min}, {om_max})"
    ax = prepare_plot(0, om_end, title=title)
    ax2 = prepare_plot(om_end-0.2, om_end, title=title)
    
    alpha = compute_alpha(om_end, L, tau, om_min, om_max, K=10)
    Q1 = plot_beta(alpha, L, tau, 0, om_end, ax, label=f"quad. step {1/10}", cheb=False)
    Q2 = plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label=f"quad. step {1/10}", cheb=False)
    
    alpha = compute_alpha(om_end, L, tau, om_min, om_max)
    plot_beta(alpha, L, tau, 0, om_end, ax, label=f"quad. step {1/20}", cheb=False, Q=Q1)
    plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label=f"quad. step {1/20}", cheb=False, Q=Q2)
    
    alpha = fourier_indicator(om_min, om_max, L*tau)
    plot_beta(alpha, L, tau, 0, om_end, ax, label="Fourier")
    plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label="Fourier")
    
    ax.legend()
    ax2.legend()
    plt.show()
    
    