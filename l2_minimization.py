# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 20:41:38 2024

@author: Michal Trojanowski
"""

from dff import *
from math import floor, ceil

def compute_alpha(om_end: float, L: int, tau: float, om_min: float, om_max: float):
    K = 10
    quad_mesh = np.linspace(0, om_end, num = K*om_end+1)
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
    om_min, om_max, om_end = 212, 214, 360
    tau = 1/om_end
    L = int(10/tau)
    alpha = compute_alpha(om_end, L, tau, om_min, om_max)
    ax = prepare_plot(0, om_end)
    Q = plot_beta(alpha, L, tau, 0, om_end, ax, label="L2 minimization")
    alpha = fourier_indicator(om_min, om_max, 10)
    plot_beta(alpha, L, tau, 0, om_end, ax, label="fourier", Q=Q)
    plt.legend()
    plt.show()