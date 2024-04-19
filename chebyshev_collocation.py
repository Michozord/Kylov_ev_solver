# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:47:48 2024

@author: Michal Trojanowski
"""

from dff import *

def chebyshev_nodes(a: float, b: float, N: int) -> np.array:
    nodes = np.array([np.cos((2*k+1)/(2*N) * np.pi) for k in range(N)])
    nodes = (a+b)/2 + (b-a)/2 * nodes
    return nodes

def compute_alpha(om_end: float, L: int, K: int, tau: float, target: Callable) -> np.array:
    mesh = chebyshev_nodes(0, om_end, K)
    Q = q_eval_mat(mesh, L, tau, cheb=True)
    rhs = 1/tau * np.array(list(map(target, mesh)))
    if K==L:
        print(f"cond(Q) = {np.linalg.cond(Q)}, det(Q) = {np.linalg.det(Q)}")
        alpha = np.linalg.solve(Q, rhs)
        return alpha
    else:
        alpha = np.linalg.lstsq(Q, rhs)[0]
        return alpha
    
    
if __name__ == "__main__":
    om_min, om_max = 10, 12
    target = indicator(om_min, om_max)
    tau = 0.0056
    om_end= 2/tau
    L = 500
    alpha = compute_alpha(om_end, L, 10*L, tau, target)
    ax = prepare_plot(0, om_end)
    ax2 = prepare_plot(om_end-0.2, om_end)
    plot_beta(alpha, L, tau, 0, om_end, ax, label="approximation", cheb=True)
    plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label="approximation", cheb=True)
    plot_nodes(chebyshev_nodes(0, om_end, L), target, ax)
    plot_nodes(chebyshev_nodes(0, om_end, L), target, ax2)
    alpha = fourier_indicator(om_min, om_max, L*tau)
    plot_beta(alpha, L, tau, 0, om_end, ax, label="fourier", num=50000)
    plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label="fourier", num=50000)
    plt.legend()
    plt.show()
        