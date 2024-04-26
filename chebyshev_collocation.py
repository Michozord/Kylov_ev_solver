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
        # alpha = np.linalg.solve(Q, rhs)
        alpha = np.linalg.lstsq(Q, rhs)[0]
        return alpha
    else:
        alpha = np.linalg.lstsq(Q, rhs)[0]
        return alpha
    
    
if __name__ == "__main__":
    om_min, om_max = 12, 14
    # target = indicator(om_min, om_max)
    target = gauss(13)
    tau = 0.0056
    om_end= 2/tau
    L = 1000
    title = r"Chebyshev collocation, T = " + str(L * tau)[0:7] + r", $\omega_{end}$ = " + str(om_end)[0:7] + ", L = " + str(L) + f", target intervall ({om_min}, {om_max})"
    ax = prepare_plot(0, om_end, title=title)
    ax2 = prepare_plot(om_end-0.2, om_end, title=title)
    
    alpha = compute_alpha(om_end, L, L, tau, target)
    Q1 = plot_beta(alpha, L, tau, 0, om_end, ax, label=f"colloc K={L} knots", cheb=True)
    Q2 = plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label=f"colloc K={L} knots", cheb=True)
    
    alpha = compute_alpha(om_end, L, 2*L, tau, target)
    plot_beta(alpha, L, tau, 0, om_end, ax, label=f"colloc K={2*L} knots", cheb=True, Q=Q1)
    plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label=f"colloc K={2*L} knots", cheb=True, Q=Q2)
    
    alpha = compute_alpha(om_end, L, 10*L, tau, target)
    plot_beta(alpha, L, tau, 0, om_end, ax, label=f"colloc K={10*L} knots", cheb=True, Q=Q1)
    plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label=f"colloc K={10*L} knots", cheb=True, Q=Q2)
    
    alpha = compute_alpha(om_end, L, 20*L, tau, target)
    plot_beta(alpha, L, tau, 0, om_end, ax, label=f"colloc K={20*L} knots", cheb=True, Q=Q1)
    plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label=f"colloc K={20*L} knots", cheb=True, Q=Q2)
    
    alpha = fourier_indicator(om_min, om_max, L*tau)
    plot_beta(alpha, L, tau, 0, om_end, ax, label="Fourier")
    plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label="Fourier")
    
    plot_nodes(chebyshev_nodes(0, om_end, L), target, ax)
    plot_nodes(chebyshev_nodes(0, om_end, L), target, ax2)
    
    ax.legend()
    ax2.legend()
    plt.show()
        