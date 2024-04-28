# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:21:16 2024

@author: Michal Trojanowski
"""

from dff import * 

def compute_alpha(om_end: float, L: int, K: int, tau: float, target: Callable) -> np.array:
    mesh = np.linspace(0, om_end, num=K)
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
    om_min, om_max = 12, 14
    target = indicator(om_min, om_max)
    tau = 0.0056
    om_end= 2/tau
    L = 200
    title = r"Equidistant collocation, T = " + str(L * tau)[0:7] + r", $\omega_{end}$ = " + str(om_end)[0:7] + ", L = " + str(L) + f", target intervall ({om_min}, {om_max})"
    ax1 = prepare_plots(0, om_end, title=title)
    ax2 = prepare_plots(om_end-0.5, om_end, title=title)
    
    alpha = compute_alpha(om_end, L, L, tau, target)
    Q1 = plot_beta(alpha, L, tau, 0, om_end, ax1, label=f"colloc K={L} knots")
    Q2 = plot_beta(alpha, L, tau, om_end-0.5, om_end, ax2, label=f"colloc K={L} knots")
    
    alpha = compute_alpha(om_end, L, 2*L, tau, target)
    plot_beta(alpha, L, tau, 0, om_end, ax1, label=f"colloc K={2*L} knots", Q=Q1)
    plot_beta(alpha, L, tau, om_end-0.5, om_end, ax2, label=f"colloc K={2*L} knots", Q=Q2)
    
    alpha = compute_alpha(om_end, L, 5*L, tau, target)
    plot_beta(alpha, L, tau, 0, om_end, ax1, label=f"colloc K={5*L} knots", Q=Q1)
    plot_beta(alpha, L, tau, om_end-0.5, om_end, ax2, label=f"colloc K={5*L} knots", Q=Q2)
    
    alpha = compute_alpha(om_end, L, 10*L, tau, target)
    plot_beta(alpha, L, tau, 0, om_end, ax1, label=f"colloc K={10*L} knots", Q=Q1)
    plot_beta(alpha, L, tau, om_end-0.5, om_end, ax2, label=f"colloc K={10*L} knots", Q=Q2)
    
    alpha = fourier_indicator(om_min, om_max, 10)
    plot_beta(alpha, L, tau, 0, om_end, ax1, label="Fourier", Q=Q1)
    plot_beta(alpha, L, tau, om_end-0.5, om_end, ax2, label="Fourier", Q=Q2)
    
    # plot_nodes(np.linspace(0, om_end, num=L), target, ax1)
    # plot_nodes(np.linspace(0, om_end, num=L), target, ax2)
    
    ax1.legend()
    ax2.legend()
    plt.show()
    