# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:21:16 2024

@author: Michal Trojanowski
"""

from dff import * 

def compute_alpha(om_end: float, L: int, K: int, tau: float, target: Callable) -> np.array:
    mesh = np.linspace(0, om_end, num=K)
    Q = q_eval_mat(mesh, L, tau, cheb=False)
    rhs = 1/tau * np.array(list(map(target, mesh)))
    if K==L:
        alpha = np.linalg.solve(Q, rhs)
        return alpha
    else:
        alpha = np.linalg.lstsq(Q, rhs)[0]
        return alpha


if __name__ == "__main__":
    om_min, om_max, om_end = 12, 14, 360
    target = indicator(om_min, om_max)
    tau = 2/om_end
    L = int(10/tau)
    alpha = compute_alpha(om_end, L, 10*L, tau, target)
    ax = prepare_plot(0, om_end)
    Q = plot_beta(alpha, L, tau, 0, om_end, ax, label="approximation")
    alpha = fourier_indicator(om_min, om_max, 10)
    plot_beta(alpha, L, tau, 0, om_end, ax, label="fourier", Q=Q)
    plt.legend()
    plt.show()
    