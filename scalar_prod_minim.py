# -*- coding: utf-8 -*-
"""
Created on Fri May 10 08:20:23 2024

@author: Michal Trojanowski
"""

from dff import * 
from typing import Callable
from math import floor, ceil
from l2_minimization import compute_alpha as alpha_l2


def compute_alpha(om_end: float, L: int, tau: float, om_min: float, om_max: float, sigma: Callable, K: Optional[int] = 20):
    quad_mesh = np.linspace(0, om_end - 1/K, num = int(K*om_end)) + 1/(2*K)
    Q = q_eval_mat(quad_mesh, L, tau)
    rhs = np.zeros(L)
    for l in range(ceil(K*om_min), floor(K*om_max)+1):
        rhs += Q[l,:] * sigma(quad_mesh[l])
    rhs /= K
    X = np.zeros((L, L))
    sigma_vec = np.array(list(map(sigma, quad_mesh)))
    Qsigma = (Q.transpose() * sigma_vec)
    X = 1/K * Qsigma @ Q
    print(f"det(X) = {np.linalg.det(X)}, cond(X) = {np.linalg.cond(X)}")
    alpha = np.linalg.solve(X, rhs)/tau
    return alpha


if __name__ == "__main__":
    om_min, om_max = 12, 14
    tau = 0.0056
    om_end= 2/tau
    L = 30
    
    ax = prepare_plots(0, om_end)
    for s in (1, 10, 50, 100):
        sigma = lambda x: s if x >= 10 and x <= 16 else 1 
        alpha = compute_alpha(om_end, L, tau, om_min, om_max, sigma)
        breakpoint()
        label = r"$\sigma(\omega) = 1 + " + str(s-1) + r"\chi_{(10, 16)}$"
        plot_beta(alpha, L, tau, 0, om_end, ax, label=label)
    
    ax.plot(x := [0, om_min-0.01, om_min+0.01, om_max-0.01, om_max+0.01, om_end], list(map(indicator(om_min, om_max), x)), ":", color="k", label="target")
    # axs.plot(x:=np.linspace(0, om_end, num=1000), list(map(lambda x: 5 if x > 10 and x < 16 else 1, x)), "--", color="k")
    # axs.set_ylim(0, 6)
    ax.legend()
    plt.show()
    