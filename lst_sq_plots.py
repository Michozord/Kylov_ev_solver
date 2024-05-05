# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:07:17 2024

@author: Michal Trojanowski
"""

from chebyshev_collocation import chebyshev_nodes, compute_alpha
from dff import *

tau = 0.0056
om_end = 2/tau
om_min, om_max = 12, 14
chi = indicator(om_min, om_max)
# chi = gauss(13)
L=100
T = L*tau

# title = r"Least squares method with Chebyshev nodes and $[\omega_{\min}, \omega_{\max}] = "+ f"[{om_min}, {om_max}]$, $L = {L}$, $T = {str(T)[0:5]}$"
title = ""
ax1, ax2 = prepare_plots(0, om_end, 0, 40, title=title, fontsize=22, ylabel=r"$|\tilde{\beta}_{\vec{\alpha}}(\omega)|$")
for K in (2*L, 10*L, 50*L):
    alpha = compute_alpha(om_end, L, K, tau, chi)
    label = f"$K = {K}$"
    plot_beta(alpha, L, tau, 0, om_end, ax1, label=label)
    plot_beta(alpha, L, tau, 0, 40, ax2, label=label)

# plot_nodes(chebyshev_nodes(0, om_end, 200), chi, ax1, label="", color="b", crosses="X")
plot_nodes(chebyshev_nodes(0, om_end, 200), chi, ax2, label="", color="b", crosses="X")

ax1.plot(x:=np.linspace(0, om_end, num=10000), list(map(chi, x)), "--", color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")
ax2.plot(x:=np.linspace(0, 40, num=1000), list(map(chi, x)), "--", color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")

L=100
alpha_four = fourier_indicator(om_min, om_max, L*tau)
# alpha_four = fourier_gauss(13, L*tau)
# plot_beta(alpha_four, L, tau, 0, om_end, ax1, label=f"Inverse Fourier method, $T={T}$", color="magenta")
plot_beta(alpha_four, L, tau, 0, 40, ax2, label=f"IFT", color="y")

ax1.legend()
ax2.legend()
plt.show()