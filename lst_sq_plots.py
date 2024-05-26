# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:07:17 2024

@author: Michal Trojanowski
"""

from chebyshev_collocation import chebyshev_nodes, compute_alpha
from dff import *

pltstl = PlotStyle()

tau = 0.0056
om_end = 2/tau
om_min, om_max = 12, 14
chi = indicator(om_min, om_max)
# chi = gauss(13)
L=100
T = L*tau

# title = r"Least squares method with Chebyshev nodes and $[\omega_{\min}, \omega_{\max}] = "+ f"[{om_min}, {om_max}]$, $L = {L}$, $T = {str(T)[0:5]}$"
title = ""
ax1, ax2 = prepare_plots(0, om_end, 0, 25, title=title, fontsize=22, ylabel=r"$|\tilde{\beta}_{\vec{\alpha}}(\omega)|$")
for K in (2*L, 10*L, 50*L):
    alpha = compute_alpha(om_end, L, K, tau, chi)
    label = f"$K = {K}$"
    stl = pltstl.get_style()
    plot_beta(alpha, L, tau, 0, om_end, ax1, style=stl, label=label)
    plot_beta(alpha, L, tau, 0, 25, ax2, style=stl, label=label)

# plot_nodes(chebyshev_nodes(0, om_end, 200), chi, ax1, label="", color="b", crosses="X")
plot_nodes(chebyshev_nodes(0, om_end, 200), chi, ax2, label="", color="b", crosses="X")

ax1.plot(x:=np.linspace(0, om_end, num=10000), list(map(chi, x)), linestyle=pltstl.target_style, color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")
ax2.plot(x:=np.linspace(0, 25, num=1000), list(map(chi, x)), linestyle=pltstl.target_style, color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")

L=100
alpha_four = fourier_indicator(om_min, om_max, L*tau)
# alpha_four = fourier_gauss(13, L*tau)
# plot_beta(alpha_four, L, tau, 0, om_end, ax1, label=f"Inverse Fourier method, $T={T}$", color="magenta")
plot_beta(alpha_four, L, tau, 0, 25, ax2, label=f"IFT", style=pltstl.get_style(), color="y")

handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
ax1.legend(handles=handles_ax2)
plt.show()