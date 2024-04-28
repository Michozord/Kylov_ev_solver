# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 22:36:53 2024

@author: Michal Trojanowski
"""

from chebyshev_collocation import chebyshev_nodes, compute_alpha
from dff import *

tau = 0.0056
om_end = 2/tau
om_min, om_max = 12, 14
chi = indicator(om_min, om_max)


# title = r"Chebyshev collocation $[\omega_{\min}, \omega_{\max}] = "+ f"[{om_min}, {om_max}]$"
title = ""
ax1, ax2 = prepare_plots(0, om_end, 0, 40, title=title, fontsize=22, ylabel=r"$|\tilde{\beta}_{\vec{\alpha}}(\omega)|$")
for L in (100, 250, 500):
    T = L*tau
    alpha = compute_alpha(om_end, L, L, tau, chi)
    label = f"$L = {L}, T = {str(T)[0:5]}$"
    plot_beta(alpha, L, tau, 0, om_end, ax1, label=label)
    plot_beta(alpha, L, tau, 0, 40, ax2, label=label)
    
# plot_nodes(chebyshev_nodes(0, om_end, 250), chi, ax1, label="", color="g", crosses="X")
plot_nodes(chebyshev_nodes(0, om_end, 250), chi, ax2, label="", color="g", crosses="X") 
# plot_nodes(chebyshev_nodes(0, om_end, 500), chi, ax1, label="", color="r", crosses="X")
plot_nodes(chebyshev_nodes(0, om_end, 500), chi, ax2, label="", color="r", crosses="X") 
plot_nodes(chebyshev_nodes(0, om_end, 100), chi, ax2, label="", color="b", crosses="X")  
ax1.plot(x:=np.linspace(0, om_end, num=10000), list(map(chi, x)), "--", color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")
ax2.plot(x:=np.linspace(0, 40, num=1000), list(map(chi, x)), "--", color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")

L=250
alpha_four = fourier_indicator(om_min, om_max, L*tau)
# plot_beta(alpha_four, L, tau, 0, om_end, ax1, label=f"Inverse Fourier method, $T={T}$", color="magenta")
plot_beta(alpha_four, L, tau, 0, 40, ax2, label=f"IFT, $T={T}$", color="y")



tau = 0.0056
om_end = 2/tau
om_min, om_max = 122, 130
chi = indicator(om_min, om_max)


# title = r"Chebyshev collocation $[\omega_{\min}, \omega_{\max}] = "+ f"[{om_min}, {om_max}]$"
title = ""
ax3, ax4 = prepare_plots(0, om_end, 110, 150, title=title, fontsize=22, ylabel=r"$|\tilde{\beta}_{\vec{\alpha}}(\omega)|$")
for L in (100, 250, 500):
    T = L*tau
    alpha = compute_alpha(om_end, L, L, tau, chi)
    label = f"$L = {L}, T = {str(T)[0:5]}$"
    plot_beta(alpha, L, tau, 0, om_end, ax3, label=label)
    plot_beta(alpha, L, tau, 110, 150, ax4, label=label)
    
# plot_nodes(chebyshev_nodes(0, om_end, 250), chi, ax3, label="", color="g", crosses="X")
plot_nodes(chebyshev_nodes(0, om_end, 250), chi, ax4, label="", color="g", crosses="X") 
# plot_nodes(chebyshev_nodes(0, om_end, 500), chi, ax3, label="", color="r", crosses="X")
plot_nodes(chebyshev_nodes(0, om_end, 500), chi, ax4, label="", color="r", crosses="X") 
plot_nodes(chebyshev_nodes(0, om_end, 100), chi, ax4, label="", color="b", crosses="X")  
ax3.plot(x:=np.linspace(0, om_end, num=10000), list(map(chi, x)), "--", color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")
ax4.plot(x:=np.linspace(110, 150, num=1000), list(map(chi, x)), "--", color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")

L=250
alpha_four = fourier_indicator(om_min, om_max, L*tau)
# plot_beta(alpha_four, L, tau, 0, om_end, ax3, label=f"Inverse Fourier method, $T={T}$", color="magenta")
plot_beta(alpha_four, L, tau, 110, 150, ax4, label=f"IFT, $T={T}$", color="y")


ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
plt.show()
    