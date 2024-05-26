# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 22:36:53 2024

@author: Michal Trojanowski
"""

from chebyshev_collocation import chebyshev_nodes, compute_alpha
from dff import *

pltstl = PlotStyle()

tau = 0.0056
om_end = 2/tau
om_min, om_max = 12, 14
chi = indicator(om_min, om_max)


# title = r"Chebyshev collocation $[\omega_{\min}, \omega_{\max}] = "+ f"[{om_min}, {om_max}]$"
title = ""
ax1, ax2 = prepare_plots(0, om_end, 0, 25, title=title, fontsize=22, ylabel=r"$|\tilde{\beta}_{\vec{\alpha}}(\omega)|$")
for L in (100, 250, 500):
    T = L*tau
    alpha = compute_alpha(om_end, L, L, tau, chi)
    label = f"$L = {L}, T = {str(T)[0:5]}$"
    stl = pltstl.get_style()
    plot_beta(alpha, L, tau, 0, om_end, ax1, style=stl, label=label)
    plot_beta(alpha, L, tau, 0, 25, ax2, style=stl, label=label)
    
# plot_nodes(chebyshev_nodes(0, om_end, 250), chi, ax1, label="", color="g", crosses="X")
plot_nodes(chebyshev_nodes(0, om_end, 250), chi, ax2, label="", color="g", crosses="X") 
# plot_nodes(chebyshev_nodes(0, om_end, 500), chi, ax1, label="", color="r", crosses="X")
plot_nodes(chebyshev_nodes(0, om_end, 500), chi, ax2, label="", color="r", crosses="X") 
plot_nodes(chebyshev_nodes(0, om_end, 100), chi, ax2, label="", color="b", crosses="X")  
ax1.plot(x:=np.linspace(0, om_end, num=10000), list(map(chi, x)), linestyle=pltstl.target_style, color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")
ax2.plot(x:=np.linspace(0, 25, num=1000), list(map(chi, x)), linestyle=pltstl.target_style, color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")

L=250
alpha_four = fourier_indicator(om_min, om_max, L*tau)
# plot_beta(alpha_four, L, tau, 0, om_end, ax1, label=f"Inverse Fourier method, $T={T}$", color="magenta")
plot_beta(alpha_four, L, tau, 0, 25, ax2, style=pltstl.get_style(), label=f"IFT, $T={T}$", color="y")



tau = 0.0056
om_end = 2/tau
om_min, om_max = 122, 130
chi = indicator(om_min, om_max)

pltstl.reset()

# title = r"Chebyshev collocation $[\omega_{\min}, \omega_{\max}] = "+ f"[{om_min}, {om_max}]$"
title = ""
ax3, ax4 = prepare_plots(0, om_end, 110, 150, title=title, fontsize=22, ylabel=r"$|\tilde{\beta}_{\vec{\alpha}}(\omega)|$", ymax=1.5)
for L in (100, 250, 500):
    T = L*tau
    alpha = compute_alpha(om_end, L, L, tau, chi)
    label = f"$L = {L}, T = {str(T)[0:5]}$"
    stl = pltstl.get_style()
    plot_beta(alpha, L, tau, 0, om_end, ax3, style=stl, label=label)
    plot_beta(alpha, L, tau, 110, 150, ax4, style=stl, label=label)
    
# plot_nodes(chebyshev_nodes(0, om_end, 250), chi, ax3, label="", color="g", crosses="X")
plot_nodes(chebyshev_nodes(0, om_end, 250), chi, ax4, label="", color="g", crosses="X") 
# plot_nodes(chebyshev_nodes(0, om_end, 500), chi, ax3, label="", color="r", crosses="X")
plot_nodes(chebyshev_nodes(0, om_end, 500), chi, ax4, label="", color="r", crosses="X") 
plot_nodes(chebyshev_nodes(0, om_end, 100), chi, ax4, label="", color="b", crosses="X")  
ax3.plot(x:=np.linspace(0, om_end, num=10000), list(map(chi, x)), linestyle=pltstl.target_style, color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")
ax4.plot(x:=np.linspace(110, 150, num=1000), list(map(chi, x)), linestyle=pltstl.target_style, color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")

L=250
alpha_four = fourier_indicator(om_min, om_max, L*tau)
# plot_beta(alpha_four, L, tau, 0, om_end, ax3, label=f"Inverse Fourier method, $T={T}$", color="magenta")
plot_beta(alpha_four, L, tau, 110, 150, ax4, style=pltstl.get_style(), label=f"IFT, $T={T}$", color="y")



tau = 0.0056
om_end = 2/tau
om_mid = 4
g = gauss(om_mid)
pltstl.reset()

# title = r"Chebyshev collocation $e^{-(x- " + str(om_mid) +r")^2}$"
title = ""
ax5, ax6 = prepare_plots(0, om_end, 0, 20, title=title, fontsize=22, ylabel=r"$|\tilde{\beta}_{\vec{\alpha}}(\omega)|$")
for L in (100, 250, 500):
    T = L*tau
    alpha = compute_alpha(om_end, L, L, tau, g)
    label = f"$L = {L}, T = {str(T)[0:5]}$"
    stl = pltstl.get_style()
    plot_beta(alpha, L, tau, 0, om_end, ax5, style=stl, label=label)
    plot_beta(alpha, L, tau, 0, 20, ax6, style=stl, label=label)
    
# plot_nodes(chebyshev_nodes(0, om_end, 250), chi, ax3, label="", color="g", crosses="X")
plot_nodes(chebyshev_nodes(0, om_end, 250), g, ax6, label="", color="g", crosses="X") 
# plot_nodes(chebyshev_nodes(0, om_end, 500), chi, ax3, label="", color="r", crosses="X")
plot_nodes(chebyshev_nodes(0, om_end, 500), g, ax6, label="", color="r", crosses="X") 
plot_nodes(chebyshev_nodes(0, om_end, 100), g, ax6, label="", color="b", crosses="X")  
ax5.plot(x:=np.linspace(0, om_end, num=10000), list(map(g, x)), linestyle=pltstl.target_style, color="black", label=r"$e^{-(\omega- " + str(om_mid) +r")^2}$")
ax6.plot(x:=np.linspace(0, 20, num=1000), list(map(g, x)), linestyle=pltstl.target_style, color="black", label=r"$e^{-(\omega- " + str(om_mid) +r")^2}$")

L=500
alpha_four = fourier_gauss(om_mid, L*tau)
# plot_beta(alpha_four, L, tau, 0, om_end, ax5, label=f"IFT, $T={T}$", color="y")
plot_beta(alpha_four, L, tau, 0, 20, ax6, style=pltstl.get_style(), label=f"IFT, $T={T}$", color="y")


handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
ax1.legend(handles=handles_ax2)
handles_ax4, labels_ax4 = ax2.get_legend_handles_labels()
ax3.legend(handles=handles_ax4)
handles_ax6, labels_ax6 = ax6.get_legend_handles_labels()
ax5.legend(handles=handles_ax2)

plt.show()
