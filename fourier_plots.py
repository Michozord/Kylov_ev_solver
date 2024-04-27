# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 18:47:24 2024

@author: Michal Trojanowski
"""

from dff import *


tau = 0.0056
om_end = 2/tau


for om_min, om_max in [(3, 6), (12, 14)]:
    chi = indicator(om_min, om_max)
    # title = r"Inverse Fourier transform method, $[\omega_{\min}, \omega_{\max}] = [" + str(om_min) + ", " + str(om_max) + "]$"
    title = ""
    ax, ax2, ax3 = prepare_plots(0, om_end, 0, 40, om_end-0.05, om_end, title=title, fontsize=22)

    for L in (50, 100, 500, 1000):
        T = tau*L
        alpha = fourier_indicator(om_min, om_max, T)
        plot_beta(alpha, L, tau, 0, om_end, ax, label=f"$L = {L}, T = {str(T)[0:5]}$")
        plot_beta(alpha, L, tau, 0, 40, ax2, label=f"$L = {L}, T = {str(T)[0:5]}$")
        plot_beta(alpha, L, tau, om_end-0.05, om_end, ax3, label=f"$L = {L}, T = {str(T)[0:5]}$")
    
    # ax.plot(x:=np.linspace(0, om_end, num=10000), list(map(chi, x)), "--", color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")
    ax2.plot(x:=np.linspace(0, 40, num=10000), list(map(chi, x)), "--", color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")
    # ax3.plot(x:=np.linspace(om_end-2, om_end, num=10000), list(map(chi, x)), "--", color="grey", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")
    
    ax.legend()
    ax2.legend()
    ax3.legend(loc='upper left')
plt.show()

