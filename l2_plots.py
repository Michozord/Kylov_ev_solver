# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:30:31 2024

@author: Michal Trojanowski
"""

from dff import * 
from l2_minimization import compute_alpha
from chebyshev_collocation import chebyshev_nodes

tau = 0.0056
om_min, om_max = 12, 14 
chi = indicator(om_min, om_max)
om_end = 2/tau


# title = r"$L^2$ minimiation, $[\omega_{\min}, \omega_{\max}] = "+ f"[{om_min}, {om_max}]$, $h = {str(1/20)[0:8]}$"
title = ""
ax1, ax2, ax3 = prepare_plots(0, om_end, 0, 40, om_end-0.5, om_end, title=title, fontsize=22, ylabel=r"$|\tilde{\beta}_{\vec{\alpha}}(\omega)|$")
for L in (50, 100, 200):
    T = L *tau
    alpha = compute_alpha(om_end, L, tau, om_min, om_max)
    label = f"$L = {L}$, $T = {str(T)[0:5]}$"
    plot_beta(alpha, L, tau, 0, om_end, ax1, label=label)
    plot_beta(alpha, L, tau, 0, 40, ax2, label=label)
    plot_beta(alpha, L, tau, om_end-0.5, om_end, ax3, label=label)

ax1.plot(x:=np.linspace(0, om_end, num=10000), list(map(chi, x)), "--", color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")
ax2.plot(x:=np.linspace(0, 40, num=1000), list(map(chi, x)), "--", color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")
ax3.plot(x:=np.linspace(om_end, om_end-0.5, num=5), list(map(chi, x)), "--", color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")


L = 50
T = L*tau

alpha_four = fourier_indicator(om_min, om_max, L*tau)
plot_beta(alpha_four, L, tau, 0, 40, ax2, label=f"IFT, $T={str(T)[0:5]}$", color="y")
plot_beta(alpha_four, L, tau, om_end-0.5, om_end, ax3, label=f"IFT, $T={str(T)[0:5]}$", color="y")



tau = 0.0056
om_min, om_max = 122, 130 
chi = indicator(om_min, om_max)
om_end = 2/tau


# title = r"$L^2$ minimiation, $[\omega_{\min}, \omega_{\max}] = "+ f"[{om_min}, {om_max}]$, $h = {str(1/20)[0:8]}$"
title = ""
ax4, ax5, ax6 = prepare_plots(0, om_end, 110, 150, om_end-0.5, om_end, title=title, fontsize=22, ylabel=r"$|\tilde{\beta}_{\vec{\alpha}}(\omega)|$")
for L in (50, 100, 200):
    T = L *tau
    alpha = compute_alpha(om_end, L, tau, om_min, om_max)
    label = f"$L = {L}$, $T = {str(T)[0:5]}$"
    plot_beta(alpha, L, tau, 0, om_end, ax4, label=label)
    plot_beta(alpha, L, tau, 110, 150, ax5, label=label)
    plot_beta(alpha, L, tau, om_end-0.5, om_end, ax6, label=label)

ax4.plot(x:=np.linspace(0, om_end, num=10000), list(map(chi, x)), "--", color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")
ax5.plot(x:=np.linspace(110, 150, num=1000), list(map(chi, x)), "--", color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")
ax6.plot(x:=np.linspace(om_end, om_end-0.5, num=5), list(map(chi, x)), "--", color="black", label=r"$\chi_{[\omega_{\min}, \omega_{\max}]}$")


L = 50
T = L*tau

alpha_four = fourier_indicator(om_min, om_max, L*tau)
plot_beta(alpha_four, L, tau, 110, 150, ax5, label=f"IFT, $T={str(T)[0:5]}$", color="y")
plot_beta(alpha_four, L, tau, om_end-0.5, om_end, ax6, label=f"IFT, $T={str(T)[0:5]}$", color="y")


ax1.legend()
ax2.legend()
ax3.legend(loc="upper left")
ax4.legend()
ax5.legend()
ax6.legend(loc="upper left")


L_max = 500
# title = r"$\mathrm{cond}(Q^TQ)$ "
title = ""
ax7 = prepare_plots(5, L_max, title=title, xlabel="$L$", ylabel=r"$\mathrm{cond}(Q^TQ)$", set_y_lim=False, fontsize=22)
Ls = range(10, L_max+1, 10)
K = 5
K2 = 20
conds = []
conds_2 = []
conds_cheb = []
for L in Ls:
    mesh = np.linspace(0, om_end - 1/K, num = int(K*om_end)) + 1/(2*K)
    Q = q_eval_mat(mesh, L, tau)
    
    conds.append(np.linalg.cond(Q.transpose() @ Q))
    mesh = np.linspace(0, om_end - 1/K2, num = int(K2*om_end)) + 1/(2*K2)
    Q = q_eval_mat(mesh, L, tau)
    conds_2.append(np.linalg.cond(Q.transpose() @ Q))
    
    mesh = chebyshev_nodes(0, om_end, int(K*om_end))
    Q = q_eval_mat(mesh, L, tau)
    conds_cheb.append(np.linalg.cond(Q.transpose() @ Q))
    
ax7.semilogy(Ls, conds, "r", label="Equidistant nodes, $h=0.2$")
ax7.semilogy(Ls, conds_2, "m", label="Equidistant nodes, $h=0.05$")
ax7.semilogy(Ls, conds_cheb, "c", label="Chebyshev nodes")
ax7.legend(loc="lower right")


plt.show()