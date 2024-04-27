# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 17:42:46 2024

@author: Michal Trojanowski
"""

from dff import * 
from equidistant_collocation import compute_alpha


tau = 0.0056
om_end = 2/tau

om_min, om_max = 39, 45
chi = indicator(om_min, om_max)
# title = r"Equidistant collocation $[\omega_{\min}, \omega_{\max}] = "+ f"[{om_min}, {om_max}]$"
title = ""
ax = prepare_plots(0, om_end, title=title, fontsize=22, ylabel=r"$|\tilde{\beta}_{\vec{\alpha}}(\omega)|$")
for L in (10, 25):
    T = tau * L 
    print(f"L = {L}")
    alpha = compute_alpha(om_end, L, L, tau, chi)
    plot_beta(alpha, L, tau, 0, om_end, ax, label=f"$L = {L}, T = {str(T)[0:5]}$")
    c = "blue" if L==10 else "green"

    plot_nodes(np.linspace(0, om_end, num=L), chi, ax, label="", color=c)

ax.legend()


L_max = 100
# title = "cond(Q) equidistant collocation"
title = ""
ax2 = prepare_plots(5, L_max, title=title, xlabel="$L$", ylabel=r"$\mathrm{cond}(Q)$", set_y_lim=False, fontsize=22)
Ls = range(5, L_max+1, 5)
conds = []
for L in Ls:
    Q = q_eval_mat(np.linspace(0, om_end, num=L), L, tau)
    conds.append(np.linalg.cond(Q))
ax2.semilogy(Ls, conds, "r")

plt.show()