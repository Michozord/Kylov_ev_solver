# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:52:48 2024

@author: Michal Trojanowski
"""

from dff import * 
from chebyshev_collocation import compute_alpha as alpha_cheb
from equidistant_collocation import compute_alpha as alpha_eq
from l2_minimization import compute_alpha as alpha_l2

om_min, om_max = 12, 14
tau = 0.0056
om_end= 2/tau
L = 200
target = indicator(om_min, om_max)
title = r"Compare methods, T = " + str(L * tau)[0:7] + r", $\omega_{end}$ = " + str(om_end)[0:7] + ", L = " + str(L) + f", target intervall ({om_min}, {om_max})"
ax1 = prepare_plot(0, om_end, title=title)
ax2 = prepare_plot(om_end-0.2, om_end, title=title)

alpha = alpha_l2(om_end, L, tau, om_min, om_max, K=5)
Q1 = plot_beta(alpha, L, tau, 0, om_end, ax1, label=f"L2 minimize", cheb=False)
Q2 = plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label=f"L2 minimize", cheb=False)

alpha = alpha_cheb(om_end, L, 2*L, tau, target)
plot_beta(alpha, L, tau, 0, om_end, ax1, label=f"Chebyshev coll., K={2*L}", cheb=True)
plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label=f"Chebyshev coll., K={2*L}", cheb=True)

alpha = alpha_cheb(om_end, L, 5*L, tau, target)
plot_beta(alpha, L, tau, 0, om_end, ax1, label=f"Chebyshev coll., K={5*L}", cheb=True)
plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label=f"Chebyshev coll., K={5*L}", cheb=True)

alpha = alpha_eq(om_end, L, 2*L, tau, target)
plot_beta(alpha, L, tau, 0, om_end, ax1, label=f"Equidistant coll., K={2*L}", cheb=False)
plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label=f"Equidistant coll., K={2*L}", cheb=False)

alpha = alpha_eq(om_end, L, 5*L, tau, target)
plot_beta(alpha, L, tau, 0, om_end, ax1, label=f"Equidistant coll., K={5*L}", cheb=False)
plot_beta(alpha, L, tau, om_end-0.2, om_end, ax2, label=f"Equidistant coll., K={5*L}", cheb=False)


ax1.legend()
ax2.legend()
plt.show()