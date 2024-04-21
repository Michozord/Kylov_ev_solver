# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:15:35 2024

@author: Michal Trojanowski
"""

import numpy as np
from matplotlib import pyplot as plt
from dff import *
from ansatz_collocation import indicator, gauss


omega_start, omega_end = 0, 360
tau = 2/omega_end           # approx. 0.0056
omega_min, omega_max = 2, 4
Ts = range(2, 20)


def cond_eq_mesh(omega_start, omega_end, target, T, tau):
    L = int(T/tau)
    mesh = np.linspace(omega_start, omega_end, num=L)
    A = np.zeros((L, L))
    for k in range(L):
        A[k,:] = tau * q_omega(mesh[k], tau, L)
    return L, np.linalg.cond(A)


def cond_cheb_mesh(omega_start, omega_end, target, T, tau):
    L = int(T/tau)
    K = L
    if K%3 != 0:
        K = K + 3 - K%3
    s = int(K/3)
    Chebyshev = np.array(list(map(lambda j: np.cos((2*j+1)/(2*s)*np.pi), range(s))))
    mesh = np.zeros(K)
    mesh[0:s] = 1/2 * ((omega_min - omega_start) * Chebyshev + (omega_start+omega_min)*np.ones(s))
    mesh[s:2*s] = 1/2 * ((omega_max - omega_min) * Chebyshev + (omega_max+omega_min)*np.ones(s))
    mesh[2*s:] = 1/2 * ((omega_end - omega_max) * Chebyshev + (omega_max+omega_end)*np.ones(s))
    beta_vec = np.array(list(map(target, mesh)))                # vector [target(omega_0), ..., target(omega_K)] - rhs of equation
    A = np.zeros((K, L))
    for k in range(K):
        A[k,:] = tau * q_omega(mesh[k], tau, L)
    return L, np.linalg.cond(A)


# target = indicator
# Ls = []
# conds = []
# for T in Ts:
#     L, cond = cond_eq_mesh(omega_start, omega_end, target, T, tau)
#     Ls.append(L)
#     conds.append(cond)
    
# plt.plot(Ls, conds)
# plt.xlabel("L")
# plt.ylabel("cond(A)")
# plt.title("Condition number of matrix A: equidistant mesh, target $\chi$")

# plt.show()



target = indicator
Ls = []
conds = []
for T in Ts:
    L, cond = cond_cheb_mesh(omega_start, omega_end, target, T, tau)
    Ls.append(L)
    conds.append(cond)
    
plt.plot(Ls, conds)
plt.xlabel("L")
plt.ylabel("cond(A)")
plt.title("Condition number of matrix A: Chebyshev mesh, target $\chi$")

plt.show()


    
    