# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 21:10:26 2024

@author: Michal Trojanowski
"""

import numpy as np
from matplotlib import pyplot as plt
from dff import *
from alpha_parser import write_to_file

def l2_matrix(L: int, omega_end: float, M: int, tau: float):
    A = np.zeros((L, L))        # A[i,j] = (q_i, q_j)_L2
    Q = np.zeros((L, M))
    # [ q_0(s_0) , ..., q_0(s_M-1)  ] 
    # [   :                  :      ]
    # [   :                  :      ]
    # [q_L-1(s_0), ..., q_L-1(s_M-1)]
    s_mesh = np.linspace(0, omega_end, num=M)
    delta_s = s_mesh[1] - s_mesh[0]
    for j in range(M):
        Q[:,j] = q_omega(s_mesh[j], tau, L)
    Q[:,1] *= 1/np.sqrt(2)
    Q[:,-1] *= 1/np.sqrt(2)
    for i in range(L):
        for j in range(i, L):
            A[i,j] = Q[i,:] @ Q[j,:] 
    A *= delta_s
    return A

def l2_vec_indicator(L: int, omega_min: float, omega_max: float, M: int, tau: float):
    v = np.zeros(L)
    s_mesh = np.linspace(omega_min, omega_max, num=M)
    delta_s = s_mesh[1] - s_mesh[0]
    c = lambda i: 1/2 if i in [0, M-1] else 1
    for i in range(M):
        q = q_omega(s_mesh[i], tau, L)      # [q_0(s_i), ..., q_L-1(s_i)]
        v += q * c(i)
    v *= delta_s
    print("M=", M, "\nv=", v)
    return v

def alpha_l2(L: int, omega_min: float, omega_max: float, omega_end: float, M: int, tau: float):
    A = l2_matrix(L, omega_end, M, tau)
    M = int(M * (omega_max - omega_min)/omega_end)
    v = l2_vec_indicator(L, omega_min, omega_max, M, tau)
    alpha = np.linalg.solve(A, v)
    return alpha


if __name__ == "__main__":
    omega_start, omega_end = 0, 360
    tau = 1/omega_end           
    omega_min, omega_max = 2, 4
    Ts = [1, 2.5, 5, 10]
    target_name = r"$\chi$"
    
    for T in Ts:
        L = int(T/tau)
        for M in [2*omega_end, 10*omega_end, 20*omega_end]:
            alpha = alpha_l2(L, omega_min, omega_max, omega_end, M, tau)/tau
            write_to_file(f"L2 minimalisation, target function {target_name}, T = {T}, L = {L}", f"M = {M} quadrature points", alpha, tau, L)

