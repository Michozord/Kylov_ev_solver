# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:42:34 2024

@author: Michal Trojanowski
"""

from dff import *
import numpy as np
from alpha_parser import *
from matplotlib import pyplot as plt

omega_min, omega_max = 2, 4

def indicator(x):
    return 1 if x >= omega_min and x <= omega_max else 0


def A_equidistant(omega_start, omega_end, target, L, tau):
    T = L*tau
    mesh = np.linspace(omega_start, omega_end, num=L)
    beta_vec = np.array(list(map(target, mesh)))                # vector [target(omega_0), ..., target(omega_K-1)] - rhs of equation
    A = np.zeros((L, L))
    for k in range(L):
        A[k,:] = tau * q_omega(mesh[k], tau, L)
    # print(f"   tau = {tau} done")
    return(np.linalg.cond(A))


if __name__ == "__main__":
    Ls = range(5, 50)
    omega_end = 10
    for tau in [2/omega_max, 1/omega_max]:
       plt.semilogy(Ls, list(map(lambda L: A_equidistant(0, omega_end, indicator, L, tau), Ls)), label=f"tau = {tau}")
    plt.semilogy(Ls, list(map(lambda L: np.exp(2*L), Ls)), ":", color = "black", label="exp(2*L)")
    plt.legend()
    plt.xlabel("L")
    plt.ylabel("cond(A)")
    plt.show()
        
    
              
    
    