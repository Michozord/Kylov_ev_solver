# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:43:32 2024

@author: Michal Trojanowski
"""

from dff import q_omega
import numpy as np
import cmath

def q_test(L, tau, omega):
    q_omegas = np.zeros(L)
    c = complex(1 - (tau*tau*omega*omega)/2, np.sqrt(1 - (1 - (tau*tau*omega*omega)/2)**2))
    c_quer = complex(1 - (tau*tau*omega*omega)/2, -np.sqrt(1 - (1 - (tau*tau*omega*omega)/2)**2))
    c_l, c_quer_l = 1, 1
    for l in range(L):
        q_omegas[l] = ((c-1)/(2 * 1j * c.imag) * c_l + (1 - c_quer)/(2 * 1j * c.imag) * c_quer_l).real
        c_l *= c
        c_quer_l *= c_quer
    return q_omegas
    
if __name__ == "__main__":
    tau = 1
    L = 10
    for omega in [0.1, 0.5, 1, 1.5, 1.99]:
        rec = q_omega(omega, tau, L)
        direct = q_test(L, tau, omega)
        delta = np.linalg.norm(rec - direct)
        print(f"{omega}: {delta} {'!!!' if delta > 1e-7 else ''}")
        
        
        
    