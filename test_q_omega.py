# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:00:58 2024

@author: Michal Trojanowski
"""

from dff import * 
import cmath

tau = 0.01
omegas = np.linspace(0, 49)
Q1 = q_eval_mat(omegas, 10, tau, cheb=True)
Q2 = np.zeros((len(omegas), 10))
for k in range(len(omegas)):
    omega = omegas[k]
    c = complex(a := 1 - tau**2 * omega**2/2, np.sqrt(1 - a*a))
    c_quer = complex(a, -np.sqrt(1 - a*a))
    for l in range(10):
        val = 0.5 * pow(c, l) + 0.5 * pow(c_quer, l)
        if abs(val.imag) > 1e-7:
            print(val.imag)
        Q2[k,l] = val.real 

print("imag ok")
print(np.linalg.norm(Q1-Q2))