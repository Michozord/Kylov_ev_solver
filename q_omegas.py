# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:20:15 2024

@author: Michal Trojanowski
"""
from dff import q_omega
import numpy as np
from matplotlib import pyplot as plt


L = 40
mesh = np.linspace(0, 20)
tau = 0.025

Q = np.zeros((len(mesh), L+1))
for i in range(len(mesh)):
    omega = mesh[i]
    Q[i,:] = q_omega(omega, tau, L)


for l in [1, 5, 14, 39]:
    plt.plot(mesh, Q[:,l], label=f"l={l}")
    plt.plot(mesh, [np.cos(l*tau*om) for om in mesh], label=f"cos({l}*tau*omega)")

plt.legend()
plt.show()