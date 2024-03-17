# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 07:48:54 2024

@author: Michal Trojanowski
"""

import numpy as np
from math import factorial
from matplotlib import pyplot as plt
from dff import *


def gamma(k, i, x):
    m = (k-i)%4
    if m == 0:
        return np.sin(x)
    elif m == 1:
        return np.cos(x)
    elif m == 2:
        return -np.sin(x)
    elif m == 3:
        return -np.cos(x)


def improper_integral(k, omega):
    #improper integral t^k * cos(omega*t) dt 
    if omega == 0:
        def ii(T):
            return T**(k+1)/(k+1) 
    else:
        def ii(T):
            return 1/omega*sum([T**i/omega**(k-i) * factorial(k)/factorial(i) * gamma(k, i, omega*T) for i in range(k+1)])
    return ii


def X_matrix (K, omega_vec, T):
    """

    Parameters
    ----------
    K : TYPE
        max. degree of polynomial alpha
    omega_vec : TYPE
        vector of omegas 
    T : TYPE
        time interval

    Returns
    -------
    X : TYPE
        

    """
    X = np.zeros((len(omega_vec), K+1))
    for j in range(len(omega_vec)):
        omega = omega_vec[j]
        for k in range(K+1):
            integral = improper_integral(k, omega)
            X[j,k] = integral(T) - integral(0)
    return X



def alpha_constructor(K, T, omega_vec, b):
    X = X_matrix(K, omega_vec, T)
    a = np.linalg.solve(X.transpose()@X, X.transpose()@b)
    print("K=", K, "a=", a)
    def alpha(omega):
        return a @ np.array([omega**i for i in range(K+1)])
    return alpha


omega_min = 2
omega_max = 4
start = -1
end = 19
T=15
tau = 0.0025
fig, ax = plt.subplots()
Ks = [5, 10, 20]
# omega_vec = [1.5**n for n in range(-10, 12)]
# omega_vec = [20/n for n in range(1, 50)] + [4 + 2/n for n in range(1, 10)] + [4 - 2/n for n in range(1, 10)]
omega_vec = [0.2*n for n in range(-5, 55)]
# omega_vec = [4 - 2/n for n in range(1, 20)] + [2 - 0.2*n for n in range(20)] + [4 + 0.2*n for n in range(20)]
b = np.array([0 if om<2 or om>4 else 1 for om in omega_vec])
for K in Ks:
    alpha = alpha_constructor(K, T, omega_vec, b)
    plot_beta(start, end, alpha, tau, int(T/tau), ax, "K="+str(K), absolute=True)  
    plt.plot(omega_vec, b, "x")
ax.grid()
plt.xlim(start, end)
plt.xticks([2*n for n in range(11)])
plt.legend()
plt.show()
        
    