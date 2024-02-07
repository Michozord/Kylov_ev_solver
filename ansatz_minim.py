# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:00:36 2024

@author: Michal Trojanowski
"""

from dff import plot_beta, beta, q_omega
import numpy as np
from matplotlib import pyplot as plt
from ansatz_collocation import gauss, indicator
from newton import newton

tau = 0.025
omega_min = 2
omega_max = 4
omega_start = 0
omega_end = 20


class functional:
    T, S = 10, 40
    L = int(T/tau)
    h = omega_end/S
    target = indicator
    kappa = lambda s: 1
    
    def __init__(self, T, S, target):
        self.T, self.S = T, S
        self.L = int(T/tau)
        self.h = omega_end/S
        self.target = target
        self.kappa = lambda s: 0.5 if s == 0 or s == S else 1
    
    def value(self, alpha):
        vec = np.array(list(map(lambda s: (beta(s*self.h, alpha, tau) - self.target(s*self.h))*np.sqrt(self.kappa(s)), range(self.S+1))))
        return self.h*vec@vec
    
    def gradient(self, alpha):
        res = np.zeros(self.L+1)
        q = np.zeros((self.S+1, self.L+1))        # q[s, k] = q_{s*h}^k
        for s in range(self.S+1):
            q[s, :] = q_omega(s*self.h, tau, self.L)
        for k in range(self.L+1):
            res[k] = sum(list(map(lambda s: (beta(s*self.h, alpha, tau) - self.target(s*self.h))*tau*q[s,k]*self.kappa(s), range(self.S+1))))
        return 2*self.h*res
    
    def hessian(self, alpha):
        H = np.zeros((self.L+1, self.L+1))
        q = np.zeros((self.S+1, self.L+1))        # q[s, k] = q_{s*h}^k
        for s in range(self.S+1):
            q[s, :] = q_omega(s*self.h, tau, self.L)
        for k in range(self.L+1):
            for n in range(k+1):
                if k == n:
                    val = 2*self.h*sum(list(map(lambda s: (tau*tau * q[s,n]*q[s,k] + beta(s*self.h, alpha, tau) - self.target(s*self.h))*self.kappa(s), range(self.S+1))))
                else:
                    val = 2*self.h*sum(list(map(lambda s: tau*tau * q[s,n]*q[s,k]*self.kappa(s), range(self.S+1))))
                    H[n,k] = val
                H[k,n] = val
        return H
        

def compute_alpha(omega_min, omega_max, omega_end, tau, target, T, S):
    F = functional(T, S, target)
    L = int(T/tau)
    alpha_start = np.zeros(L+1)
    alpha = newton(F.gradient, F.hessian, alpha_start)
    # alpha = newton(F.gradient, alpha_start, fprime=F.hessian)
    print("Newton found local extremum: F(alpha) =", F.value(alpha))
    return alpha


fig, ax = plt.subplots()
for T in [1, 2.5, 5, 10]:
    for S in [20, 40, 100, 400]:
        print(f"T = {T}, S = {S}")
        try:
            alpha = compute_alpha(omega_min, omega_max, omega_end, tau, indicator, T, S)
            L = int(T/tau)
            plot_beta(0, omega_end, alpha, tau, L, ax, label=f"S = {S}")
        except RuntimeError:
            print("Newton did not converge!")
            pass
    ax.grid()
    plt.xlim(omega_start, omega_end)
    plt.xticks([2*n for n in range(11)])
    plt.legend()
    plt.title(f"Minimalisation approach, start value: 0-vector, T = {T}")
plt.show()
    
    