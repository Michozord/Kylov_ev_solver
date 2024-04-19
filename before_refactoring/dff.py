# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:20:39 2024

@author: Michal Trojanowski
"""

import numpy as np
from matplotlib import pyplot as plt
from typing import Callable

def q_omega(omega: float, tau: float, L: int) -> np.array:
    q_omegas = [1., 1.]
    for l in range(1, L):
        q_omegas.append((2-tau**2 * omega**2)*q_omegas[-1] - q_omegas[-2])
    return np.array(q_omegas)[1:]

def q_omega_cheb(omega: float, tau: float, L: int) -> np.array:
    q_omegas = [1., 1 - tau**2 * omega**2/2]
    for l in range(1, L-1):
        q_omegas.append((2-tau**2 * omega**2)*q_omegas[-1] - q_omegas[-2])
    return np.array(q_omegas)
    

def beta (omega: float, alpha: np.array, tau: float, multiply_with_tau=True) -> float:
    q = q_omega(omega, tau, len(alpha))
    if multiply_with_tau:
        return tau * alpha @ q
    else:
        return alpha @ q
    
def beta_cheb (omega: float, alpha: np.array, tau: float, multiply_with_tau=True) -> float:
    q = q_omega_cheb(omega, tau, len(alpha))
    if multiply_with_tau:
        return tau * alpha @ q
    else:
        return alpha @ q


def plot_beta(start: float, end: float, alpha, tau, L, ax, label=None, absolute=True, multiply_with_tau=True):
    mesh = np.linspace(start, end, num=5000)
    if isinstance(alpha, Callable):
        alpha_vec = np.array(list(map(lambda l: alpha(tau*l), range(0, L))))
    else:
        alpha_vec = np.array(alpha)
    abs_fun = abs if absolute else lambda x: x
    ax.plot(mesh, list(map(lambda om: abs_fun(beta(om, alpha_vec, tau, multiply_with_tau)), mesh)), label=label)


def plot_beta_cheb(start: float, end: float, alpha, tau, L, ax, label=None, absolute=True, multiply_with_tau=True):
    mesh = np.linspace(start, end, num=5000)
    if isinstance(alpha, Callable):
        alpha_vec = np.array(list(map(lambda l: alpha(tau*l), range(0, L))))
    else:
        alpha_vec = np.array(alpha)
    abs_fun = abs if absolute else lambda x: x
    ax.plot(mesh, list(map(lambda om: abs_fun(beta_cheb(om, alpha_vec, tau, multiply_with_tau)), mesh)), label=label)
        

if __name__ == "__main__":
    omega = 1
    L = 10
    tau = 0.025
    print(q_omega(omega, tau, L))