# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:20:39 2024

@author: Michal Trojanowski
"""

import numpy as np
from matplotlib import pyplot as plt
from typing import Callable

def q_omega(omega: float, tau: float, M: int) -> np.array:
    q_omegas = [1., 1.]
    for l in range(1, M):
        q_omegas.append((2-tau**2 * omega**2)*q_omegas[-1] - q_omegas[-2])
    return np.array(q_omegas)
    

def beta (omega: float, alpha: np.array, tau: float) -> float:
    q = q_omega(omega, tau, len(alpha)-1)
    return tau * alpha @ q


def plot_beta(start: float, end: float, alpha, tau, M, ax, label=None, absolute=True):
    mesh = np.linspace(start, end, num=1000)
    if isinstance(alpha, Callable):
        alpha_vec = np.array(list(map(lambda l: alpha(tau*l), range(0, M+1))))
    else:
        alpha_vec = np.array(alpha)
    abs_fun = abs if absolute else lambda x: x
    ax.plot(mesh, list(map(lambda om: abs_fun(beta(om, alpha_vec, tau)), mesh)), label=label)
        
    

