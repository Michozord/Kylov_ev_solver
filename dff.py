# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:08:07 2024

@author: Michal Trojanowski
"""

from matplotlib import pyplot as plt
import numpy as np
from typing import Callable, Optional, Union
from types import FunctionType

def indicator(om_min: float, om_max: float) -> Callable:
    chi = lambda x: 1 if x >= om_min and x <= om_max else 0
    return chi

def gauss(om_mid: float) -> Callable:
    return (lambda x : np.exp(-(x-om_mid)*(x-om_mid)))

def fourier_indicator(om_min: float, om_max: float, T: float) -> Callable:
    def alpha(t: float) -> float:
        if t == 0:
            return 2/np.pi * (om_max - om_min)
        elif t > 0 and t <= T:
            return 4/(np.pi*t) * np.sin(t * (om_max-om_min)/2) * np.cos(t * (om_max+om_min)/2)
        else:
            return 0
    return alpha

def q_eval_mat(omegas: np.array, L: int, tau: float, cheb: Optional[bool] = False) -> np.array:
    K = len(omegas)
    tau2 = tau*tau
    omegas2 = omegas*omegas
    Q = np.zeros((K, L))
    if not cheb:
        Q[:,0] = np.ones(K)
        Q[:,1] = (1 - tau2 * omegas2) * np.ones(K)
        for l in range(2, L):
            Q[:, l] = (2 - tau2 * omegas2) * Q[:, l-1] - Q[:, l-2]
        return Q
    else:
        Q[:,0] = np.ones(K)
        Q[:,1] = (1 - tau2 * omegas2/2) * np.ones(K)
        for l in range(2, L):
            Q[:, l] = (2 - tau2 * omegas2) * Q[:, l-1] - Q[:, l-2]
        return Q
        
def plot_beta(alpha: Union[Callable, np.array], L: int, tau: float, start: float, 
              end: float, ax: plt.axis, cheb: Optional[bool] = False, label: Optional[str] = "", 
              color: Optional[str]="", num: Optional[int] = 10000, Q: Optional[np.array] = None) -> np.array:
    if isinstance(alpha, FunctionType):
        alpha = np.array(list(map(lambda l: alpha(tau*l), range(L))))
    plot_mesh = np.linspace(start, end, num=num)
    if Q is None:
        Q = q_eval_mat(plot_mesh, L, tau, cheb=cheb)
    vals = abs(tau * Q @ alpha)
    if color:
        ax.plot(plot_mesh, vals, label=label, color=color)
    else:
        ax.plot(plot_mesh, vals, label=label)
    return Q 

def plot_nodes(nodes: np.array, target: Callable, ax: plt.axis):
    ax.plot(nodes, list(map(target, nodes)), "x", color = "black")


def prepare_plot(start: float, end: float, title: Optional[str]="", 
                 xlabel: Optional[str]="", ylabel: Optional[str]="") -> plt.axis:
    fig, ax = plt.subplots()
    plt.xlim(start, end)
    plt.ylim(-0.5, 2)
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return ax    
    
if __name__ == "__main__":
    omegas = np.linspace(0, 10, num=11)
    tau = 0.1
    L = 4
    print(q_eval_mat(omegas, L, tau))
    
    


        