# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:08:07 2024

@author: Michal Trojanowski
"""

from matplotlib import pyplot as plt
from math import ceil, floor
import numpy as np
from typing import Callable, Optional, Union
from types import FunctionType

# TODO: Class AlphaGenerator
# TODO: Verbose options ?

def _indicator(om_min: float, om_max: float) -> Callable:
    chi = lambda x: 1 if x >= om_min and x <= om_max else 0
    return chi

def _chebyshev_nodes(a: float, b: float, N: int) -> np.array:
    # chebyshev nodes in omega^2
    a2, b2 = a*a, b*b
    nodes = np.array([np.cos((2*k+1)/(2*N) * np.pi) for k in range(N)])
    nodes = (a2+b2)/2 + (a2-b2)/2 * nodes
    nodes = np.sqrt(nodes)
    return nodes


def _generate_fourier(om_min: float, om_max: float, T: float) -> Callable:
    def alpha(t: float) -> float:
        if t == 0:
            return 2/np.pi * (om_max - om_min)
        elif t > 0 and t <= T:
            return 4/(np.pi*t) * np.sin(t * (om_max-om_min)/2) * np.cos(t * (om_max+om_min)/2)
        else:
            return 0
    return alpha

def _generate_chebyshev(om_end: float, L: int, K: int, tau: float, target: Callable) -> np.array:
    mesh = _chebyshev_nodes(0, om_end, K)
    Q = q_eval_mat(mesh, L, tau, cheb=True)

    rhs = 1/tau * np.array(list(map(target, mesh)))
    if K==L:
        print(f"cond(Q) = {np.linalg.cond(Q)}, det(Q) = {np.linalg.det(Q)}")
        alpha = np.linalg.solve(Q, rhs)
        return alpha
    else:
        QTQ = np.transpose(Q) @ Q
        print(f"cond(QTQ) = {np.linalg.cond(QTQ)}, det(QTQ) = {np.linalg.det(QTQ)}")
        alpha = np.linalg.solve(QTQ, np.transpose(Q) @ rhs)
        return alpha
    
def _generate_l2(om_end: float, L: int, tau: float, om_min: float, om_max: float, K: Optional[int] = 20):
    quad_mesh = np.linspace(0, om_end - 1/K, num = int(K*om_end)) + 1/(2*K)
    Q = q_eval_mat(quad_mesh, L, tau)
    rhs = np.zeros(L)
    for l in range(ceil(K*om_min), floor(K*om_max)+1):
        rhs += Q[l,:]
    rhs /= K
    X = 1/K * Q.transpose() @ Q 
    print(f"det(X) = {np.linalg.det(X)}, cond(X) = {np.linalg.cond(X)}")
    alpha = np.linalg.solve(X, rhs)/tau
    return alpha


def _q_eval_mat(omegas: np.array, L: int, tau: float, cheb: Optional[bool] = True) -> np.array:
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
        
def plot_filter(alpha: Union[Callable, np.array], L: int, tau: float, start: float, 
              end: float, ax: plt.axis, cheb: Optional[bool] = True, label: Optional[str] = "", 
              color: Optional[str]="", style="solid", num: Optional[int] = 10000, Q: Optional[np.array] = None) -> np.array:
    if isinstance(alpha, FunctionType):
        alpha = np.array(list(map(lambda l: alpha(tau*l), range(L))))
    plot_mesh = np.linspace(start, end, num=num)
    if Q is None:
        Q = q_eval_mat(plot_mesh, L, tau, cheb=cheb)
    vals = abs(tau * Q @ alpha)
    if color:
        ax.plot(plot_mesh, vals, linestyle=style, label=label, color=color)
    else:
        ax.plot(plot_mesh, vals, linestyle=style, label=label)
    return Q 


def plot_nodes(nodes: np.array, target: Callable, ax: plt.axis, label: Optional[str]="", color: Optional[str]="black", crosses: Optional[str]="x", size: Optional[int]=12):
    ax.plot(nodes, list(map(target, nodes)), crosses, color = color, label=label, markersize=size)
    

########### TODO: TEMPORARY FUNCTIONALITY, TO REMOVE
    
def plot_beta(*args, **kwargs):
    return plot_filter(*args, **kwargs)

def fourier_indicator(*args, **kwargs):
    return _generate_fourier(*args, **kwargs)

def q_eval_mat(*args, **kwargs):
    return _q_eval_mat(*args, **kwargs)

def indicator(*args, **kwargs):
    return _indicator(*args, **kwargs)

    
    


        