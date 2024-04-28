# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:08:07 2024

@author: Michal Trojanowski
"""

from matplotlib import pyplot as plt
import matplotlib.style as style
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
              end: float, ax: plt.axis, cheb: Optional[bool] = True, label: Optional[str] = "", 
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

def plot_nodes(nodes: np.array, target: Callable, ax: plt.axis, label: Optional[str]="", color: Optional[str]="black"):
    ax.plot(nodes, list(map(target, nodes)), "x", color = color, label=label, markersize=12)


def prepare_plots(*ranges, title: Optional[str]="", xlabel: Optional[str]=r"$\omega$", 
                 ylabel: Optional[str]=r"$|\tilde{\beta}_{\alpha}(\omega)|$",
                 fontsize: Optional[int]=None, set_y_lim: Optional[bool]=True) -> Union[plt.axis, tuple[plt.axis]]:
    style.use("classic")
    plt.rcParams.update({'axes.formatter.offset_threshold': 5, 'lines.linewidth': 1.5})
    if fontsize:
        plt.rcParams.update({'font.size': fontsize})
    
    if int(len(ranges))%2 != 0:
        raise ValueError(f"{len(ranges)} starts/ends provided. For last plot there is no end value!")
    
    axes = tuple()
    for start, end in [(ranges[i], ranges[i + 1]) for i in range(0, len(ranges), 2)]:
        fig = plt.figure(facecolor="white")
        ax = plt.subplot()
        ax.set_xlim(start, end)
        if set_y_lim:
            ax.set_ylim(-0.5, 1.5)
        plt.grid()
        axes += (ax,)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
    if len(axes) == 1:
        return axes[0]
    else:
        return axes
    
if __name__ == "__main__":
    omegas = np.linspace(0, 10, num=11)
    tau = 0.1
    L = 4
    print(q_eval_mat(omegas, L, tau))
    
    


        