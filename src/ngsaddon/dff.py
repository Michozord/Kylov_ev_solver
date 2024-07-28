# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:08:07 2024

@author: Michal Trojanowski
"""

from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
from math import ceil, floor
import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass
from enum import Enum


class FilterType(Enum):
    L2 = 1
    CHEB = 2
    FOURIER = 3


class Filter(np.ndarray):
    def __new__(cls, array_input, filter_type, om_end: float, tau: float):
        obj = np.asarray(array_input).view(cls)
        obj.filter_type = filter_type
        obj.tau = tau
        obj.om_end = om_end
        obj.L = len(obj)
        return obj
      
    def __array_finalize__(self, obj):
        if obj is None: 
            return
        self.filter_type = getattr(obj, 'filter_type', None)
        self.tau = getattr(obj, 'tau', None)
        self.om_end = getattr(obj, 'om_end', None)
        self.L = getattr(obj, 'L', None)
        
    def __repr__(self):
        return f"{FilterType(self.filter_type).name} Filter object on (0, {self.om_end}), L = {self.L}, tau = {self.tau}\n" + super().__repr__()
    
    def plot(self, start: Optional[int] = 0, end: Optional[int] = None, 
             ax: Optional[Axes] = None, num: Optional[int] = 10000, 
             **kwargs) -> Axes:
        if end is None: end = self.om_end
        if ax is None:
            fig, ax = plt.subplots()
            plt.grid()
            ax.set_title(f"{FilterType(self.filter_type).name} Filter on (0, {self.om_end}): L = {self.L}, " + r"$\tau$ "+ f"= {self.tau}")
        plot_mesh = np.linspace(start, end, num=num)
        Q = _q_eval_mat(self.L, self.tau, plot_mesh)
        vals = abs(self.tau * Q @ self)
        ax.plot(plot_mesh, vals, **kwargs)
        return ax


@dataclass
class FilterGenerator:
    _L: int
    _tau: float
    _om_min: float
    _om_max: float
    _om_end: float
    
    def __post_init__(self):
        if self.L <= 0:
            raise ValueError("Number of time-steps (L) cannot be negative!")
        if self.tau < 0:
            raise ValueError("Time-step (tau) cannot be negative!")
        self._om_check()
        self._cfl_check()

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, value: int):
        if value <= 0:
            raise ValueError("Number of time-steps (L) cannot be negative!")
        self._L = value

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, value: float):
        if value < 0:
            raise ValueError("Time-step (tau) cannot be negative!")
        self._tau = value
        self._cfl_check()

    @property
    def om_min(self):
        return self._om_min

    @om_min.setter
    def om_min(self, value: float):
        self._om_min = value
        self._om_check()

    @property
    def om_max(self):
        return self._om_max

    @om_max.setter
    def om_max(self, value: float):
        self._om_max = value
        self._om_check()

    @property
    def om_end(self):
        return self._om_end

    @om_end.setter
    def om_end(self, value: float):
        self._om_end = value
        self._om_check()
        self._cfl_check()
    
    def _cfl_check(self):
        if self.tau >= 2/self.om_end + 1e-13: 
            raise ValueError(f"CFL condition is violated: tau = {self.tau} > {2/self.om_end} = 2/om_end. For help, see documentation.")
    
    def _om_check(self):
        if self.om_min >= self.om_max or self.om_max > self.om_end or self.om_min < 0:
            raise ValueError("Invalid omega values for FilterGenerator! There must be 0 <= om_min < om_max <= om_end.")
        
    def chebyshev(self, K: int) -> Filter:
        mesh = self._chebyshev_nodes(0, self.om_end, K)
        Q = _q_eval_mat(self.L, self.tau, mesh)
        target = self._indicator()

        rhs = 1/self.tau * np.array(list(map(target, mesh)))
        if K==self.L:
            print(f"[Info]: cond(Q) = {np.linalg.cond(Q)}, det(Q) = {np.linalg.det(Q)}")
            alpha = np.linalg.solve(Q, rhs)
        else:
            QTQ = np.transpose(Q) @ Q
            print(f"[Info]: cond(QTQ) = {np.linalg.cond(QTQ)}, det(QTQ) = {np.linalg.det(QTQ)}")
            alpha = np.linalg.solve(QTQ, np.transpose(Q) @ rhs)
        return Filter(alpha, filter_type = FilterType.CHEB, om_end = self.om_end, tau=self.tau)
    
    def l2(self, K: Optional[int] = 20) -> Filter:
        quad_mesh = np.linspace(0, self.om_end - 1/K, num = int(K*self.om_end)) + 1/(2*K)
        Q = _q_eval_mat(self.L, self.tau, quad_mesh)
        rhs = np.zeros(self.L)
        for l in range(ceil(K*self.om_min), floor(K*self.om_max)+1):
            rhs += Q[l,:]
        rhs /= K
        X = 1/K * Q.transpose() @ Q 
        print(f"[Info]: det(X) = {np.linalg.det(X)}, cond(X) = {np.linalg.cond(X)}")
        alpha = np.linalg.solve(X, rhs)/self.tau
        return Filter(alpha, filter_type = FilterType.L2, om_end = self.om_end, tau=self.tau)
    
    def fourier(self) -> Filter:
        T = self.L * self.tau
        def alpha(t: float) -> float:
            if t == 0:
                return 2/np.pi * (self.om_max - self.om_min)
            elif t > 0 and t <= T:
                return 4/(np.pi*t) * np.sin(t * (self.om_max-self.om_min)/2) * np.cos(t * (self.om_max+self.om_min)/2)
            else:
                return 0
        return Filter([alpha(self.tau*l) for l in range(self.L)], filter_type = FilterType.FOURIER, om_end = self.om_end, tau=self.tau)
    
    def _indicator(self) -> Callable:
        chi = lambda x: 1 if x >= self.om_min and x <= self.om_max else 0
        return chi
    
    def plot_chebyshev_nodes(self, N: int, ax: Optional[Axes] = None, marker="x", **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
            plt.grid()
            ax.set_title(f"{N} Chebyshev nodes in $\\omega^2$.")
        nodes = self._chebyshev_nodes(0, self.om_end, N)
        ax.plot(nodes, list(map(self._indicator(), nodes)), "x", **kwargs)
        return ax
    
    @staticmethod
    def _chebyshev_nodes(a: float, b: float, N: int) -> np.array:
        # chebyshev nodes in omega^2
        a2, b2 = a*a, b*b
        nodes = np.array([np.cos((2*k+1)/(2*N) * np.pi) for k in range(N)])
        nodes = (a2+b2)/2 + (a2-b2)/2 * nodes
        nodes = np.sqrt(nodes)
        return nodes


def _q_eval_mat(L, tau, omegas: np.array) -> np.array:
    K = len(omegas)
    tau2 = tau*tau
    omegas2 = omegas*omegas
    Q = np.zeros((K, L))
    Q[:,0] = np.ones(K)
    Q[:,1] = (1 - tau2 * omegas2/2) * np.ones(K)
    for l in range(2, L):
        Q[:, l] = (2 - tau2 * omegas2) * Q[:, l-1] - Q[:, l-2]
    return Q
    

