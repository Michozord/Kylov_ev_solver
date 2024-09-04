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
    """
    Simple enum to distinguish types of filter functions
    """
    L2 = 1          # L2 minimization
    CHEB = 2        # Collocation / Least squares in Chebyshev nodes 
    FOURIER = 3     # Fourier transform
    OTHER = 0       # other 


class Filter(np.ndarray):
    """
    Class to store filter as a numpy ndarray (actually evaluation of weights
    alpha at points 0, tau, 2*tau, ..., tau*(L-1)) with its parameters:
    time-step tau, omega_end, number of time-steps L and derivation method of 
    the filter (FilterType).
    """
    def __new__(cls, array_input, filter_type, om_end: float, tau: float):
        """
        Constructor of a new filter.

        Parameters
        ----------
        array_input
            Evaluation of weights alpha at points 0, tau, 2*tau, ..., tau*(L-1)
            as a list, tuple or anything that can be casted to a numpy ndarray.
        filter_type : FilterType
        
        om_end : float
            Omega_end, om_end > 0.
        tau : float
            Time-step, tau > 0.

        Returns
        -------
        obj

        """
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
        # string representation
        return f"{FilterType(self.filter_type).name} Filter object on (0, {self.om_end}), L = {self.L}, tau = {self.tau}\n" + super().__repr__()
    
    def plot(self, start: Optional[float] = 0, end: Optional[float] = None, 
             ax: Optional[Axes] = None, num: Optional[int] = 10000, 
             **kwargs) -> Axes:
        """
        This method plots filter function beta(omega). The method creates new 
        axis or creates plot on given one.

        Parameters
        ----------
        start : Optional[float], optional
            Start of the plot. The default is 0.
        end : Optional[float], optional
            End of the plot. The default is None: in this case end = om_end.
        ax : Optional[Axes], optional
            Axes object, where the plot is created, if not None. Otherwise the method 
            plots on a new axis. The default is None.
        num : Optional[int], optional
            Fineness of the plot, i.e., number of sample points in the interval 
            (start, end). The default is 10000.
        **kwargs 
            Kwargs for matplotlib.axes.Axes.plot() method.

        Returns
        -------
        Axes
            Axes object with the plot.

        """
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
    
    
    def plot2(self, start: Optional[int] = 0, end: Optional[int] = None, 
             ax: Optional[Axes] = None, num: Optional[int] = 10000, 
             **kwargs) -> Axes:
        """
        Copy of the function Filter.plot(). The only exception is that it plots
        beta(omega^2), not beta(omega).

        Parameters
        ----------
        see Filter.plot().
        

        Returns
        -------
        see Filter.plot().
        

        """
        if end is None: end = self.om_end
        if ax is None:
            fig, ax = plt.subplots()
            plt.grid()
            ax.set_title(f"{FilterType(self.filter_type).name} Filter on (0, {self.om_end}): L = {self.L}, " + r"$\tau$ "+ f"= {self.tau}")
        plot_mesh = np.sqrt(np.linspace(start, end, num=num))
        Q = _q_eval_mat(self.L, self.tau, plot_mesh)
        vals = abs(self.tau * Q @ self)
        ax.plot(plot_mesh*plot_mesh, vals, **kwargs)
        return ax


@dataclass
class FilterGenerator:
    """
    This class has methods, that generate weights (alpha) in standard way:
    by L2 minimization or collocation / least-squares in Chebyshev nodes.
    Parameters
    ----------
    _L: int
        number of time-steps
    _tau: float
        time-step
    _om_min: float
        start of the target interval
    _om_max: float
        end of the target interval
    _om_end: float
        end of the controlled interval
    
    """
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
        """
        Returns weights (as Filter) obtained by collocation or least-squares 
        fitting in Chebyshev nodes in omega^2.

        Parameters
        ----------
        K : int
            Number of nodes.

        Returns
        -------
        Filter

        """
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
        """
        Returns weights (as Filter) obtained by L2 minimization.

        Parameters
        ----------
        K : Optional[int], optional
            Number of sample points for numerical quadrature in each unit. The default is 20.

        Returns
        -------
        Filter

        """
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
        """
        Returns weights (as Filter) obtained by inverse Fourier transform. Note:
        this method works only for negative Laplacian problem! 

        Returns
        -------
        Filter

        """
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
        """
        Returns
        -------
        Callable
            Indicator function chi_[om_min, om_max] (x).

        """
        chi = lambda x: 1 if x >= self.om_min and x <= self.om_max else 0
        return chi
    
    def plot_chebyshev_nodes(self, N: int, ax: Optional[Axes] = None, marker="x", **kwargs) -> Axes:
        """
        Plots N Chebyshev nodes in omega^2 on omega-scaled axis.

        Parameters
        ----------
        N : int
            Number of nodes.
        ax : Optional[Axes], optional
            Axes object, where nodes should be plotted. If None, plot is on 
            a new axis. The default is None.
        marker : str, optional
            A matplotlib marker. The default is "x".
        **kwargs
            matplotlib.axes.Axes.plot() method.

        Returns
        -------
        Axes
            Axes object with plotted nodes.

        """
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
    

