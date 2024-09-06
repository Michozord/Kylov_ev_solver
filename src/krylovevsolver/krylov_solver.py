# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:44:27 2024

@author: Michal Trojanowski
"""

from ngsolve import * 
from ngsolve.fem import CoefficientFunction
from ngsolve.comp import ProxyFunction
from netgen.geom2d import SplineGeometry
from scipy.sparse import csr_matrix
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import time
from typing import List, Tuple, Dict, Callable

from krylovevsolver.dff import Filter


class Results(dict):
    """
    Simple class to store results of the Krylov iteration. Each item has key k 
    numer of iteration and value containing eigenprairs.
    """
    def __getitem__(self, k):
        if k == -1:
            if not self.items():
                raise KeyError("No items in this dictionary.")
            max_key = max(self.keys())
            return super().__getitem__(max_key)
        else:
            return super().__getitem__(k)


class KrylovSolver():
    """
    This class performs FEM with Krylov iteration: discretizes the solution space, 
    computes the discretization matrices and solves for its eigenpairs using Krylov 
    iteration.
    
    Parameters
    ----------
    s: Callable[[ngsolve.comp.ProxyFunction, ngsolve.comp.ProxyFunction], ngsolve.fem.CoefficientFunction]
        Left-hand-site of the weak formulation of the problem to solve.
    m : Callable[[ngsolve.comp.ProxyFunction, ngsolve.comp.ProxyFunction], ngsolve.fem.CoefficientFunction]
        Right-hand-site of the weak formulation of the problem to solve.
    mesh : ngsolve.comp.Mesh
        Mesh object of the discretized domain.
    L : int
        L > 0. Number of time-steps in each iteation.
    tau : tau
        tau > 0. Size of each time-step.
    alpha : Filter
        Filter object representing discrete filter function (dff).
    m_min : int, optional
        From this iteration onwards the results are saved in KrylovSolver.results. The default is 2.
    m_max : int, optional
        Maximal number of Krylov iterations. The default is 30.
    """
    def __init__(self, s: Callable[[ProxyFunction, ProxyFunction], CoefficientFunction], m: Callable[[ProxyFunction, ProxyFunction], CoefficientFunction], mesh: comp.Mesh, L: int, tau: float, alpha: Filter, m_min: int = 2, m_max: int = 30):
       self.s = s
       self.m = m
       self.mesh = None
       self.fes = None
       self.gf = None
       self.MinvS = None
       self.true_eigvals = []
       self.L = L
       self.tau = tau
       self.alpha = alpha
       self.m_min = m_min
       self._m_max = m_max
       self.mesh = mesh
       self.results = Results()

        
        
    def discretize(self, **kwargs):
        """
        This method discretizes the problem: creates solution space with its basis, 
        prepares matrices M and S.

        Parameters
        ----------
        **kwargs 
            kwargs for generation of ngsolve.H1 solution space.

        """
        tm = time.time()
        self.fes = H1(self.mesh, **kwargs)     # H1 solution space
        self.gf = GridFunction(self.fes, multidim=self.mesh.nv)    # basis functions 
        for i in range (self.mesh.nv):
            self.gf.vecs[i][:] = 0
            self.gf.vecs[i][i] = 1
                   
        u, v = self.fes.TnT()  # symbolic objects for trial and test functions in H1
        
        print(f"Triangularization done after {time.time()-tm:.5f} seconds:\n\t{self.fes.ndof} degrees of freedom.")
        tm = time.time()
        
        s = BilinearForm(self.fes)
        s += self.s(u, v)*dx
        s.Assemble()
        S = csr_matrix(s.mat.CSR()).toarray()
        
        m = BilinearForm(self.fes)
        m += self.m(u, v)*dx
        m.Assemble()
        M = csr_matrix(m.mat.CSR()).toarray()
        self.MinvS = np.linalg.inv(M) @ S   
        print(f"Discretization matrices computed after {time.time()-tm:.5f} seconds.")
        
    def compute_true_eigvals(self):
        """
        Computes true eigenvalues of the M^-1 S matrix. If this method has been called, 
        true eigenvalues are added to plots in KrylovSolver.plot_results() method.
        
        NOTE: This method should be used in small-scale examples for comparison of the
        results of the Krylov iteration with true eigenvalues only! In large-scale 
        examples it ruins the performance of the whole method, since it uses direct 
        solver on large matrices S and M.

        """
        print("Warning! This method should be used for demonstation of the results in small-scale examples only. In large-scale problems it ruins performance of the Krylov eigenvalue solver!")
        eigvals, eigvecs = np.linalg.eig(self.MinvS)
        if np.sqrt(max(eigvals)) > 2/self.tau:
            print(f"Warning! There are eigenvalues of MinvS exceeding controlled interval! {np.sqrt(max(eigvals))} > {2/self.tau}")
        self.true_eigvals = eigvals
        

    def solve(self) -> Results[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Core method, that performs the Krylov iteration to compute eigenvalues omega^2
        with corresponding eigenvectors. It stores the results of steps between m_min and m_max
        in the KrylovSolver.results property and returns them.

        Returns
        -------
        results : Results[int, Tuple[np.ndarray, np.ndarray]]
            Results in each step between m_min and m_max in a dictionary-like form.
            Each item has key k numer of iteration and value - a Tuple corresponding 
            to k-th Krylov step and contains:
                eigvals : np.ndarray
                    np.array of all obtained eigenvalues (omega^2 in this step)
                eigvecs: np.ndarray
                    np.array with eigenvectors in columns. eigvecs[:,i] is an eigenvector to eigvals[i].
        """    
        N = self.MinvS.shape[0]
        r = np.random.rand(N)
        r /= np.linalg.norm(r)
        r = r.reshape((N,1))
        self.B = deepcopy(r)
        self._solve(1)  # solve eigenvalue problem starting at 1 
        return self.results
    
    @property
    def m_max(self):
        return self._m_max

    @m_max.setter
    def m_max(self, m_max: int):
        if m_max <= self.m_max:
            print("New value of m_max cannot be smaller than the old one.")
            pass
        old_m_max = self._m_max
        self._m_max = m_max
        if self.results:
            self._solve(old_m_max + 1)
    
    def _solve(self, start: int):
        # this method performs the Krylov iteration starting at the point start
        # to self.m_max
        tm = time.time()
        L, tau, alpha = self.L, self.tau, self.alpha
        N = self.MinvS.shape[0]
        tau2 = tau*tau
        MinvS = self.MinvS
        for k in range(start, self.m_max+1):
            y_pp = deepcopy(self.B[:,-1])              # y_(l-2)
            y_p = y_pp - tau2/2 * MinvS @ y_pp   # y_(l-1)
            b = tau * alpha[0] * y_pp + tau * alpha[1]* y_p
            for l in range(2, L):   
                y = - tau2 * MinvS @ y_p + 2 * y_p - y_pp
                b += tau * alpha[l] * y
                y_pp = y_p
                y_p = y
                
            # Gram-Schmidt:
            proj = np.zeros(b.shape)
            for i in range(k):
                proj += ((self.B[:,i] @ b) * self.B[:,i]).reshape(proj.shape)
            b -= proj
            
            if np.linalg.norm(b) < 1e-13:
                print(f"Exact eigenspace is a subspace of {k-1} Krylov space, ||b_{k}|| = {np.linalg.norm(b)}.\nBreaking Krylov iteration.")
                A = self.B.transpose() @ MinvS @ self.B
                eigvals, eigvecs = np.linalg.eig(A)
                self.results[k] = (np.real(eigvals), self.B @ np.real(eigvecs))
                break
            
            b /= np.linalg.norm(b)
            self.B = np.concatenate((self.B, b.reshape((N, 1))), axis=1)
    
            if k >= self.m_min:
                
                A = self.B.transpose() @ MinvS @ self.B
                eigvals, eigvecs = np.linalg.eig(A)
                self.results[k] = (np.real(eigvals), self.B @ np.real(eigvecs))
        print(f"Krylov iteration done after {time.time()-tm:.5f} seconds.")

    
    
    def _color(self, value):
        # color and marker of omega depending on its accuracy
        dist = np.min(np.abs(self.true_eigvals - value))
        norm = mcolors.Normalize(vmin=-15, vmax=0)
        cmap = plt.get_cmap('cividis')
        marker = "x" if dist > 1e-5 else "o"
        return marker, cmap(norm(np.log(dist)))
        
        
    def plot(self, start: float, end: float, title: str="", plot_filter: bool=True, 
                     label_om: str=r"$\omega$", label_step: str=r"$k$", 
                     label_filter: str=r"$|\tilde{\beta}_{\vec{\alpha}}(\omega)|$",
                     ev_marker: str="x", ev_color: str="blue",
                     filter_plot_kwargs: dict={"color":"red"}):
        """
        Generates plot presenting obtained resonances (omegas). The resonance-axis
        is scaled in omega (presents square roots of eigenvalues).

        Parameters
        ----------
        start : float
            start point of the plot.
        end : float
            end point of the plot.
        title : str, optional
            Title of the plot. The default is "".
        plot_filter : bool, optional
            If True, plot of the filter function is appended to the plot. The default is True.
        label_om : str, optional
            Label on the omega (horizontal)-axis. The default is r"$\\omega$".
        label_step : str, optional
            Label on the iteration-step k (vertical left)-axis. The default is r"$k$".
        label_filter : str, optional
            Label on the filter function (vertical right)-axis. The default is r"$|\\tilde{\\beta}_{\\vec{\\alpha}}(\\omega)|$".
        ev_marker : str, optional
            Marker for omegas. The default is "x".
        ev_color : str, optional
            Color of eigenvalues. The default is "blue".
        filter_plot_kwargs : dict, optional
            Dictionary with kwargs for the plot of the filter function. All kwargs of plt.plot method are supported. The default is {"color":"red"}.

        Raises
        ------
        RuntimeError
            If there are no results in the KrylovSolver. Use solve() method and try again.

        """
        if not self.results:
            raise RuntimeError("There are no results to plot!")
            
        fig = plt.figure()
        plt.title(title)
        ax1 = plt.subplot()
        ax1.grid(axis="y")
        ax1.set_xlim(start, end)
        ax1.set_ylim(0, self.m_max)
        ax1.set_ylabel(label_step, fontname="serif")
        ax1.set_xlabel(label_om, fontname="serif")
        if plot_filter:
            ax2 = ax1.twinx()
            ax2.set_ylim(0, 1.2)
            ax2.set_ylabel(label_filter, fontname="serif")
            fig.tight_layout()
            self.alpha.plot(start, end, ax2, **filter_plot_kwargs)
        
        for omega in np.sqrt(self.true_eigvals):
            ax1.axvline(omega, linestyle=':', color='grey') # vertical lines in true eigvals
            
        for k, (eigvals, _) in self.results.items():
            for eigval in eigvals:
                marker, clr = self._color(eigval) if len(self.true_eigvals)>0 else (ev_marker, ev_color)
                # marker, clr = ev_marker, ev_color
                ax1.plot(np.sqrt(abs(eigval)), k, marker, color=clr, markerfacecolor='none')
        
        plt.show()
        
    
    def plot2(self, start: float, end: float, title: str="", plot_filter: bool=True, 
                     label_om: str=r"$\omega^2$", label_step: str=r"$k$", 
                     label_filter: str=r"$|\tilde{\beta}_{\vec{\alpha}}(\omega^2)|$",
                     ev_marker: str="x", ev_color: str="blue",
                     filter_plot_kwargs: dict={"color":"red"}):
        """
        Generates plot presenting obtained eigenvalues (omegas^2). The eigenvalue-axis
        is scaled in omega^2 (presents eigenvalues).

        Parameters
        ----------
        see method KrylovSolver.plot()

        Raises
        ------
        see method KrylovSolver.plot()

        """
        
        if not self.results:
            raise RuntimeError("There are no results to plot!")
            
        fig = plt.figure()
        plt.title(title)
        ax1 = plt.subplot()
        ax1.grid(axis="y")
        ax1.set_xlim(start, end)
        ax1.set_ylim(0, self.m_max)
        ax1.set_ylabel(label_step, fontname="serif")
        ax1.set_xlabel(label_om, fontname="serif")
        if plot_filter:
            ax2 = ax1.twinx()
            ax2.set_ylim(0, 1.2)
            ax2.set_ylabel(label_filter, fontname="serif")
            fig.tight_layout()
            self.alpha.plot2(start, end, ax2, **filter_plot_kwargs)
        
        for omega in np.sqrt(self.true_eigvals):
            ax1.axvline(omega, linestyle=':', color='grey') # vertical lines in true eigvals
            
        for k, (eigvals, _) in self.results.items():
            for eigval in eigvals:
                marker, clr = self._color(eigval) if len(self.true_eigvals)>0 else (ev_marker, ev_color)
                # marker, clr = ev_marker, ev_color
                ax1.plot(eigval, k, marker, color=clr, markerfacecolor='none')
        
        plt.show()
                
        
    def get_single_result(self, ev: float, k: int=-1) -> tuple[float, np.array]:
        """
        Returns computed eigenvalue closest to given ev with its eigenvector 
        after k-th step of the Krylov iteration.

        Parameters
        ----------
        ev : float
            Eigenvalue (omega^2), to which closest value should be returned.
        k : int, optional
            Step of the itereation. Use -1 for last iteration. The default is -1.

        Raises
        ------
        RuntimeError
            If there are no results in the KrylovSolver. Use solve() method and try again.
        ValueError
            If given step k is not in stored results.

        Returns
        -------
        float
            Eigenvalue (omega^2) in results of the k-th step closest to ev.
        np.array
            Eigenvector to sought eigenvalue.

        """
        if not self.results:
            raise RuntimeError("There are no results to return!")
        
        eigvals, eigvecs = self.results[k]
        
        index = np.nanargmin(np.abs(eigvals - ev))
        return eigvals[index], eigvecs[:,index]
            
 
    
