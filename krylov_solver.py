# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:44:27 2024

@author: Michal Trojanowski
"""

from ngsolve import *
# from ngsolve.webgui import Draw
# import netgen.geom2d as geom2d
from scipy.sparse import csr_matrix
from dff import *
from l2_minimization import compute_alpha as alpha_l2
from chebyshev_collocation import compute_alpha as alpha_cheb
import numpy as np
from types import FunctionType
from copy import deepcopy
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors


class KrylovSolver():
    def __init__(self, mesh, L, tau, alpha, m_min = 2, m_max = 50):
       self.mesh = None
       self.fes = None
       self.gf = None
       self.M_inv = None
       self.S = None
       self.true_eigvals = []
       self.L = L
       self.tau = tau
       self.alpha = alpha
       self.m_min = m_min
       self.m_max = m_max
       self.mesh = mesh
        
        
    def discretize(self):       
        self.fes = H1(self.mesh, order=1)     # H1 solution space
        self.gf = GridFunction(self.fes, multidim=self.mesh.nv)    # basis functions 
        for i in range (self.mesh.nv):
            self.gf.vecs[i][:] = 0
            self.gf.vecs[i][i] = 1
            
        
        u, v = self.fes.TnT()  # symbolic objects for trial and test functions in H1
        
        s = BilinearForm(self.fes)
        s += grad(u)*grad(v)*dx
        s.Assemble()
        self.S = csr_matrix(s.mat.CSR()).toarray()
        
        m = BilinearForm(self.fes)
        m += u*v*dx
        m.Assemble()
        M = csr_matrix(m.mat.CSR()).toarray()
        self.M_inv = np.linalg.inv(M)
        self._compute_true_eigvals()
        
        
    def _compute_true_eigvals(self):
        eigvals, eigvecs = np.linalg.eig(self.M_inv @ self.S)
        self.true_eigvals = eigvals
        
            
    def solve(self):
        L, tau, alpha = self.L, self.tau, self.alpha
        M_inv, S = self.M_inv, self.S
        N = S.shape[0]
    
        if s:= M_inv.shape != (N,N):
            raise ValueError(f"Matrix M^-1 has invalid dimension {s}, while start vector has dimension {N}!")
        if s:= S.shape != (N,N):
            raise ValueError(f"Matrix S has invalid dimension {s}, while start vector has dimension {N}!")
            
        r = np.random.rand(N)
        r /= np.linalg.norm(r)
        r = r.reshape((N,1))
        
        if isinstance(alpha, FunctionType):
            alpha = np.array(list(map(lambda l: alpha(tau*l), range(L))))
        
        tau2 = tau*tau
        MinvS = M_inv @ S
        
        results = []
        
        B = deepcopy(r)
        for k in range(1, self.m_max+1):
            y_pp = deepcopy(B[:,-1])              # y_(l-2)
            y_p = y_pp - tau2/2 * MinvS @ y_pp    # y_(l-1)
            b = tau * alpha[0] * y_pp + tau * alpha[1]* y_p
            for l in range(2, L):   
                y = - tau2 * MinvS @ y_p + 2 * y_p - y_pp
                b += tau * alpha[l] * y
                y_pp = y_p
                y_p = y
                
            # Gram-Schmidt:
            proj = np.zeros(b.shape)
            for i in range(k):
                proj += ((B[:,i] @ b) * B[:,i]).reshape(proj.shape)
            b -= proj
            
            if np.linalg.norm(b) < 1e-13:
                print(f"Exact eigenspace is a subspace of {k-1} Krylov space, ||b_{k}|| = {np.linalg.norm(b)}.\nBreaking Krylov iteration.")
                A = B.transpose() @ MinvS @ B
                eigvals, eigvecs = np.linalg.eig(A)
                results.append((k, np.real(eigvals), B @ np.real(eigvecs)))
                break
            
            b /= np.linalg.norm(b)
            B = np.concatenate((B, b.reshape((N, 1))), axis=1)
    
            if k >= self.m_min:
                
                A = B.transpose() @ MinvS @ B
                eigvals, eigvecs = np.linalg.eig(A)
                results.append((k, np.real(eigvals), B @ np.real(eigvecs)))
        
        self.results = results                    
        return results
    
    def _prepare_plot(self, start, end, title=""):
        fig = plt.figure()
        plt.title(title)
        ax1 = plt.subplot()
        ax1.grid(axis="y")
        ax1.set_xlim(start, end)
        ax1.set_ylim(0, self.m_max)
        ax1.set_ylabel(r"$k$")
        ax1.set_xlabel(r"$\omega$")
        ax2 = ax1.twinx()
        ax2.set_ylim(0, 1.2)
        ax2.set_ylabel(r"$|\beta_\alpha(\omega)|$")
        fig.tight_layout()
        for omega in np.sqrt(self.true_eigvals):
            ax1.axvline(omega, linestyle=':', color='grey') # vertical lines in true eigvals
        return ax1, ax2
    
    
    def _color(self, value):
        dist = np.min(np.abs(self.true_eigvals - value))
        norm = mcolors.Normalize(vmin=-15, vmax=0)
        cmap = plt.get_cmap('cividis')
        marker = "x" if dist > 1e-5 else "o"
        return marker, cmap(norm(np.log(dist)))
        
        
    def plot_results(self, start, end, title=""):
        if self.results is None:
            raise RuntimeError("There are no results to plot!")
        # E = np.zeros((self.fes.ndof+1, len(self.results)))
        # for i in range(len(self.results)):
        #     result = self.results[i]
        #     eigvals_sorted = np.sort(result[1])
        #     E[0,i] = result[0]
        #     E[1:len(eigvals_sorted)+1,i] = eigvals_sorted
        
        ax1, ax2 = self._prepare_plot(start, end, title=title)
        plot_beta(self.alpha, self.L, self.tau, start, end, ax2, color="red")        
        for k, eigvals, _ in self.results:
            for eigval in eigvals:
                marker, clr = self._color(eigval)
                ax1.plot(np.sqrt(eigval), k, marker, color=clr, markerfacecolor='none')
        
        # plt.show()
    
    def convergence(self, true_eigval):
        eigval_index = (np.abs(self.true_eigvals - true_eigval)).argmin()
        
        if self.results is None:
            raise RuntimeError("There are no results to plot!")
            
        dists = []
        ks = []
        for k, eigvals, _ in self.results:
            for eigval in eigvals:
                closest_eigval_index = (np.abs(self.true_eigvals - eigval)).argmin()
                if closest_eigval_index != eigval_index:
                    continue
                ks.append(k)
                dists.append(np.abs(self.true_eigvals[eigval_index] - eigval))
        return self.true_eigvals[eigval_index], ks, dists 
 
    
def plot_convergence(*data):
    fig, ax = plt.subplots()
    for ks, dists, label in data:
        plt.semilogy(ks, dists, label=label, marker="x", linestyle="solid")

    plt.legend()
    # plt.show()


def test():
    tau = 0.0056        # controlled interval up to omega_end = 360
    L = 200
    alpha1 = fourier_indicator(12, 15, tau*L)
    alpha2 = alpha_l2(2/tau, L, tau, 12, 15, K=10)
    alpha3 = alpha_cheb(2/tau, L, L, tau, indicator(12, 15))
    ax = prepare_plots(0, 2/tau)
    plot_beta(alpha1, L, tau, 0, 2/tau, ax, label="Fourier")
    plot_beta(alpha2, L, tau, 0, 2/tau, ax, label="L2")
    plot_beta(alpha3, L, tau, 0, 2/tau, ax, label="Chebyshev coll")
    plt.legend()
    
    seek_ev = 14*14
    
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
    
    solver = KrylovSolver(mesh, L, tau, alpha1, m_max = 50)
    solver.discretize()
    solver.solve()
    solver.plot_results(0, 25, "Fourier filter function")
    omega2_1, ks1, dists1 = solver.convergence(seek_ev)
    
    solver2 = KrylovSolver(mesh, L, tau, alpha2, m_max = 50)
    solver.discretize()
    solver.solve()
    solver.plot_results(0, 25, "L2 filter function")
    omega2_2, ks2, dists2 = solver.convergence(seek_ev)
    
    solver3 = KrylovSolver(mesh, L, tau, alpha3, m_max = 50)
    solver.discretize()
    solver.solve()
    solver.plot_results(0, 25, "Chebyshev filter function")
    omega2_3, ks3, dists3 = solver.convergence(seek_ev)
    
    plot_convergence((ks1, dists1, "Fourier"), (ks2, dists2, "L2"), (ks3, dists3, "Chebyshev"))
    
    plt.show()
    
test()
