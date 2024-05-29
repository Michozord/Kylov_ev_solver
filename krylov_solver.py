# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:44:27 2024

@author: Michal Trojanowski
"""

from ngsolve import *
# from ngsolve.webgui import Draw
from netgen.geom2d import SplineGeometry
from scipy.sparse import csr_matrix
from dff import *
from l2_minimization import compute_alpha as alpha_l2
from chebyshev_collocation import compute_alpha as alpha_cheb
import numpy as np
from types import FunctionType
from copy import deepcopy
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.style as style


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
        
        print(f"Discretization: {self.fes.ndof} degrees of freedom.")
        
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
        if np.sqrt(max(eigvals)) > 2/self.tau:
            raise RuntimeWarning(f"There are eigenvalues of MinvS exceeding controlled interlval! {np.sqrt(max(eigvals))} > {2/self.tau}")
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
        plt.rcParams.update({'mathtext.fontset' : 'cm', 'grid.color' : 'black', 'grid.linestyle' : ':'})
        plt.title(title)
        ax1 = plt.subplot()
        ax1.grid(axis="y")
        ax1.set_xlim(start, end)
        ax1.set_ylim(0, self.m_max)
        ax1.set_ylabel(r"$k$", fontname="serif")
        ax1.set_xlabel(r"$\omega$", fontname="serif")
        ax2 = ax1.twinx()
        ax2.set_ylim(0, 1.2)
        ax2.set_ylabel(r"$|\tilde{\beta}_{\vec{\alpha}}(\omega)|$", fontname="serif")
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
            
        ax1, ax2 = self._prepare_plot(start, end, title=title)
        plot_beta(self.alpha, self.L, self.tau, start, end, ax2, color="red")
        # ax1.plot(x:=[np.sqrt(k*k*np.pi*np.pi + j*j) for k in range(10) for j in range(10)],[0]*len(x), "X", color="k")
        for k, eigvals, _ in self.results:
            for eigval in eigvals:
                marker, clr = self._color(eigval)
                ax1.plot(np.sqrt(eigval), k, marker, color=clr, markerfacecolor='none')
        
        # plt.show()
        
    def closest_true_eigval(self, val):
        dists = np.abs(self.true_eigvals - val)
        index = np.nanargmin(dists)
        return index, self.true_eigvals[index]
        
        
    
    def convergence(self, value):
        seek_eigval_index, seek_true_eigval = self.closest_true_eigval(value)
        
        ks, resids = [], []
        
        for k, eigvals, _ in self.results:
            dists = np.abs(eigvals - seek_true_eigval)
            closest_index = np.nanargmin(dists)
            # if np.abs(self.closest_true_eigval(eigvals[closest_index])[1] - seek_true_eigval) > 1e-13:
            #     continue
            ks.append(k)
            resids.append(dists[closest_index])

        return seek_true_eigval, ks, resids 
    
    def get_single_result(self, k, ev):
        if self.results is None:
            raise RuntimeError("There are no results to return!")
        
        for result in self.results:
            if k == result[0]:
                k, eigvals, eigvecs = result
                break
        else:
            raise ValueError(f"k={k} not in results!")
        
        index = np.nanargmin(np.abs(eigvals - ev))
        return eigvals[index], eigvecs[:,index]
            
 
    
def plot_convergence(*data):
    fig, ax = plt.subplots(facecolor="white")
    for ks, dists, label, clr, stl, mrk in data:
        plt.semilogy(ks, dists, label=label, marker=mrk, linestyle=stl, color=clr, markersize=15)
    plt.xlim(((10, 35)))
    plt.legend(loc="lower left")
    plt.grid()
    # plt.show()
    
def plot_convergence_2(*data):
    fig, axes = plt.subplots(ncols=2, facecolor="white", sharey=True)
    stl = "solid"
    for ks, dists, label, mrk in data[:2]:
        ax = axes[0]
        ax.semilogy(ks, dists, label=label, marker=mrk, linestyle=stl, markersize=15)
        ax.set_xlim((0, 25))
        ax.set_ylabel(r"$\text{error in } {\omega_{1,1}^2}$")        # TODO: omega_??
    for ks, dists, label, mrk in data[2:]:
        ax = axes[1]
        ax.semilogy(ks, dists, label=label, marker=mrk, linestyle=stl, markersize=15)
        ax.set_xlim((0, 35))
        ax.set_ylabel(r"$\text{error in }{\omega_{1,1}^2}$")        # TODO: omega_??
    
    for ax in axes:
        ax.legend(loc="lower left")
        ax.grid()
        ax.set_xlabel("$k$")
    plt.title(r"Error in $\omega^2$ in convergence to $\approx 7^2$ and $\approx 12^2$ for different filter functions")



def test():
    tau = 0.0056        # controlled interval up to omega_end = 360
    L = 200

    om_min_1, om_max_1 = 11, 13
    om_min_2, om_max_2 = 6, 8
    # alpha1 = alpha_l2(2/tau, L, tau, om_min_1, om_max_1)
    # alpha2 = alpha_l2(2/tau, L, tau, om_min_2, om_max_2)
    alpha1 = alpha_cheb(2/tau, L, 5*L, tau, indicator(om_min_1, om_max_1))
    alpha2 = alpha_cheb(2/tau, L, 5*L, tau, indicator(om_min_2, om_max_2))
    
    # alpha3 = alpha_cheb(2/tau, L, L, tau, indicator(om_min, om_max))
    ax = prepare_plots(0, 2/tau)
    plot_beta(alpha1, L, tau, 0, 2/tau, ax, label=f"({om_min_1}, {om_max_1})")
    plot_beta(alpha2, L, tau, 0, 2/tau, ax, label=f"({om_min_2}, {om_max_2})")
    # plot_beta(alpha3, L, tau, 0, 2/tau, ax, label="Chebyshev coll")
    plt.legend()
    
    seek_ev_1, seek_ev_2 = 7**2, 12**2
    
    # mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
    geo = SplineGeometry()
    geo.AddRectangle((0,0),(pow(2, 1/3),1))
    # geo.AddRectangle((0,0),(2,1))
    mesh = Mesh(geo.GenerateMesh(maxh=0.05))
    
    solver = KrylovSolver(mesh, L, tau, alpha1, m_max = 50)
    solver.discretize()
    solver.solve()
    solver.plot_results(5, 15, f"Chebyshev filter function ({om_min_1}, {om_max_1})")
    omega2_1, ks1, dists1 = solver.convergence(seek_ev_1)
    omega2_3, ks3, dists3 = solver.convergence(seek_ev_2)
    
    solver2 = KrylovSolver(mesh, L, tau, alpha2, m_max = 50)
    solver2.discretize()
    solver2.solve()
    solver2.plot_results(5, 15, f"Chebyshev filter function, ({om_min_2}, {om_max_2})")
    omega2_2, ks2, dists2 = solver2.convergence(seek_ev_1)
    omega2_4, ks4, dists4 = solver2.convergence(seek_ev_2)
    
    plt.show()
    style.use("classic")
    plt.rcParams.update({'axes.formatter.offset_threshold': 5, 'lines.linewidth': 1.5, 'font.size' : 22, 'markers.fillstyle': 'none'})
    
    print(np.sqrt(omega2_1), np.sqrt(omega2_3))
    
    # solver3 = KrylovSolver(mesh, L, tau, alpha3, m_max = 50)
    # solver3.discretize()
    # solver3.solve()
    # solver3.plot_results(0, 25, "Chebyshev filter function")
    # omega2_3, ks3, dists3 = solver.convergence(seek_ev)
    
    plot_convergence_2((ks2, dists2, f"{np.sqrt(seek_ev_1)}: $({om_min_2}, {om_max_2})$",  "o"),
                       (ks1, dists1, f"{np.sqrt(seek_ev_1)}: $({om_min_1}, {om_max_1})$",  "x"),
                       (ks2, dists4, f"{np.sqrt(seek_ev_2)}: $({om_min_2}, {om_max_2})$",  "o"),
                       (ks1, dists3, f"{np.sqrt(seek_ev_2)}: $({om_min_1}, {om_max_1})$",  "x"))
    
    plt.show()
    

if __name__ == "__main__":
    test()