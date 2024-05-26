# -*- coding: utf-8 -*-
"""
Created on Sun May 26 22:05:00 2024

@author: Michal Trojanowski
"""

from ngsolve import *
# from ngsolve.webgui import Draw
from scipy.sparse import csr_matrix
from dff import *
import numpy as np
from types import FunctionType
from copy import deepcopy

def _direct_solver(A, B, MinvS):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    max_index = np.argmax(eigenvalues)
    coord_v = eigenvectors[:, max_index]
    lam = (coord_v @ A @ coord_v)/(coord_v @ coord_v)
    v = B @ coord_v
    # lam = _is_eigenvector(MinvS, v)
    if lam is None:
        # lam = (v @ MinvS @ v)/(v @ v)
        # breakpoint()
        return lam, v
    else:
        return lam, v
            

def krylov_solver(M_inv, S, tau, L, m_min, m_max, alpha):
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
    for k in range(1, m_max+1):
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
            omega2, v = _direct_solver(A, B, MinvS)
            results.append((k-1, omega2, v))
            break
        
        b /= np.linalg.norm(b)
        B = np.concatenate((B, b.reshape((N, 1))), axis=1)

        if k >= m_min:
            
            A = B.transpose() @ MinvS @ B
            omega2, v = _direct_solver(A, B, MinvS)
            if False:
                raise RuntimeError("does not work")
            else:
                results.append((k, np.real(omega2), np.real(v)))
                print(f"k = {k}:\tomega = {np.sqrt(np.real(omega2))}")
    # breakpoint()
    return results


def test():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
    
    fes = H1(mesh, order=1)     # H1 solution space
    gf = GridFunction(fes, multidim=mesh.nv)    # basis functions 
    for i in range (mesh.nv):
        gf.vecs[i][:] = 0
        gf.vecs[i][i] = 1
        
    
    u, v = fes.TnT()  # symbolic objects for trial and test functions in H1
    
    s = BilinearForm(fes)
    s += grad(u)*grad(v)*dx
    s.Assemble()
    S = csr_matrix(s.mat.CSR()).toarray()
    
    m = BilinearForm(fes)
    m += u*v*dx
    m.Assemble()
    M = csr_matrix(m.mat.CSR()).toarray()
    
    M_inv = np.linalg.inv(M)
    
    tau = 0.0056
    L = 100
    m_min = 10
    m_max = 137
    alpha = fourier_indicator(12, 14, tau*L)
    
    eigvals, eigvecs = np.linalg.eig(M_inv @ S)
    
    results = krylov_solver(M_inv, S, tau, L, m_min, m_max, alpha)
    return eigvals, eigvecs, results
    
eigvals, eigvecs, results = test()
    

