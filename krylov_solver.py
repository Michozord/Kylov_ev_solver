# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:37:32 2024

@author: Michal Trojanowski
"""

import numpy as np
from types import FunctionType
from copy import deepcopy
from dff import *
# import cmath


def power_iter(A):
    def stop_criterion(r,v):
        q = ((A@r)[0])/(r[0])
        if np.linalg.norm(q*r - v) < 1e-3:
            return False
        else:
            return True
    # def stop_criterion(r,v):
    #     return i < 100
        
    N = A.shape[0]
    r = np.random.rand(N)
    v = A @ r
    i = 1
    while stop_criterion(r,v):
        r = deepcopy(v)
        v = A@v
        v /= np.linalg.norm(v)
        i += 1
        if i > 500:
            print(f"Warning! Maximal number of iterations exceeded.")
            break
    # print(f"Power iteration returned after {i} steps.")
    val = (A@v)[0]/v[0]
    return (v, val)
        
    

def krylov_solver(M_inv, S, r, tau, L, m_min, m_max, alpha):
    N = len(r)
    if s:= M_inv.shape != (N,N):
        raise ValueError(f"Matrix M^-1 has invalid dimension {s}, while start vector has dimension {N}!")
    if s:= S.shape != (N,N):
        raise ValueError(f"Matrix S has invalid dimension {s}, while start vector has dimension {N}!")
    r /= np.linalg.norm(r)
    
    if isinstance(alpha, FunctionType):
        alpha = np.array(list(map(lambda l: alpha(tau*l), range(L))))
    
    tau2 = tau*tau
    MinvS = M_inv @ S
    
    results = []
    
    r = r.reshape((N, 1))
    B= deepcopy(r)
    for k in range(1, m_max+1):
        y_pp = deepcopy(r)              # y_(l-2)
        y_p = r - tau2/2 * MinvS @ r    # y_(l-1)
        b = tau * alpha[0] * r.reshape((N,1))
        for l in range(2, L):   # maybe check needed, idk
            y = - tau2 * MinvS @ y_p + 2 * y_p - y_pp
            b += tau * alpha[l] * y
            y_pp = y_p
            y_p = y
            
        # Gram-Schmidt:
        proj = np.zeros(b.shape)
        for i in range(k):
            proj += ((B[:,i] @ b)[0] * B[:,i]).reshape(proj.shape)
        b -= proj
        
        if np.linalg.norm(b) < 1e-13:
            print(f"Exact eigenspace is a subspace of {k-1} Krylov space, ||b_{k}|| = {np.linalg.norm(b)}.\nBreaking Krylov iteration.")
            A = B.transpose() @ MinvS @ B
            omega2, v = _direct_solver(A, B, MinvS)
            results.append((k-1, omega2, v))
            break
        
        b /= np.linalg.norm(b)
        B = np.concatenate((B, b), axis=1)

        if k >= m_min:
            A = B.transpose() @ MinvS @ B
            omega2, v = _direct_solver(A, B, MinvS)
            if omega2 is None:
                breakpoint()
                raise RuntimeError("does not work")
            else:
                results.append((k, omega2, v))
                print(f"k = {k}:\tomega = {np.sqrt(np.real(omega2))}")
    breakpoint()
    return results


def _direct_solver(A, B, MinvS):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    max_index = np.argmax(eigenvalues)
    max_eigenvalue = eigenvalues[max_index]
    coord_v = eigenvectors[:, max_index]
    v = B @ coord_v
    lam = _is_eigenvector(MinvS, v)
    if lam is None:
        return None, None
    else:
        return lam, v
            
        
    
def _is_eigenvector(A, v, tol=1e-6):
    Av = A@v
    ratios = Av / v
    non_zero_indices = np.nonzero(v)[0]
    unique_ratios = np.unique(ratios[non_zero_indices])
    
    # If all the non-zero ratios are the same within a tolerance, then v is an eigenvector
    if len(unique_ratios) == 1 and (np.abs(Av - unique_ratios[0] * v) < tol).all():
        return unique_ratios[0]
    else:
        return None


def test():
    N = 100
    M = np.eye(N)
    S = np.eye(N)
    M_inv = np.eye(N)
    tau = 0.0056
    L = 100
    m_min = 10
    m_max = 40
    alpha = fourier_indicator(12, 14, tau*L)
    r = np.random.rand(N)
    for i in range(1, N):
        S[i,i] = 300/(i*i)
        
    results = krylov_solver(M_inv, S, r, tau, L, m_min, m_max, alpha)
    
test()


def test_power_iteration():
    A = np.array([[2, 0], [0, 3]])
    print(power_iter(A))

# test_power_iteration()

