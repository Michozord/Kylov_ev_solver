# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:29:47 2024

@author: Michal Trojanowski
"""

from sympy import Symbol, expand
import numpy as np


x = Symbol('x')


def generate_polynomials(M, tau, T):
    
    # Initial conditions
    q = [1, 1]
    
    # Generate polynomials using the recurrence relation
    for l in range(2, M+1):
        print("generating polynomial ", l, "\n")
        ql = expand((2 - tau**2 * x**2) * q[l-1] - q[l-2])
        q.append(ql)
    
    print("polynomials generated\n")
    return q


def sympy_plot(alpha, ax, tau, M, T):
    mesh = np.linspace(0, 20, num=100)
    beta = 0
    q = generate_polynomials(M, tau, T)
    print("computing sum...")
    for l in range(M+1):
        beta += expand(alpha(tau*l) * tau * q[l])
    print("done\nevaluating polynomials...")
    val = [beta.subs(x, om) for om in mesh]
    print("done")
    ax.plot(mesh, val, label="sympy")
    


    
    
