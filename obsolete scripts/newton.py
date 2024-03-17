# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:56:19 2024

@author: Michal Trojanowski
"""

import numpy as np

def newton(f, df, x, lam_min = 1e-8, q = 0.5):
    i = 0
    lam = 1
    tol = 1e-3
    while np.linalg.norm(f(x)) > tol:
        if i > 50:
            print("Newton did not converge!")
            return x
        incr = np.linalg.solve(df(x), -f(x))
        fx_norm = np.linalg.norm(f(x))
        print(f"i = {i}, ||DF(x)|| = {fx_norm}")
        while lam >= lam_min and np.linalg.norm(f(x+lam*incr)) >= fx_norm:
            lam *= q
        if lam < lam_min:
            raise RuntimeError("Lambda too small!")
        else:
            x = x + lam*incr
            lam = min(1, lam/q)
            i += 1
            
    return x


if __name__ == "__main__":
    def f(x):
        return np.array([x[0]*x[0], 3*x[1]])
    
    def df(x):
        return np.array([[2*x[0], 0], [0, 3]])
    
    x = newton(f, df, np.array([1, 1]))
    print(x)