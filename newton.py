# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:56:19 2024

@author: Michal Trojanowski
"""

import numpy as np

def newton(f, df, x):
    i = 0
    tol = 1e-5
    while np.linalg.norm(f(x)) > tol:
        if i > 50:
            raise RuntimeError("Newton did not converge!")
        incr = np.linalg.solve(df(x), f(x))
        x = x - incr
        i += 1
            
    return x