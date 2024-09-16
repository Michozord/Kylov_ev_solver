# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:19:16 2024

@author: Michal Trojanowski
"""

from ngsolve import grad
from ngsolve.comp import ProxyFunction
from ngsolve.fem import CoefficientFunction

    

def s(u: ProxyFunction, v: ProxyFunction) -> CoefficientFunction:
    return grad(u)*grad(v)

def m(u: ProxyFunction, v: ProxyFunction) -> CoefficientFunction:
        return u*v
    
        