# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:19:16 2024

@author: Michal Trojanowski
"""

from ngsolve import *

class Problem():
    # TODO: complete this template class
    ...
    
    
class NegativeLaplacian(Problem):
    def s(u: comp.ProxyFunction, v: comp.ProxyFunction):
        return grad(u)*grad(v)
    
    def m(u: comp.ProxyFunction, v: comp.ProxyFunction):
        return u*v
    
        