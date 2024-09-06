# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:23:37 2024

@author: Michal Trojanowski
"""

from ngsolve import *
from netgen.geom2d import SplineGeometry
from matplotlib import pyplot as plt
import numpy as np

from ngsaddon import KrylovSolver
from ngsaddon.dff import Filter, FilterGenerator
from ngsaddon.negative_laplacian import s, m


def test():
    tau = 0.0056        # controlled interval up to omega_end = 360
    L = 100

    om_min_1, om_max_1 = 11, 13
    om_min_2, om_max_2 = 6, 8
    alpha1 = FilterGenerator(L, tau, om_min_1, om_max_1, 2/tau).chebyshev(1000)
    alpha2 = FilterGenerator(L, tau, om_min_2, om_max_2, 2/tau).chebyshev(1000)
        
    seek_ev_1, seek_ev_2 = 7**2, 12**2
    
    # mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
    geo = SplineGeometry()
    geo.AddRectangle((0,0),(pow(2, 1/3),1))
    # geo.AddRectangle((0,0),(2,1))
    mesh = Mesh(geo.GenerateMesh(maxh=0.05))
    
    solver = KrylovSolver(s, m, mesh, L, tau, alpha1, m_max = 50)
    solver.discretize()
    solver.solve()
    # solver.plot_results(5, 15, f"Chebyshev filter function ({om_min_1}, {om_max_1})")
    solver.plot(5, 15, "")
    solver.plot2(25, 225, "")
    
    # solver2 = KrylovSolver(mesh, L, tau, alpha2, m_max = 50)
    # solver2.discretize()
    # solver2.solve()
    # # solver2.plot_results(5, 15, f"Chebyshev filter function, ({om_min_2}, {om_max_2})")
    # solver2.plot_results(5, 15, "")
    
    
    
test()