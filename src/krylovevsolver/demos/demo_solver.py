# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:23:37 2024

@author: Michal Trojanowski
"""

from ngsolve import *
from netgen.geom2d import SplineGeometry
from matplotlib import pyplot as plt
import numpy as np

from krylovevsolver import KrylovSolver
from krylovevsolver.dff import Filter, FilterGenerator
from krylovevsolver.negative_laplacian import s, m


tau = 0.0056        # controlled interval up to omega_end = 360
L = 100

om_min, om_max = 11, 13
# om_min, om_max = 6, 8
alpha = FilterGenerator(L, tau, om_min, om_max, 2/tau).chebyshev(1000)
    
seek_ev_1, seek_ev_2 = 7**2, 12**2

geo = SplineGeometry()
geo.AddRectangle((0,0),(2,1))
mesh = Mesh(geo.GenerateMesh(maxh=0.05))
# mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))

solver = KrylovSolver(s, m, mesh, L, tau, alpha, m_max = 50)
solver.discretize()
solver.solve()
# solver.compute_true_eigvals()
solver.plot(5, 15, "")
solver.plot2(25, 225, "")
print(solver.get_single_result(seek_ev_1)[0])
print(solver.get_single_result(seek_ev_2, 40)[0])
    