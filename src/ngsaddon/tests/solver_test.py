# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:51:29 2024

@author: Michal Trojanowski
"""

import unittest 
from ngsaddon import KrylovSolver
from ngsaddon.dff import Filter, FilterGenerator
from ngsolve import *
from netgen.geom2d import SplineGeometry
import numpy as np

tol = 10e-1

class SolverTest(unittest.TestCase):
    def test1(self):
        sought_omegas = np.array([6.3031300268411945, 6.784734955712434, 7.513367041612873])
        
        om_min_1, om_max_1 = 6, 8
        tau = 0.0056
        L = 100
        alpha1 = FilterGenerator(L, tau, om_min_1, om_max_1, 2/tau).fourier()
        geo = SplineGeometry()
        geo.AddRectangle((0,0),(pow(2, 1/3),1))
        mesh = Mesh(geo.GenerateMesh(maxh=0.05))
        solver = KrylovSolver(mesh, L, tau, alpha1, m_max = 30)
        solver.discretize(1)
        solver.solve()
        found_omegas = sorted([np.sqrt(ev) for ev in solver.results[-1][0] if om_min_1**2 < ev and ev < om_max_1**2])
        for sought_omega in sought_omegas:
            with self.subTest():
                self.assertTrue(any([abs(sought_omega - found) < tol for found in found_omegas]))

        
        
    def test2(self):
        sought_omegas = np.array([11.919492599237676,
                                  12.173609393329663,
                                  12.620126129337908,
                                  12.725924302192096,
                                  12.981106038488175])
        
        om_min_1, om_max_1 = 11, 13
        tau = 0.0056
        L = 200
        alpha1 = FilterGenerator(L, tau, om_min_1, om_max_1, 2/tau).fourier()
        geo = SplineGeometry()
        geo.AddRectangle((0,0),(pow(2, 1/3),1))
        mesh = Mesh(geo.GenerateMesh(maxh=0.05))
        solver = KrylovSolver(mesh, L, tau, alpha1, m_max = 30)
        solver.discretize(1)
        solver.solve()
        found_omegas = sorted([np.sqrt(ev) for ev in solver.results[-1][0] if om_min_1**2 < ev and ev < om_max_1**2])
        for sought_omega in sought_omegas:
            with self.subTest():
                self.assertTrue(any([abs(sought_omega - found) < tol for found in found_omegas]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

        
        
        

        