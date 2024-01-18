# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:20:39 2024

@author: Michal Trojanowski
"""

import numpy as np

def q_omega(omega, tau, M):
    q_omegas = [1., 1.]
    for l in range(1, M):
        q_omegas += [(2-tau**2 * omega**2)*q_omegas[-1] - q_omegas[-2]]
    return q_omegas
    

def beta (alpha, omega, tau):
    pass