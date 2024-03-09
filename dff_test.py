# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:06:35 2024

@author: Michal Trojanowski
"""

from dff import *
import numpy as np
from matplotlib import pyplot as plt
from sympy_test import sympy_plot


def alpha_fourier_indicator(T, omega_min, omega_max):
    
    def alpha(t):
        if t > T:
            return 0
        elif t == 0:
            return 2 * (omega_max-omega_min)/np.pi
        elif 0 < t and t <= T:
            return 4/(np.pi * t) * np.sin(t*(omega_max-omega_min)/2) * np.cos(t*(omega_max+omega_min)/2)
        else:
            return alpha(-t)
        
    return alpha


def alpha_fourier_gauss(T, omega_min, omega_max):
    omega_mid = (omega_max+omega_min)/2
    
    def alpha(t):
        if t > T:
            return 0
        elif t < 0:
            return alpha(-t)
        else:
            return 2 * np.sqrt(1/np.pi) * np.exp(-t*t/4) * np.cos(omega_mid*t)
    
    return alpha

if __name__ == "__main__":
    omega_min = 2
    omega_max = 4
    start = 0
    end = 20
    Ts = [1, 2.5, 5, 10]
    tau = 0.025
    fig, ax = plt.subplots()
    for T in Ts:
        alpha = alpha_fourier_gauss(T, omega_min, omega_max)
        plot_beta(start, end, alpha, tau, int(T/tau), ax, "T="+str(T), absolute=True)  
    ax.grid()
    plt.xlim(start, end)
    plt.ylim(-0.5, 1.5)
    plt.legend()
    plt.show()
        
    
    
    # omega_min = 2
    # omega_max = 4
    # start = 0
    # end = 20
    # T = 1
    # tau = 0.025
    # fig, ax = plt.subplots()
    # alpha = lambda t : t
    # plot_beta(start, end, alpha, tau, int(T/tau), ax, "T="+str(T), absolute=False)
    # mesh = np.linspace(start, end, num=1000)
    # val = tau**2 * np.array([np.array(range(41)) @ q_omega(omega, tau, 40) for omega in mesh])
    # ax.plot(mesh, val, label="ref")
    # ax.grid()
    # plt.xlim(start, end)
    # plt.ylim(-0.5, 1.5)
    # plt.legend()
    # plt.show()
    
    
    # omega_min = 2
    # omega_max = 4
    # start = 0
    # end = 20
    # T = 5
    # tau = 0.025
    # fig, ax = plt.subplots()
    # alpha = alpha_constructor(T, omega_min, omega_max)
    # plot_beta(start, end, alpha, tau, int(T/tau), ax, "plot_beta", absolute=False)
    # sympy_plot(alpha, ax, tau, int(T/tau), T)
    
    # ax.grid()
    # plt.xlim(start, end)
    # plt.ylim(-0.5, 1.5)
    # plt.legend()
    # plt.show()



