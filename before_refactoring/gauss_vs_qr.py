# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:07:10 2024

@author: Michal Trojanowski
"""

from dff import *
import numpy as np
from matplotlib import pyplot as plt
from ansatz_collocation import indicator

def dets_conds(omega_end, omega_min, omega_max, T, tau):
    L = int(T/tau)
    Ks = [i*L for i in range(1, 11)]
    ret = np.zeros((len(Ks), 5))
    for k in range(len(Ks)):
        K = Ks[k]
        ret[k, 0] = K
        target = indicator(omega_min, omega_max)
        mesh = np.linspace(0, omega_end, num=K)
        beta_vec = np.array(list(map(target, mesh)))                # vector [target(omega_0), ..., target(omega_K-1)] - rhs of equation
        A = np.zeros((K, L))
        for i in range(K):
            A[i,:] = tau * q_omega(mesh[i], tau, L)
        # breakpoint()
        if L==K:
            ret[k, 1] = np.linalg.det(A)
            ret[k, 2] = np.linalg.cond(A)
            ret[k, 3] = ret[k, 1]
            ret[k, 4] = ret[k, 2]
        else:               
            # linear minimalisation problem:
            # A = Q  [ R ]
            #        [ 0 ]
            # => 
            # Q* b = [ c ]
            #        [ d ]
            # =>
            # solve R alpha = c
            Q, R = np.linalg.qr(A, mode='complete')
            R = R[:L,:]     # upper triangle square matrix
            # breakpoint()
            ret[k, 1] = np.linalg.det(R)
            ret[k, 2] = np.linalg.cond(R)
            ATA = A.transpose()@A
            # breakpoint()
            ret[k, 3] = np.linalg.det(ATA)
            ret[k, 4] = np.linalg.cond(ATA)
    return ret
    
    
if __name__ == "__main__":
    omega_min, omega_max = 10, 12
    omega_end = 360
    T = 1
    tau =  1/omega_end
    dc = dets_conds(omega_end, omega_min, omega_max, T, tau)
    
    fig, ax = plt.subplots()
    plt.semilogy(dc[:,0], dc[:,2], label = "QR decomposition")
    plt.semilogy(dc[:,0], dc[:,4], label = "Gauss")
    plt.xlabel("K")
    plt.ylabel("cond")
    plt.legend()
    plt.title(f"({omega_min}, {omega_max}), T = {T}, L = {int(T/tau)}")
    plt.show()
    
    plt.plot(dc[:,0], dc[:,1], label="QR decomposition")
    plt.plot(dc[:,0], dc[:,3], label="Gauss")
    plt.legend()
    plt.xlabel("K")
    plt.title(f"({omega_min}, {omega_max}), T = {T}, L = {int(T/tau)}")
    plt.ylabel("det")
    plt.show()