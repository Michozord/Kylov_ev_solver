# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:35:28 2024

@author: Michal Trojanowski
"""

from dff import *
import numpy as np
from matplotlib import pyplot as plt


def alpha(t):
    return 1 if t>1 and t<2 else 0


start = 0
end = 20
Ts = [np.pi, np.pi*10]
tau = 0.0025
fig, ax = plt.subplots()
for T in Ts:
    plot_beta(start, end, alpha, tau, int(T/tau), ax, "T="+str(T), absolute=False)  
ax.grid()
plt.xlim(start, end)
#plt.ylim(-0.5, 1.5)
plt.legend()