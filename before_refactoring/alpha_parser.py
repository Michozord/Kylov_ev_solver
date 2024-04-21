# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:13:56 2024

@author: Michal Trojanowski
"""

import numpy as np

file_path = r"C:\Users\micha\Documents\Studia\Bachelorarbeit\alphas.txt"
  

def write_to_file(title: str, label: str, alpha: np.array, tau: float, L: int):
    with open(file_path, "a") as file:
        file.write(f"title: {title}; label: {label}; tau: {tau}; L: {L}\n")
        for i in alpha:
            file.write(str(i) + " ")
        file.write("\n" + "-"*10 + "\n")
    return 


def read_file(title: str, label: str):
    with open(file_path, "r") as file:
        lines = file.readlines()
        i = 0
        while not lines[i].startswith(f"title: {title}; label: {label};"):
            i += 3
            if i >= len(lines):
                raise AttributeError(f"Given title {title} and label {label} are not in the file!")
        header = lines[i].split(";")
        tau = float(header[2].removeprefix(" tau: "))
        L = int(header[3].removeprefix(" L: "))
        alpha = np.fromstring(lines[i+1], sep=" ", dtype = float)
    
    return alpha, tau, L 
