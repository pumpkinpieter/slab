#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 13:46:22 2025

@author: pv
"""

import numpy as np


class Basic():
    def __init__(self, Zs):
        self.Zs = Zs

    def __add__(self, other):
        if self.Zs[-1] == other.Zs[0]:
            idx = slice(1, None)
        else:
            idx = slice(None, None)
        both_Zs = np.concatenate((self.Zs, other.Zs[idx]))
        return Basic(both_Zs)


Z1 = np.linspace(0, 1, 3)
Z2 = np.linspace(1, 20, 5)

A = Basic(Z1)
B = Basic(Z2)
C = A + B
print(A.Zs, B.Zs, C.Zs)
