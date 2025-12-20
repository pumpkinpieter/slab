#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 18:49:33 2022

@author: pv
"""

from sympy.interactive.printing import init_printing
from sympy.matrices import Matrix
from sympy import symbols

# %%

init_printing(scale=1.3)

a, b, c, d = symbols("a b c a")

M = Matrix([[a, b], [c, a]])
J = Matrix([[0, 1], [-1, 0]])

# %%

S = (M.T @ J.T @ M.T @ M @ J @ M)

S.simplify()
# %%
S

# %%

M2 = M.T @ M
M2.simplify()

# %%
M2
