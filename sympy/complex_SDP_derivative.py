#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 18:49:33 2022

@author: pv
"""

from sympy.interactive.printing import init_printing
from sympy import symbols, diff, sqrt, pycode

# %%

init_printing(scale=1)

k0, nr, ni, x, s = symbols("K0 nr ni x sdp_sign")

ni = 0

# %%

y = (k0 * nr * x) / ((k0 * nr)**2 + x**2) * (
    k0 * ni + s * sqrt(x**2 + k0**2 * (nr**2 + ni**2)))

# %%

y
# %%

yp = diff(y, x)

# %%

yp

# %%

pycode(yp)
