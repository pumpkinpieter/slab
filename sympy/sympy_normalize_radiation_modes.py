#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 18:49:33 2022

@author: pv
"""

from sympy.interactive.printing import init_printing
from sympy.matrices import Matrix
from sympy import symbols, cos, sin, diff

# %%

init_printing(scale=1.2)

x, Z, Zp, T = symbols("x Z Z' T")

a0, b0, an, bn = symbols("a_0 b_0 a_n b_n")

a0p, b0p, anp, bnp = symbols("a'_0 b'_0 a'_n b'_n")

M = Matrix([['M00', 'M01'], ['M10', 'M11']])

# %%

E1L = a0 * cos(Z*x) + b0 * sin(Z*x)
E2L = a0p * cos(Zp*x) + b0p * sin(Zp*x)

E1R = an * cos(Z*x) + bn * sin(Z*x)
E2R = anp * cos(Zp*x) + bnp * sin(Zp*x)

# %%

upper = (E1R * diff(E2R, x) - diff(E1R, x) * E2R).subs(x, T)
lower = (E1L * diff(E2L, x) - diff(E1L, x) * E2L).subs(x, -T)

exp = upper - lower
exp = exp.simplify()


# %%

C1R = M * Matrix([a0, b0])
C2R = M * Matrix([a0p, b0p])

exp = exp.subs([(an, C1R[0]), (bn, C1R[1])])
exp = exp.subs([(anp, C2R[0]), (bnp, C2R[1])])

exp = exp.simplify()
exp

# %%
(2*sin(x)*cos(x)).simplify()
