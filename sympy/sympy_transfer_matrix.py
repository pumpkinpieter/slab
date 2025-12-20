#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 18:49:33 2022

@author: pv
"""

from sympy.interactive.printing import init_printing
from sympy.matrices import Matrix
from sympy.printing import latex
from sympy import symbols, exp, I, pi, Symbol

# %%

init_printing(scale=1.3)


omega, mu0 = symbols("omega mu_0")

beta, k0, n_l, n_r = symbols("beta k_0 n_l n_r")

# %%


def StateMatrix_TM(rho_idx, z_idx, negate_rho=False):

    rho = Symbol("rho_" + str(rho_idx))
    if negate_rho:
        rho = -rho

    Z = Symbol("Z_" + str(z_idx))
    eps = Symbol("n_" + str(z_idx))**2

    eps_frac = Z / (omega * eps)

    M = Matrix([[exp(I * Z * rho), exp(-I * Z * rho)],
                [-eps_frac * exp(I * Z * rho), eps_frac * exp(-I * Z * rho)]])
    return M


def StateMatrix_TE(rho_idx, z_idx, negate_rho=False):

    rho = Symbol("rho_" + str(rho_idx))
    if negate_rho:
        rho = -rho

    Z = Symbol("Z_" + str(z_idx))
    eps = Symbol("epsilon_" + str(z_idx))

    mu_frac = Z / (omega * mu0)

    M = Matrix([[exp(I * Z * rho), exp(-I * Z * rho)],
                [mu_frac * exp(I * Z * rho), -mu_frac * exp(-I * Z * rho)]])
    return M


# %%

ML_TE = StateMatrix_TE('j', 'j-1')
MR_TE = StateMatrix_TE('j', 'j')

T_TE = MR_TE.inv() @ ML_TE

T_TE.simplify()

# %%

T_TE

# %%

ML_TM = StateMatrix_TM('j', 'j-1')
MR_TM = StateMatrix_TM('j', 'j')

T_TM = ML_TM.inv() @ MR_TM

T_TM.simplify()

# %%

T_TM

# %%

detTE = T_TE.det()
detTE = detTE.simplify()
detTE

# %%


detTM = T_TM.det()
detTM = detTM.simplify()
detTM

# %%

ML_TE = StateMatrix_TE('i', 'i')
MR_TE = StateMatrix_TE('i', 'i+1')

T = MR_TE.inv() @ ML_TE

T.simplify()

# %%

T

# %%

tdet = T.det()

tdet = tdet.simplify()
print(tdet)
tdet2 = tdet**2

# %%

tdet.subs(symbols("Z_i"), k0**2 * n_l**2 - beta**2)

# %%


def TransferMatrix_TM(rho_idx, z_idx_left, z_idx_right, negate_rho=False):

    rho = Symbol("rho_" + str(rho_idx))
    if negate_rho:
        rho = -rho
    n_l = Symbol("n_" + str(z_idx_left))
    n_r = Symbol("n_" + str(z_idx_right))
    Z_l = Symbol("Z_" + str(z_idx_left))
    Z_r = Symbol("Z_" + str(z_idx_right))

    Exp_minus = I * (Z_r - Z_l) * rho
    Exp_plus = I * (Z_r + Z_l) * rho

    A_minus = Z_r * n_l**2 - Z_l * n_r**2
    A_plus = Z_r * n_l**2 + Z_l * n_r**2

    M = Matrix([[exp(-Exp_minus) * A_plus, exp(-Exp_plus) * A_minus],
                [exp(Exp_plus) * A_minus, exp(Exp_minus) * A_plus]])
    return M * 1 / (2 * Z_r * n_l**2)


def TransferMatrix_TE(rho_idx, z_idx_left, z_idx_right, negate_rho=False):

    rho = Symbol("rho_" + str(rho_idx))
    if negate_rho:
        rho = -rho
    Z_l = Symbol("Z_" + str(z_idx_left))
    Z_r = Symbol("Z_" + str(z_idx_right))

    Exp_minus = I * (Z_r - Z_l) * rho
    Exp_plus = I * (Z_r + Z_l) * rho

    A_minus = Z_r - Z_l
    A_plus = Z_r + Z_l

    M = Matrix([[exp(-Exp_minus) * A_plus, exp(-Exp_plus) * A_minus],
                [exp(Exp_plus) * A_minus, exp(Exp_minus) * A_plus]])

    return M * 1 / (2 * Z_r)

# %%


T2 = TransferMatrix_TE('i', 'i', 'i+1')
T2

# %%

# Find eigenvalue equation for TM modes with interface symmetric about origin

T1 = TransferMatrix_TM('', 0, 1, negate_rho=True)
T2 = TransferMatrix_TM('', 1, 0)

M = T2 * T1
M.simplify()
M

# %%

M[1, 1].simplify()

# %%

print(latex(M[1, 1]))

# %%

M[0, 0].simplify()

# %%

print(latex(M[0, 0]))


# %%

# Find eigenvalue equation for TE modes with interface symmetric about origin

T1 = TransferMatrix_TE('', 0, 1, negate_rho=True)
T2 = TransferMatrix_TE('', 1, 0)

M = T2 * T1
M.simplify()
M

# %%

M[1, 1].simplify()

# %%

print(latex(M[1, 1]))

# %%

M[0, 0].simplify()

# %%

print(latex(M[0, 0]))


# %%

# Find eigenvalue equation for TE modes with interface symmetric about origin

T1 = TransferMatrix_TE('2', 0, 1, negate_rho=True)
T2 = TransferMatrix_TE('1', 1, 0, negate_rho=True)
T3 = TransferMatrix_TE('1', 0, 1)
T4 = TransferMatrix_TE('2', 1, 0)

M = T4 * T3 * T2 * T1
M.simplify()
M

# %%

M[1, 1].simplify()

# %%

print(latex(M[1, 1]))

# %%

M[0, 0].simplify()

# %%

print(latex(M[0, 0]))


# %%
