#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 12:54:58 2025

@author: pv
"""
Z_left = K_left * np.sin(np.arccos(Beta / K_left))
Z_right = K_right * np.sin(np.arccos(Beta / K_right))


def transfer_matrix(self, Input, Rho, n_left, n_right, field_type='TE',
                     Ztype_left='standard', Ztype_right='standard',
                     direction='LR', plane='Z'):
     """Matrix giving coefficients of field in next layer from previous.

        This version takes scaled beta inputs and forms Z_left and Z_right
        from them.  This allows one to choose between equivalent forms for
        Z which satisfy Z^2 = K^2 - Beta^2, which can move branch cuts in the
        determinant and help visualize things in the complex beta plane.

        However, as integration for propagating the radiation plane takes place
        over the positive Z axis, this version may eventually be deprecated or
        modified to only take Beta as input when forced by user via keyword
        argument."""

      K0 = self.K0
       try:
            len(Input)
            Input = np.array(Input, dtype=complex)
        except ValueError:
            Input = np.array([Input], dtype=complex)

        if plane == 'Beta':
            Beta = Input
            if Ztype_left == 'standard':
                Z_left = np.sqrt(K0**2 * n_left**2 - Beta**2,
                                 dtype=complex)
            elif Ztype_left == 'imag':
                Z_left = 1j * np.sqrt(Beta**2 - K0**2 * n_left**2,
                                      dtype=complex)
            else:
                raise ValueError('Ztype must be standard or imag.')

            if Ztype_right == 'standard':
                Z_right = np.sqrt(K0**2 * n_right**2 - Beta**2,
                                  dtype=complex)
            elif Ztype_right == 'imag':
                Z_right = 1j * np.sqrt(Beta**2 - K0**2 * n_right**2,
                                       dtype=complex)
            else:
                raise ValueError('Ztype must be standard or imag.')

        elif plane == 'Z':
            Z0 = Input
            Z_left = self.Zi_from_Z0(Z0, n_left)
            Z_right = self.Zi_from_Z0(Z0, n_right)

        elif plane == 'Psi':
            Psi0 = Input
            Z0 = K0 * self.n0 * np.sin(Psi0)
            Z_left = self.Zi_from_Z0(Z0, n_left)
            Z_right = self.Zi_from_Z0(Z0, n_right)
        else:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        if direction not in ['RL', 'LR']:
            raise ValueError('Direction must be RL or LR.')

        # swap indices to go other direction
        # Note n_left and n_right should still be provided to function in LR
        # direction.
        if direction == 'RL':
            n_right, n_left = n_left, n_right

        M = np.zeros(Input.shape + (2, 2), dtype=np.complex128)

        Exp_minus = 1j * (Z_right - Z_left) * Rho
        Exp_plus = 1j * (Z_right + Z_left) * Rho

        Ymat = np.zeros_like(M)
        Y = 1 / Z_right
        Ymat[..., 0, :] = np.array([Y.T, Y.T]).T
        Ymat[..., 1, :] = np.array([Y.T, Y.T]).T

        if field_type == 'TM':
            Ymat = 1 / (2 * n_left**2) * Ymat
            A_minus = Z_right * n_left**2 - Z_left * n_right**2
            A_plus = Z_right * n_left**2 + Z_left * n_right**2

        elif field_type == 'TE':
            Ymat = 1 / 2 * Ymat
            A_minus = Z_right - Z_left
            A_plus = Z_right + Z_left
        else:
            raise ValueError('Field type must be TE or TM.')

        M[..., 0, :] = np.array([(np.exp(-Exp_minus) * A_plus).T,
                                 (np.exp(-Exp_plus) * A_minus).T]).T

        M[..., 1, :] = np.array([(np.exp(Exp_plus) * A_minus).T,
                                 (np.exp(Exp_minus) * A_plus).T]).T
        return Ymat * M
