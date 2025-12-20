#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 20:40:22 2021.

@author: pv

Class for modeling fibers with 1D transverse plane.
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from warnings import warn
from celluloid import Camera
from scipy import integrate
from collections.abc import Iterable

from ipywidgets import interactive, FloatSlider, Layout


class SlabExact():
    """Class to model layered slab waveguides."""

    def __init__(self, scale=1e-6, Ts=[2, 2, 2], ns=[1.3, 1.5, 1.3], wl=1.8e-6,
                 xrefs=None, no_xs=False, Shift=0, symmetric=False):
        self.no_xs = no_xs

        # Check inputs for errors
        self.check_parameters(Ts, ns, xrefs)

        self.Shift = Shift
        self.symmetric = symmetric
        self.scale = scale
        self.L = scale
        self.shift = Shift * scale
        if xrefs is None:
            xrefs = np.array(np.array(Ts)*30, dtype=int)+1
        self.xrefs = xrefs

        self.Ts = Ts  # Property that also sets "geometry and mesh"

        self.ns_in = deepcopy(ns)  # Copy input refractive index information

        self.wl = wl  # Property that also sets refractive indices
        self.radiation_mode_normalization_methods = \
            "'paper', 'ours', and 'eigvec'."

    @property
    def wl(self):
        """Get wavelength."""
        return self._wl

    @wl.setter
    def wl(self, wl):
        """ Set wavelength and associated material parameters."""

        self._wl = wl
        N = len(self.ns_in)

        # Below allows for callable wavelength functions for implementing
        # dispersion, either via Sellmeier or opticalmaterials.py
        self.ns = np.array([self.ns_in[i](wl)
                            if callable(self.ns_in[i])
                            else self.ns_in[i] for i in range(N)])

        self.k0 = 2 * np.pi / wl
        self.K0 = self.k0 * self.scale

        self.n0 = np.max(self.ns[[0, -1]])  # n for integral expansion
        self.n_low = np.min(self.ns)
        self.n_high = np.max(self.ns)

        self.ks = self.k0 * self.ns
        self.Ks = self.K0 * self.ns
        self.K_low, self.K_high = np.min(self.Ks), np.max(self.Ks)

        # Z value where radiation modes have hyperbolic parts (0 if n_low = n0)
        self.Z_hyperbolic = self.K0 * np.sqrt(self.n0**2 - self.n_low**2)

        # Z value where radiation modes shift from propagating to evanescent
        self.Z_evanescent = self.Zi_from_Beta(0).real

        # Z value before which we have one-sided radiation modes.
        self.Z_one_sided = self.K0 * np.sqrt(np.max(self.ns[[0, -1]])**2
                                             - np.min(self.ns[[0, -1]])**2)

    @property
    def Ts(self):
        """Get ts."""
        return self._Ts

    @Ts.setter
    def Ts(self, Ts):
        """ Set interface locations and domain arrays."""
        Ts = np.array(Ts, dtype=float)
        self._Ts = Ts
        self.ts = Ts * self.scale
        self.Rhos = np.array([sum(Ts[:i]) for i in range(len(Ts)+1)])
        if self.symmetric:
            self.Shift = 1/2 * self.Rhos[-1]
        self.Rhos -= self.Shift
        self.rhos = self.Rhos * self.scale

        if not self.no_xs:
            self.Xs = [np.linspace(self.Rhos[i], self.Rhos[i+1], self.xrefs[i])
                       for i in range(len(self.Rhos)-1)]
            self.all_Xs = np.concatenate(
                [self.Xs[i][:-1] for i in range(len(self.Xs))])
            self.all_Xs = np.append(self.all_Xs, self.Xs[-1][-1])

    def check_parameters(self, ts, ns, xrefs):

        # Check that all relevant inputs have same length
        lengths = [len(ts), len(ns)]
        names = ['ts', 'ns']

        if complex(ns[-1]).real > complex(ns[0]).real:
            raise ValueError('If outer regions have different refractive \
indices, arrange waveguide so left most region has the higher index.')

        if xrefs is not None:
            lengths.append(len(xrefs))
            names.append('xrefs')

        lengths = np.array(lengths)

        same = all(x == lengths[0] for x in lengths)

        if not same:
            string = "Provided parameters not of same length: \n\n"
            for name, length in zip(names, lengths):
                string += name + ': ' + str(length) + '\n'
            raise ValueError(string + "\nModify above inputs as necessary and \
try again.")

    def condition_list(self, xs):
        '''Return condition list for piecewise function definitions.'''
        try:
            len(xs)
            xs = np.array(xs)
        except TypeError:
            xs = np.array([xs])

        # Set up piecewise condition list
        conds = [(self.Rhos[i] <= xs)*(xs < self.Rhos[i+1])
                 for i in range(len(self.Rhos)-1)]

        # Extend outer region conditions to +- infinity
        conds[0] = (xs < self.Rhos[1])
        conds[-1] = (self.Rhos[-2] <= xs)

        if len(self.ns) == 1:  # Only one entry, no restrictions wanted
            conds[0] = (-np.inf <= xs)*(xs <= np.inf)
        return conds

    def Beta_from_Zi(self, Z, ni=None):
        """Find non-dimensional Beta from non-dimensional Z."""
        if ni is None:
            ni = self.n0
        return np.sqrt(self.K0**2 * ni**2 - Z**2, dtype=complex)

    def Zi_from_Beta(self, Beta, ni=None):
        """Get scaled Z from scaled Beta."""
        if ni is None:
            ni = self.n0
        return np.sqrt(self.K0**2 * ni**2 - Beta**2, dtype=complex)

    def Zi_from_Z0(self, Z0, ni):
        '''Get Z on region i (with refractive index ni) from Z0. Vectorized
        in Z0 and ni'''
        # if ni == self.n0:
        #     return Z0
        Z0, ni = np.array(Z0), np.array(ni)
        return (np.sign(Z0[..., np.newaxis]) *
                np.sqrt((ni[np.newaxis]**2 - self.n0**2) * self.K0**2 +
                        Z0[..., np.newaxis]**2, dtype=complex))[..., 0]

    def Psi_i_from_Psi0(self, Psi0, ni):
        '''Get Psi on region i (with refractive index ni) from Psi0.'''
        if ni == self.n0:
            return Psi0
        return np.arccos(self.n0 / ni * np.cos(Psi0))

    def region_index(self, xs):
        '''Return region indices in which (non-dimensional) xs lie.  Indexing
        is based on left-continuity, so if x is in (rho_i, rho_i+1) it gets
        index i.  Points outside of domain on left get index 0, on right get
        index len(Rhos)-1.  Vectorized.'''
        try:
            len(xs)
            xs = np.array(xs)
        except TypeError:
            xs = np.array([xs])
        idx = np.zeros_like(xs, dtype=int)

        idx[np.where(xs <= self.Rhos[0])] = 0
        for j in range(len(self.Rhos)-1):
            idx[np.where((xs <= self.Rhos[j+1]) * (xs > self.Rhos[j]))] = j
        idx[np.where((xs > self.Rhos[-1]))] = j

        if len(xs) == 1:
            return idx[0]
        return idx

    def ns_from_xs(self, xs):
        '''Return refractive indices from regions in which (non-dimensional) xs
        lie.'''
        return self.ns[self.region_index(xs)]

    def sdp(self, s, sdp_sign=-1, plane='Z'):
        '''Return points along steepest descent path parameterized by 's'.'''

        if sdp_sign not in [1, -1]:
            raise ValueError('Sign must be integer equal to 1 or -1.')

        if plane not in ['Z', 'Beta', 'Psi']:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        K0 = self.K0
        nr, ni = self.n0.real, self.n0.imag

        if plane == 'Z':
            f = K0 * nr * s / ((K0 * nr)**2 + s**2)
            g = np.sqrt(s**2 + K0**2 * (nr**2 + ni**2))
            return s + 1j * f * (K0 * ni + sdp_sign * g)
        elif plane == 'Beta':
            if ni != 0:
                warn('Beta plane sdp not verified for complex n.')
            return K0 * self.n0 + 1j * s
        else:
            if ni != 0:
                warn('Psi plane sdp not verified for complex n.')
            return sdp_sign * np.sign(s) * np.arccos(1 / np.cosh(s)) + 1j * s

    def sdp_derivative(self, s, sdp_sign=-1, plane='Z'):
        '''Return imag part of dZdx along steepest descent path.'''

        if sdp_sign not in [1, -1]:
            raise ValueError('Sign must be integer equal to 1 or -1.')

        if plane not in ['Z', 'Beta', 'Psi']:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        K0 = self.K0
        nr, ni = self.n0.real, self.n0.imag

        if plane == 'Z':
            A = K0 * nr * sdp_sign * s**2 / \
                ((K0**2*nr**2 + s**2) * np.sqrt(K0**2 * (ni**2 + nr**2) + s**2))

            B = 2 * K0 * nr * s**2 * \
                (K0*ni + sdp_sign * np.sqrt(K0**2 * (ni**2 + nr**2) + s**2)) / \
                (K0**2 * nr**2 + s**2)**2

            C = K0 * nr * (K0 * ni + sdp_sign * np.sqrt(K0**2 *
                           (ni**2 + nr**2) + s**2))/(K0**2 * nr**2 + s**2)
            return 1 + 1j * (A - B + C)

        elif plane == 'Beta':
            raise NotImplementedError('Beta plane sdp derivartive not yet \
implemented.')

        else:
            raise NotImplementedError('Psi plane sdp derivative not yet \
implemented.')

# ------------ Matrices and Determinant (Eigenvalue) Functions -----------------

    def transfer_matrix(self, C, Rho, n_left, n_right, field_type='TE',
                        Ztype_left='standard', Ztype_right='standard',
                        plane='Z', derivate=False):
        """Matrix giving coefficients of field in next layer from previous.

        Takes non-dimensional complex inputs (C) and forms Z_left and Z_right
        from them.  This allows one to choose between equivalent forms for
        Z which satisfy Z^2 = K^2 - C^2, which can move branch cuts in the
        determinant and help visualize things in the complex Beta plane."""

        if field_type not in ['TE', 'TM']:
            raise ValueError('Must have field_type either TE or TM.')

        if plane not in ['Z', 'Beta', 'Psi']:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        # if derivate and plane != 'Z':
        #     raise NotImplementedError("Derivative only for Z plane.")

        try:
            len(C)
            C = np.array(C, dtype=complex)
        except TypeError:
            C = np.array([C], dtype=complex)

        # derivative_scaling = 1

        if plane == 'Beta':
            Beta = C
            if Ztype_left == 'standard':
                Z_left = np.sqrt(self.K0**2 * n_left**2 - Beta**2,
                                 dtype=complex)
            elif Ztype_left == 'imag':
                Z_left = 1j * np.sqrt(Beta**2 - self.K0**2 * n_left**2,
                                      dtype=complex)
            else:
                raise ValueError('Ztype must be standard or imag.')

            if Ztype_right == 'standard':
                Z_right = np.sqrt(self.K0**2 * n_right**2 - Beta**2,
                                  dtype=complex)
            elif Ztype_right == 'imag':
                Z_right = 1j * np.sqrt(Beta**2 - self.K0**2 * n_right**2,
                                       dtype=complex)
            else:
                raise ValueError('Ztype must be standard or imag.')

            if derivate:
                Z0 = self.Zi_from_Beta(Beta, ni=self.n0)
            #     derivative_scaling = -Beta / Z0

        elif plane == 'Z':
            Z0 = C
            Z_left = self.Zi_from_Z0(C, n_left)
            Z_right = self.Zi_from_Z0(C, n_right)

        else:
            Psi0 = C
            Z0 = self.K0 * self.n0 * np.sin(Psi0)
            # Z_left = self.Zi_from_Z0(Z0, n_left)
            # Z_right = self.Zi_from_Z0(Z0, n_right)
            Psi_left = self.Psi_i_from_Psi0(Psi0, n_left)
            Psi_right = self.Psi_i_from_Psi0(Psi0, n_right)
            Z_left = self.K0 * n_left * np.sin(Psi_left)
            Z_right = self.K0 * n_right * np.sin(Psi_right)

            # if derivate:
            #     derivative_scaling = self.K0 * self.n0 * np.cos(Psi0)

        M = np.zeros(C.shape + (2, 2), dtype=np.complex128)

        Exp_minus = 1j * (Z_right - Z_left) * Rho
        Exp_plus = 1j * (Z_right + Z_left) * Rho
        front_term = 1 / Z_right

        if field_type == 'TM':
            front_term *= 1 / (2 * n_left**2)
            A_minus = Z_right * n_left**2 - Z_left * n_right**2
            A_plus = Z_right * n_left**2 + Z_left * n_right**2

        elif field_type == 'TE':
            front_term *= 1 / 2
            A_minus = Z_right - Z_left
            A_plus = Z_right + Z_left
        else:
            raise ValueError('Field type must be TE or TM.')

        M[..., 0, 0] = front_term * np.exp(-Exp_minus) * A_plus
        M[..., 0, 1] = front_term * np.exp(-Exp_plus) * A_minus

        M[..., 1, 0] = front_term * np.exp(Exp_plus) * A_minus
        M[..., 1, 1] = front_term * np.exp(Exp_minus) * A_plus

        if derivate:
            D = np.zeros_like(M)
            front_term = Z0 / (Z_right**2 * Z_left)
            D[..., 0, 0] = front_term * A_minus * (1 + 1j * Z_right * Rho)
            D[..., 0, 1] = -front_term * A_plus * (1 + 1j * Z_right * Rho)
            D[..., 1, 0] = -front_term * A_plus * (1 - 1j * Z_right * Rho)
            D[..., 1, 1] = front_term * A_minus * (1 - 1j * Z_right * Rho)
            M = M * D

        return M

    def transmission_matrix(self, C, field_type='TE', Ztype_far_left='imag',
                            Ztype_far_right='imag', up_to_region=-1, plane='Z',
                            derivate=False):
        """Total product of TE transfer matrices."""

        try:
            len(C)
            C = np.array(C, dtype=complex)
        except TypeError:
            C = np.array([C], dtype=complex)

        T = np.zeros(C.shape + (2, 2), dtype=complex)
        T[..., :, :] = np.eye(2, dtype=complex)

        Rhos = self.Rhos
        ns = self.ns

        if up_to_region >= 0:
            up_to_region = up_to_region - len(Rhos) + 1

        enum = range(1, len(Rhos)+up_to_region)

        if not derivate:
            for i in enum:

                nl, nr = ns[i-1], ns[i]
                rho = Rhos[i]

                if i == 1:
                    L, R = Ztype_far_left, 'standard'

                elif i == len(Rhos) - 2:
                    L, R = 'standard', Ztype_far_right

                else:
                    L, R = 'standard', 'standard'

                T = self.transfer_matrix(C, rho, nl, nr,
                                         field_type=field_type,
                                         Ztype_left=L, Ztype_right=R,
                                         plane=plane) @ T
            return T
        else:
            S = np.zeros_like(T)
            for i in enum:

                T = np.zeros(C.shape + (2, 2), dtype=complex)
                T[..., :, :] = np.eye(2, dtype=complex)

                for j in enum:
                    if j == i:
                        derivate = True
                    else:
                        derivate = False
                    nl, nr = ns[j-1], ns[j]
                    rho = Rhos[j]

                    if j == 1:
                        L, R = Ztype_far_left, 'standard'

                    elif j == len(Rhos) - 2:
                        L, R = 'standard', Ztype_far_right

                    else:
                        L, R = 'standard', 'standard'

                    T = self.transfer_matrix(C, rho, nl, nr,
                                             field_type=field_type,
                                             Ztype_left=L, Ztype_right=R,
                                             plane=plane,
                                             derivate=derivate) @ T
                S += T
            return S

    def determinant(self, C, plane='Z', mode_type='guided', field_type='TE',
                    Normalizer=None, sign=1, derivate=False):
        """Eigenvalue function (formerly determinant of matching matrix, hence
        nomenclature)."""

        if field_type not in ['TE', 'TM']:
            raise ValueError('Must have field_type either TE or TM.')

        if plane not in ['Z', 'Beta', 'Psi']:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        if mode_type not in ['guided', 'leaky', 'radiation']:
            raise ValueError("Mode type must be 'guided', 'leaky' or \
'radiation'.")

        if mode_type == 'guided':
            M = self.transmission_matrix(C, plane=plane, field_type=field_type,
                                         derivate=derivate)
            return M[..., 1, 1]

        elif mode_type == 'leaky':
            M = self.transmission_matrix(C, plane=plane, field_type=field_type,
                                         derivate=derivate)
            return M[..., 0, 0]

        else:
            if Normalizer is None:
                Normalizer = self.normalizer('ours')
            return Normalizer.pole_locations(C, sign=sign, ft=field_type,
                                             plane=plane)

    def transmission_determinant(self, C, field_type='TE', up_to_region=-1,
                                 plane='Z'):
        '''Determinant of transmission matrix.'''
        try:
            len(C)
            C = np.array(C, dtype=complex)
        except TypeError:
            C = np.array([C], dtype=complex)

        if plane not in ['Z', 'Beta', 'Psi']:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        if plane == 'Beta':
            Z0 = self.Zi_from_Beta(C, ni=self.ns[0])
            Zd = self.Zi_from_Beta(C, ni=self.ns[up_to_region])
        elif plane == 'Z':
            Z0 = self.Zi_from_Z0(C, ni=self.ns[0])
            Zd = self.Zi_from_Z0(C, ni=self.ns[up_to_region])
        else:
            Psi0 = C
            Psi_d = self.Psi_i_from_Psi0(Psi0, self.ns[up_to_region])
            Z0 = self.n0 * np.sin(Psi0)
            Zd = self.ns[up_to_region] * np.sin(Psi_d)

        base = Z0 / Zd
        if field_type == 'TM':
            base *= self.ns[up_to_region]**2 / self.ns[0]**2
        return base

# ---------- Building Fields from Propagation Constants in Beta Plane ---------

    def coefficients(self, C, Normalizer=None, field_type='TE',
                     mode_type='guided', plane='Z', sign='+1',
                     up_to_region=-1, rounding=12, manual_coeffs=None):
        """Return field coefficients given propagation constant Beta."""

        if field_type not in ['TE', 'TM']:
            raise ValueError('Must have field_type either TE or TM.')

        if plane not in ['Z', 'Beta', 'Psi']:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        if mode_type not in ['guided', 'leaky', 'radiation']:
            raise ValueError("Mode type must be 'guided', 'leaky' or \
'radiation.")

        # Single scalar inputs need to be given at least a single dimension
        try:
            len(C)
            C = np.array(C, dtype=complex)
        except TypeError:
            C = np.array([C], dtype=complex)

        Ztype_far_left = 'standard'
        Ztype_far_right = 'standard'

        if mode_type in ['guided', 'leaky']:
            Ztype_far_left = 'imag'
            Ztype_far_right = 'imag'

        # Set up initial vector array M0.
        # This will contain initial vector for each coefficient array,
        # and also be overwritten to store the new coefficients
        # as we apply transfer matrix.
        M0 = np.zeros(C.shape + (2, 1), dtype=complex)

        if mode_type == 'guided':
            M = self.transmission_matrix(C, field_type=field_type,
                                         Ztype_far_left=Ztype_far_left,
                                         Ztype_far_right=Ztype_far_right,
                                         plane=plane,
                                         derivate=False)

            M_prime = self.transmission_matrix(C, field_type=field_type,
                                               Ztype_far_left=Ztype_far_left,
                                               Ztype_far_right=Ztype_far_right,
                                               plane=plane,
                                               derivate=True)

            d_prime, c = M_prime[..., 1, 1], M[..., 1, 0]
            check_idx = 1
            M0[..., :, 0] = np.array([0, np.sqrt(1j * c / d_prime,
                                                 dtype=complex)[0]])
            if plane == 'Beta':
                M0 *= 1j

        elif mode_type == 'leaky':
            M = self.transmission_matrix(C, field_type=field_type,
                                         Ztype_far_left=Ztype_far_left,
                                         Ztype_far_right=Ztype_far_right,
                                         plane=plane,
                                         derivate=False)

            M_prime = self.transmission_matrix(C, field_type=field_type,
                                               Ztype_far_left=Ztype_far_left,
                                               Ztype_far_right=Ztype_far_right,
                                               plane=plane,
                                               derivate=True)

            a_prime, b = M_prime[..., 0, 0], M[..., 0, 1]
            check_idx = 0
            M0[..., :, 0] = np.array([np.sqrt(1j * b / a_prime,
                                              dtype=complex)[0], 0])
            if plane == 'Beta':
                M0 *= 1j
        else:
            if Normalizer is None:
                Normalizer = self.normalizer('ours')

            M0 = Normalizer.normalization(C, sign=sign, ft=field_type,
                                          plane=plane)
        if manual_coeffs is not None:
            M0[..., :, 0] = manual_coeffs

        Rhos, ns = self.Rhos, self.ns

        if up_to_region >= 0:
            up_to_region = up_to_region - len(Rhos) + 1

        Coeffs = np.zeros(C.shape + (2, len(Rhos)+up_to_region),
                          dtype=complex)

        # set first vectors in each coefficient array
        Coeffs[..., :, 0] = M0[..., :, 0]

        Ztypes = ['standard' for i in range(len(self.Rhos)-1)]
        Ztypes[0], Ztypes[-1] = Ztype_far_left, Ztype_far_right

        for i in range(1, len(Rhos)+up_to_region):
            nl, nr = ns[i-1], ns[i]
            Rho = Rhos[i]
            T = self.transfer_matrix(C, Rho, nl, nr,
                                     field_type=field_type,
                                     Ztype_left=Ztypes[i-1],
                                     Ztype_right=Ztypes[i],
                                     plane=plane)

            M0 = (T @ M0)  # apply T to vectors
            Coeffs[..., :, i] = M0[..., :, 0]  # update coefficient arrays

        # Reduce dimension for length 1 inputs
        if len(C) == 1:
            Coeffs = Coeffs[0]

        # Round to avoid false blowup, noted in guided modes.
        # Fundamental had lower error and rounding=16 worked, but HOMs
        # had more noise and required rounding=12
        Coeffs = np.round(Coeffs, decimals=rounding)

        # Check for correct coefficients if mode type is guided or leaky
        if mode_type in ['guided', 'leaky'] and len(C) == 1:
            if Coeffs.T[-1, check_idx] != 0:
                warn(message='Provided mode type %s, but coefficients in outer \
region do not align with this. User may wish to check supplied \
propagation constant and/or rounding parameter.' % mode_type)

        return Coeffs

    def regional_field(self, C, index, coeffs, Ztype='standard', plane='Z'):
        """Return field on one region of fiber."""

        if plane not in ['Z', 'Beta', 'Psi']:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        A, B = coeffs[:]
        K0 = self.K0
        n = self.ns[index]

        if plane == 'Beta':
            Beta = C
            if Ztype == 'standard':
                Z = np.sqrt(K0**2 * n**2 - C**2, dtype=complex)
            elif Ztype == 'imag':
                Z = 1j * np.sqrt(C**2 - K0**2 * n**2, dtype=complex)
            else:
                raise ValueError('Ztype must be standard or imag.')
        elif plane == 'Z':
            Z = self.Zi_from_Z0(C, ni=n)
            Beta = self.Beta_from_Zi(Z, ni=n)
        else:
            Psi0 = C
            Beta = K0 * self.n0 * np.cos(Psi0)
            Z = K0 * n * np.sin(self.Psi_i_from_Psi0(Psi0, n))

        def F(xs, zs=None):

            try:
                len(xs)
                xs = np.array(xs)
            except TypeError:
                xs = np.array([xs])

            if len(xs.shape) > 1:
                raise ValueError('Please provide single dimension arrays for \
xs and zs to levarage product nature of fields.')

            ys = (A * np.exp(1j * Z * xs) + B * np.exp(-1j * Z * xs))

            if zs is not None:
                try:
                    len(zs)
                    zs = np.array(zs)
                except TypeError:
                    zs = np.array([zs])

                if len(zs.shape) > 1:
                    raise ValueError('Please provide single dimension arrays \
for xs and zs to levarage product nature of fields.')

                ys = np.outer(np.exp(1j * Beta * zs), ys)

            return ys

        return F

    def fields(self, C, Normalizer=None, field_type='TE', mode_type='guided',
               sign='+1', rounding=12, plane='Z', manual_coeffs=None):
        '''Return fields at propagation constant given by 'C' in plane
        determined by 'plane'.'''

        # Give C at least one dimension (for vectorization)
        try:
            len(C)
            C = np.array(C, dtype=complex)
        except TypeError:
            C = np.array([C], dtype=complex)

        if plane not in ['Z', 'Beta', 'Psi']:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        M = self.coefficients(C, Normalizer=Normalizer, mode_type=mode_type,
                              field_type=field_type, sign=sign,
                              rounding=rounding, plane=plane,
                              manual_coeffs=manual_coeffs).T

        Ztypes = ['standard' for i in range(len(self.Rhos)-1)]

        if mode_type in ['guided', 'leaky']:
            Ztypes[0], Ztypes[-1] = 'imag', 'imag'

        if plane == 'Beta':
            Beta = C
        elif plane == 'Z':
            Beta = self.Beta_from_Zi(C)
        else:
            Beta = self.K0 * self.n0 * np.cos(C)

        # Get list of functions, one for each region
        Fs = []
        for i in range(len(self.Rhos)-1):
            Fs.append(self.regional_field(C, i, M[i], Ztype=Ztypes[i],
                                          plane=plane))

        # Return piecewise defined function
        def F(xs, zs=None):

            # Give xs at least one dimension
            try:
                len(xs)
                xs = np.array(xs)
            except TypeError:
                xs = np.array([xs])

            if len(xs.shape) > 1:
                raise ValueError('Please provide single dimension arrays \
for xs and zs to levarage product nature of fields.')

            # Set up piecewise condition list
            conds = self.condition_list(xs)
            ys = np.piecewise(xs+0j, conds, Fs)  # 0j gives complex output

            if zs is not None:
                try:
                    len(zs)
                    zs = np.array(zs)
                except TypeError:
                    zs = np.array([zs])
                if len(zs.shape) > 1:
                    raise ValueError('Please provide single dimension \
arrays for xs and zs to levarage product nature of fields.')
                ys = np.outer(np.exp(1j * Beta * zs), ys)
            return ys

        return F

    def evaluate_fields(self, C, x, z=0, Normalizer=None, field_type='TE',
                        mode_type='radiation', sign='+1', plane='Z'):
        '''Return value of field with propagation constant C at x, z in plane
        determined by 'plane'. Vectorized in C.'''

        if plane not in ['Z', 'Beta', 'Psi']:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        i = self.region_index(x)

        if plane == 'Beta':
            Beta = C
            Zi = np.array(self.Zi_from_Beta(C, ni=self.ns[i]))

        elif plane == 'Z':
            Zi = np.array(self.Zi_from_Z0(C, ni=self.ns[i]))
            Beta = self.Beta_from_Zi(Zi, ni=self.ns[i])

        else:
            Beta = self.K0 * self.n0 * np.cos(C)
            Zi = self.K0 * self.ns[i] * \
                np.sin(self.Psi_i_from_Psi0(C, self.ns[i]))

        Cs = self.coefficients(C, Normalizer=Normalizer,
                               up_to_region=i, mode_type=mode_type,
                               field_type=field_type, sign=sign, plane=plane)

        return (Cs[..., 0, i] * np.exp(1j * Zi * x) +
                Cs[..., 1, i] * np.exp(-1j * Zi * x)) * np.exp(1j * Beta * z)

# ----------------------- Radiation Mode Normalizations -----------------------

    def normalizer(self, method):

        if method == 'paper':

            class PaperMethod():
                def __init__(selfz):
                    pass

                def normalization(selfz, C, sign=1, ft='TE', plane='Z'):

                    if plane not in ['Z', 'Beta', 'Psi']:
                        raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")
                    try:
                        len(C)
                        C = np.array(C)
                    except TypeError:
                        C = np.array([C])

                    M0 = np.zeros(C.shape + (2, 1), dtype=complex)
                    M = self.transmission_matrix(C, field_type=ft, plane=plane,
                                                 Ztype_far_left='standard',
                                                 Ztype_far_right='standard')

                    r1 = -M[..., 1, 0] / M[..., 1, 1]
                    detM = self.transmission_determinant(C, field_type=ft,
                                                         plane=plane)
                    t2 = 1 / (M[..., 1, 1] * detM)
                    frac = (-M[..., 1, 0] * detM) / M[..., 0, 1]

                    b = int(sign) * np.sqrt(frac, dtype=complex)

                    C = np.sqrt(1 / (r1 + b * t2), dtype=complex)

                    M0[..., 0, :] = C[..., np.newaxis]
                    M0[..., 1, :] = C[..., np.newaxis].conjugate()

                    M0 *= 1 / (2 * np.sqrt(np.pi))

                    return M0

                def pole_locations(selfz, C, sign=1, ft='TE', plane='Z'):

                    if plane not in ['Z', 'Beta', 'Psi']:
                        raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

                    M = self.transmission_matrix(C, field_type=ft, plane=plane,
                                                 Ztype_far_left='standard',
                                                 Ztype_far_right='standard')

                    detM = self.transmission_determinant(C, field_type=ft,
                                                         plane=plane)
                    b = np.sqrt((-M[..., 1, 0] * detM) / M[..., 0, 1],
                                dtype=complex)
                    plus = -M[..., 1, 0] + b / detM
                    minus = -M[..., 1, 0] - b / detM
                    return plus * minus / M[..., 1, 1]**2

            return PaperMethod()

        elif method == 'ours':

            class OurMethod():
                def __init__(selfz):
                    pass

                def normalization(selfz, C, sign=1, ft='TE', plane='Z'):

                    if plane not in ['Z', 'Beta', 'Psi']:
                        raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")
                    try:
                        len(C)
                        C = np.array(C)
                    except TypeError:
                        C = np.array([C])

                    M0 = np.zeros(C.shape + (2, 1), dtype=complex)
                    M = self.transmission_matrix(C, field_type=ft, plane=plane,
                                                 Ztype_far_left='standard',
                                                 Ztype_far_right='standard')

                    detM = self.transmission_determinant(C, field_type=ft,
                                                         plane=plane)

                    frac = (-M[..., 1, 0] * detM) / M[..., 0, 1]

                    b = int(sign) * np.sqrt(frac, dtype=complex)

                    C1 = np.sqrt((b.conj() - M[..., 0, 1]) / M[..., 0, 0],
                                 dtype=complex)
                    C2 = np.sqrt(M[..., 0, 0] / (b.conj() - M[..., 0, 1]),
                                 dtype=complex)

                    M0[..., 0, :] = C1[..., np.newaxis]
                    M0[..., 1, :] = C2[..., np.newaxis]

                    M0 *= 1 / (2 * np.sqrt(np.pi))

                    return M0

                def pole_locations(selfz, C, sign=1, ft='TE', plane='Z'):

                    if plane not in ['Z', 'Beta', 'Psi']:
                        raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

                    M = self.transmission_matrix(C, field_type=ft, plane=plane,
                                                 Ztype_far_left='standard',
                                                 Ztype_far_right='standard')

                    detM = self.transmission_determinant(C, field_type=ft,
                                                         plane=plane)

                    b = int(sign) * np.sqrt((-M[..., 1, 0] * detM) /
                                            M[..., 0, 1], dtype=complex)

                    return (b.conj() - M[..., 0, 1]) * M[..., 0, 0]

            return OurMethod()

        elif method == 'eigvec':

            class EigvecMethod():
                def __init__(selfz):
                    pass

                def normalization(selfz, C, sign=1, ft='TE', plane='Z'):

                    if plane not in ['Z', 'Beta', 'Psi']:
                        raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

                    sign = int(sign)
                    if sign not in [-1, 1]:
                        raise ValueError("Sign must be 1 or -1 (int).")

                    try:
                        len(C)
                        C = np.array(C)
                    except TypeError:
                        C = np.array([C])

                    M0 = np.zeros(C.shape + (2, 1), dtype=complex)
                    M = self.transmission_matrix(C, field_type=ft, plane=plane,
                                                 Ztype_far_left='standard',
                                                 Ztype_far_right='standard')

                    trQs = M[..., 0, 0] * M[..., 1, 0] + \
                        M[..., 0, 1] * M[..., 1, 1]

                    L = trQs + sign * np.sqrt(trQs**2 + 4 *
                                              M[..., 0, 0] * M[..., 1, 1],
                                              dtype=complex)

                    C1 = 2 * M[..., 0, 0] * M[..., 1, 1]
                    C2 = L - 2 * M[..., 0, 0] * M[..., 1, 0]

                    factor = np.sqrt(L * (C1 ** 2 + C2 ** 2), dtype=complex)

                    C1 /= factor
                    C2 /= factor

                    M0[..., 0, :] = C1[..., np.newaxis]
                    M0[..., 1, :] = C2[..., np.newaxis]

                    M0 *= np.sqrt(1 / np.pi)

                    return M0

                def pole_locations(selfz, C, sign=1, ft='TE', plane='Z'):

                    if plane not in ['Z', 'Beta', 'Psi']:
                        raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

                    M = self.transmission_matrix(C, field_type=ft, plane=plane,
                                                 Ztype_far_left='standard',
                                                 Ztype_far_right='standard')

                    return - 4 * M[..., 1, 1] * M[..., 0, 0]

            return EigvecMethod()

        else:
            raise ValueError('Given normalization method %s not found. \
Available methods are %s' % (method, self.radiation_mode_normalization_methods))

# --------------------------- Spectral Integration ----------------------------

    def radiation_transform(self, Cs, f0=None, s=None, Lx=None, Rx=None,
                            plane='Z', field_type='TE', sign='+1',
                            Normalizer=None, **intargs):
        '''
        Return transform coefficients (alphas) for input field f0 from Cs using
        scipy vectorized integration techniques.
        '''
        if (f0 is None and s is None) or (f0 is not None and s is not None):
            raise ValueError("Exactly one of either input field 'f0' or source \
point 's' must be provided.")

        delta_transform = f0 is None and s is not None

        if delta_transform:
            return self.evaluate_fields(Cs, s, field_type=field_type,
                                        mode_type='radiation',
                                        sign=sign, plane=plane,
                                        Normalizer=Normalizer)

        else:
            if Lx is None:
                Lx = self.Rhos[0]
            if Rx is None:
                Rx = self.Rhos[-1]

            def integrand(x, C, sign='+1'):
                return f0(x) * self.evaluate_fields(C, x, field_type=field_type,
                                                    mode_type='radiation',
                                                    sign=sign, plane=plane,
                                                    Normalizer=Normalizer)

            return integrate.quad_vec(integrand, Lx, Rx, args=(Cs, sign),
                                      **intargs)[0]

    def spectral_integrand(self, Cs, f0=None, x=0, z=0, s=None,
                           Lx=None, Rx=None, plane='Z',
                           class_A_only=False, class_B_only=False,
                           field_type='TE', alpha_A=None, alpha_B=None,
                           Normalizer=None, det_power=0):
        '''Return integrand of radiation expansion of f0 at Cs, x, z.'''

        if class_A_only and class_B_only:
            raise ValueError("Only one of class_A_only and class_B_only flags \
can be set to True.")
        if Lx is None:
            Lx = self.Rhos[0]
        if Rx is None:
            Rx = self.Rhos[-1]

        if class_A_only:
            if alpha_A is None:
                alpha_A = self.radiation_transform(Cs, f0=f0, s=s, Lx=Lx, Rx=Rx,
                                                   field_type=field_type,
                                                   sign='+1', plane=plane,
                                                   Normalizer=Normalizer)
            field_value = self.evaluate_fields(Cs, x, z=z,
                                               field_type=field_type,
                                               mode_type='radiation',
                                               sign='+1', plane=plane,
                                               Normalizer=Normalizer)
            return alpha_A * field_value

        elif class_B_only:
            if alpha_B is None:
                alpha_B = self.radiation_transform(Cs, f0=f0, s=s, Lx=Lx, Rx=Rx,
                                                   field_type=field_type,
                                                   sign='-1', plane=plane,
                                                   Normalizer=Normalizer)

            field_value = self.evaluate_fields(Cs, x, z=z,
                                               field_type=field_type,
                                               mode_type='radiation',
                                               sign='-1', plane=plane,
                                               Normalizer=Normalizer)
            return alpha_B * field_value

        else:
            if alpha_A is None:
                alpha_A = self.radiation_transform(Cs, f0=f0, s=s, Lx=Lx, Rx=Rx,
                                                   field_type=field_type,
                                                   sign='+1', plane=plane,
                                                   Normalizer=Normalizer)

            if alpha_B is None:
                alpha_B = self.radiation_transform(Cs, f0=f0, s=s, Lx=Lx, Rx=Rx,
                                                   field_type=field_type,
                                                   sign='-1', plane=plane,
                                                   Normalizer=Normalizer)

            field_value_A = self.evaluate_fields(Cs, x, z=z,
                                                 field_type=field_type,
                                                 mode_type='radiation',
                                                 sign='+1', plane=plane,
                                                 Normalizer=Normalizer)

            field_value_B = self.evaluate_fields(Cs, x, z=z,
                                                 field_type=field_type,
                                                 mode_type='radiation',
                                                 sign='-1', plane=plane,
                                                 Normalizer=Normalizer)
            if det_power > 0:
                M = self.transmission_matrix(Cs, plane=plane)
                detQ = -4 * M[..., 0, 0] * M[..., 1, 1]
            else:
                detQ = 1
            return detQ**det_power * \
                (alpha_A * field_value_A + alpha_B * field_value_B)

    def radiation_transform_dZ_approx(self, f0, Lx=None, Rx=None,
                                      Z_base=1e-5, dZ=1e-6,
                                      field_type='TE', sign='+1',
                                      Normalizer=None, **intargs):
        if Lx is None:
            Lx = self.Rhos[0]
        if Rx is None:
            Rx = self.Rhos[-1]

        Fp = self.dFdZ_approx(Z_base=Z_base, dZ=dZ, field_type=field_type,
                              Normalizer=Normalizer, sign=sign)

        def integrand(x):
            return f0(x) * Fp(x)

        return integrate.quad(integrand, Lx, Rx, complex_func=True,
                              **intargs)[0]

    def dFdZ_approx(self, Z_base, dZ=1e-6, Normalizer=None, field_type='TE',
                    sign='+1', order=2, rounding=12):
        '''Return approximate first derivative of radiation mode F with regard
        to propagation constant Z at Z_base.  Ultimately we will do this
        exactly, and then deprecate this and related functions.'''

        if order not in [1, 2]:
            raise ValueError("Order must be 1 or 2.")
        FZ0 = self.fields_Z(Z_base, mode_type='radiation',
                            field_type=field_type,
                            plane='Z',
                            Normalizer=Normalizer,
                            rounding=rounding,
                            sign=sign)

        FdZ1 = self.fields_Z(Z_base + dZ,  mode_type='radiation',
                             field_type=field_type,
                             plane='Z',
                             Normalizer=Normalizer,
                             rounding=rounding,
                             sign=sign)
        if order == 1:
            return lambda x, zs=0: (FdZ1(x, zs=zs) - FZ0(x, zs=zs)) / dZ

        else:
            FdZ2 = self.fields_Z(Z_base + 2 * dZ, mode_type='radiation',
                                 field_type=field_type,
                                 plane='Z',
                                 Normalizer=Normalizer,
                                 rounding=rounding,
                                 sign=sign)

            return lambda x, zs=0: (4 * FdZ1(x, zs=zs) -
                                    1 * FdZ2(x, zs=zs) -
                                    3 * FZ0(x, zs=zs)) / (2*dZ)

# ------------------------------- Space Wave ---------------------------------

    def space_wave_approx(self, f0, Z_base=1e-7, dZ=1e-6, Lx=None, Rx=None,
                          sign=1, field_type='TE', Normalizer=None,
                          **intargs):

        if Lx is None:
            Lx = self.Rhos[0]
        if Rx is None:
            Rx = self.Rhos[-1]

        dC = self.radiation_transform_dZ_approx(f0, Z_base=Z_base, Lx=Lx, Rx=Rx,
                                                field_type=field_type,
                                                Normalizer=Normalizer,
                                                sign=sign, dZ=dZ,
                                                **intargs)

        Kn = self.K0 * self.n0
        C1 = -(1 + 1j) * np.sqrt(np.pi) / 2

        def SpaceWave(x, z):
            if np.any(z == 0):
                raise ValueError('Space wave not defined at z=0.')
            dF = self.dFdZ_approx(Z_base, dZ=dZ, sign=sign,
                                  field_type=field_type,
                                  Normalizer=Normalizer)
            C2 = np.sqrt(Kn / z) ** 3
            z0s = np.zeros_like(z)
            return (C1 * np.exp(1j * Kn * z) * dC * dF(x, z0s).T * C2).T

        return SpaceWave

# ---------------------------- Exact transforms ------------------------------

    def delta_transform(self, C, s=0, field_type='TE', sign='+1',
                        Normalizer=None):
        """Evaluate radiation transform of delta function delta(x-s)."""
        return self.evaluate_fields(C, s, field_type=field_type,
                                    mode_type='radiation', sign=sign,
                                    Normalizer=Normalizer)

    def integrate_interval(self, Z, L, R, field_type='TE', sign='+1',
                           paper_method=False):
        """Integrate radiation field F(x, Z) on x in (a,b)."""
        # Lj = self.region_index(L)
        # Rj = self.region_index(R)
        pass

# --------------------------- Complex Contours -------------------------------

    def real_contour(self, x_start, x_end, N, s_start=0, s_end=1):
        if x_start == x_end:
            raise ValueError('Please provide non-trivial real interval.')
        if N % 2 == 0:
            N += 1
        Ss = np.linspace(s_start, s_end, N)

        Cs = x_start + (x_end - x_start) * (Ss - s_start) / (s_end - s_start)
        dCdS = (x_end - x_start) / (s_end - s_start) * np.ones(N)
        return {'Cs': Cs, 'dCdS': dCdS, 'Ss': Ss, 'contour_type': 'real'}

    def horizontal_contour(self, y, x_start, x_end, N, s_start=0, s_end=1):
        D = self.real_contour(x_start, x_end, N, s_start, s_end)
        D['Cs'] = D['Cs'] + 1j * y
        D['contour_type'] = 'horizontal'
        return D

    def circular_contour(self, center, radius, N, orientation='ccw',
                         phase_shift=0, total_angle=2*np.pi):
        if N % 2 == 0:
            N += 1
        if orientation not in ['cw', 'ccw']:
            raise ValueError("Orientation must be 'cw' or 'ccw'.")
        if orientation == 'ccw':
            sign = 1
        else:
            sign = -1
        Ss = phase_shift + np.linspace(0, sign * total_angle, N)

        Cs = center + radius * np.exp(1j * Ss)
        dCdS = 1j * radius * np.exp(1j * Ss)

        return {'Cs': Cs, 'dCdS': dCdS, 'Ss': Ss, 'contour_type': 'circle'}

    def half_circle_contour(self, center, radius, N, orientation='cw',
                            phase_shift=np.pi/2):

        D = self.circular_contour(center, radius, N, orientation=orientation,
                                  phase_shift=phase_shift, total_angle=np.pi)
        D['contour_type'] = 'half_circle'

        return D

    def sdp_contour(self, x_start, x_end, N, s_start=0, s_end=1, sdp_sign=-1,
                    plane='Z'):
        if x_start == x_end:
            raise ValueError('Please provide non-trivial interval.')
        if N % 2 == 0:
            N += 1
        Ss = np.linspace(s_start, s_end, N)
        xs = x_start + (x_end - x_start) * (Ss - s_start) / (s_end - s_start)
        Cs = self.sdp(xs, sdp_sign=sdp_sign, plane=plane)

        dxds = (x_end - x_start) / (s_end - s_start) * np.ones(N)
        dCdS = dxds * self.sdp_derivative(xs, sdp_sign=sdp_sign, plane=plane)

        return {'Cs': Cs, 'dCdS': dCdS, 'Ss': Ss, 'contour_type': 'sdp'}

    def vertical_contour(self, x, y_start, y_end, N, s_start=0, s_end=1):
        if y_start == y_end:
            raise ValueError('Please provide non-trivial vertical interval.')
        if N % 2 == 0:
            N += 1
        Ss = np.linspace(s_start, s_end, N)
        ys = y_start + (y_end - y_start) * (Ss - s_start) / (s_end - s_start)
        Cs = x + 1j * ys

        dyds = (y_end - y_start) / (s_end - s_start) * np.ones(N)
        dCdS = 1j * dyds

        return {'Cs': Cs, 'dCdS': dCdS, 'Ss': Ss, 'contour_type': 'vertical'}

    def vertical_contour_to_sdp(self, x, N, sdp_sign=-1):
        y_start, y_end = 0, self.sdp(x, sdp_sign=sdp_sign, plane='Z').imag
        return self.vertical_contour(x, y_start, y_end, N)

    def imaginary_contour(self, y_start, y_end, N, s_start=0, s_end=1):
        D = self.vertical_contour(0, y_start, y_end, N, s_start=s_start,
                                  s_end=s_end)
        D['contour_type'] = 'imaginary'
        return D

# ------------------------------ Propagation ----------------------------------

    def propagator(self):
        '''Return class P that propagates input fields.'''

        class Propagator():

            def __init__(selfz, contour, f0=None, exact_transform=None,
                         exact_kwargs={}, Lx=self.Rhos[0], Rx=self.Rhos[-1],
                         field_type='TE', sign='+1', Normalizer=None, plane='Z',
                         **integration_args):

                both = f0 is None and exact_transform is None
                neither = f0 is not None and exact_transform is not None

                if both or neither:
                    raise ValueError('Must provide precisely one of either \
input field f0 or exact transform function.')

                for key, val in contour.items():
                    setattr(selfz, key, val)

                if len(set([len(selfz.Cs), len(selfz.Ss),
                            len(selfz.dCdS)])) != 1:
                    raise ValueError('Length of Cs, dCdS and Ss must be equal.')

                selfz.dS = selfz.Ss[1:] - selfz.Ss[:-1]

                # Set transform function.  Pre-provide extra arguments using
                # lambda functions.
                if f0 is not None:
                    selfz.f0 = f0
                    selfz.Lx, selfz.Rx = Lx, Rx

                    func = (lambda C_lambda:
                            selfz.radiation_transform(C_lambda,
                                                      field_type=field_type,
                                                      Lx=Lx, Rx=Rx,
                                                      sign=sign,
                                                      Normalizer=Normalizer,
                                                      plane=plane,
                                                      **integration_args))

                    selfz.transform = func

                else:
                    exact_func = (lambda C_lambda:
                                  exact_transform(C_lambda, **exact_kwargs))
                    selfz.transform = exact_func

                selfz.sign = sign
                selfz.field_type = field_type

                selfz.alphas = selfz.transform(selfz.Cs)

                # Get radiation modes (Fs) from parent class
                selfz.Fs = []

                for C in selfz.Cs:
                    # get field and append to list
                    F = self.fields(C, mode_type='radiation',
                                    field_type=field_type, sign=sign,
                                    Normalizer=Normalizer,
                                    plane=plane)
                    selfz.Fs.append(F)

            def radiation_transform(selfz, Cs, Lx=self.Rhos[0],
                                    Rx=self.Rhos[-1], field_type='TE',
                                    sign='+1', Normalizer=None, plane='Z',
                                    **intargs):
                '''
                Return transform coefficients (alphas) for input field f0 from
                Cs using scipy vectorized integration techniques.
                '''
                ft = field_type
                mt = 'radiation'
                N = Normalizer

                def integrand(x, C, sign='+1'):
                    return selfz.f0(x) * self.evaluate_fields(C, x,
                                                              field_type=ft,
                                                              mode_type=mt,
                                                              Normalizer=N,
                                                              plane=plane,
                                                              sign=sign)

                return integrate.quad_vec(integrand, Lx, Rx,
                                          args=(Cs, sign), **intargs)[0]

            def propagate(selfz, xs, zs, method='simpsons'):
                '''
                Propagate radiation field of input function selfz.f0 (or exact
                transform).

                Parameters
                ----------
                xs : float or float array
                    Value of x along direction transverse to propagation.
                zs : float or float array
                    Value of z along direction of propagation.
                method: str, optional
                    Quadrature method.  May be left_endpoint, right_endpoint or
                    trapezoidal. Default is trapezoid.

                Returns
                -------
                ys: complex or complex array
                    Field strengths at input xs and ys.
                '''
                Nz = len(selfz.Cs)
                alphas, Fs, dCdS = selfz.alphas, selfz.Fs, selfz.dCdS
                dS = selfz.dS

                if method == 'left_endpoint':
                    ys = sum([alphas[i] * Fs[i](xs, zs) * dCdS[i] *
                              dS[i] for i in range(Nz-1)])

                elif method == 'right_endpoint':
                    ys = sum([alphas[i+1] * Fs[i+1](xs, zs) * dCdS[i+1] *
                              dS[i] for i in range(Nz-1)])

                elif method == 'trapezoid':
                    ys = alphas[0] * Fs[0](xs, zs) * dCdS[0] * dS[0]
                    ys += sum([alphas[i] * Fs[i](xs, zs) * dCdS[i] *
                               (dS[i] + dS[i-1]) for i in range(1, Nz-1)])
                    ys += alphas[-1] * Fs[-1](xs, zs) * dCdS[-1] * dS[-1]
                    ys *= 1 / 2

                elif method == 'simpsons':
                    if len(selfz.Cs) % 2 != 1:
                        raise ValueError('Contour must have odd number of \
points points (even number of intervals) for Simpsons rule.')

                    set_dS = set(np.round(selfz.dS, decimals=12))
                    if len(set_dS) != 1:
                        raise ValueError('Simpsons rule requires equally \
spaced spaced intervals.')

                    dS = set_dS.pop()
                    upper = int((len(selfz.Cs)-1)/2) + 1
                    ys = alphas[0] * Fs[0](xs, zs) * dCdS[0]
                    ys += 4 * sum([alphas[2*i-1] * Fs[2*i-1](xs, zs) *
                                   dCdS[2*i-1] for i in range(1, upper)])
                    ys += 2 * sum([alphas[2*i] * Fs[2*i](xs, zs) *
                                   dCdS[2*i] for i in range(1, upper-1)])
                    ys += alphas[-1] * Fs[-1](xs, zs) * dCdS[-1]
                    ys *= 1/3 * dS

                return ys

            def slice_propagate(selfz, ind_var, slice_at=0,
                                constant_variable='z', method='simpsons'):
                '''
                View cross section (slice) of propagated radiation field.

                One may view the field at a constant value of z, yielding a
                transverse view of the field at a particular point in its
                direction of propagation, or a constant value of x,
                yielding a view of its propagation at a fixed point in the
                transverse domain.

                Parameters
                ----------
                ind_var : float or float array
                    Independent variable along which to evaluate field.
                slice_at : float, optional
                    Value at which to take cross section. The default is 0.
                constant_variable : str, optional
                    Variable to hold constant. May be 'x' or 'z'. The default
                    is 'z'.
                method: str, optional
                    Quadrature method.  May be left_endpoint, right_endpoint or
                    trapezoidal. Default is trapezoid.

                Returns
                -------
                ys: complex or complex array
                    Field strengths at input variables and slice.
                '''

                if constant_variable == 'z':
                    x, z = ind_var, slice_at

                elif constant_variable == 'x':
                    x, z = slice_at, ind_var

                else:
                    raise TypeError('Constant variable must be x or z.')

                Nz = len(selfz.Cs)
                alphas, Fs, dCdS = selfz.alphas, selfz.Fs, selfz.dCdS
                dS = selfz.dS

                if method == 'left_endpoint':
                    # print(method + '\n')
                    ys = sum([alphas[i] * Fs[i](x, z) * dCdS[i] *
                              dS[i] for i in range(Nz-1)])

                elif method == 'right_endpoint':
                    # print(method + '\n')
                    ys = sum([alphas[i+1] * Fs[i+1](x, zs=z) * dCdS[i+1] *
                              dS[i] for i in range(Nz-1)])

                elif method == 'trapezoid':
                    # print(method + '\n')
                    if len(selfz.Cs) == 1:
                        raise ValueError('At least 2 points required for use \
of Trapezoid rule.')
                    ys = alphas[0] * Fs[0](x, zs=z) * dCdS[0] * dS[0]
                    ys += sum([alphas[i] * Fs[i](x, zs=z) * dCdS[i] *
                               (dS[i] + dS[i-1]) for i in range(1, Nz-1)])
                    ys += alphas[-1] * Fs[-1](x, zs=z) * dCdS[-1] * dS[-1]
                    ys *= 1 / 2

                elif method == 'simpsons':
                    # print(method + '\n')
                    if len(selfz.Cs) % 2 != 1 and len(selfz.Cs) > 1:
                        raise ValueError('Contour must have odd number of \
points points (even number of intervals) for Simpsons rule.')

                    if len(selfz.Cs) == 1:
                        raise ValueError('At least 3 points required for use \
of Simpson rule.')

                    set_dS = set(np.round(selfz.dS, decimals=12))

                    if len(set_dS) != 1:
                        raise ValueError('Simpsons rule requires equally \
spaced spaced intervals.')

                    dS = set_dS.pop()
                    upper = int((len(selfz.Cs)-1)/2) + 1
                    ys = alphas[0] * Fs[0](x, zs=z) * dCdS[0]
                    ys += 4 * sum([alphas[2*i-1] * Fs[2*i-1](x, zs=z) *
                                   dCdS[2*i-1] for i in range(1, upper)])
                    ys += 2 * sum([alphas[2*i] * Fs[2*i](x, zs=z) *
                                   dCdS[2*i] for i in range(1, upper-1)])
                    ys += alphas[-1] * Fs[-1](x, zs=z) * dCdS[-1]
                    ys *= 1/3 * dS

                # return ys
                if constant_variable == 'z':
                    return ys[0]
                else:
                    return ys[:, 0]

            def plot_transform(selfz, xs=None, figsize=(11, 4), part='real',
                               ax=None, plot_axis=True, legend_fontsize=12,
                               **linekwargs):
                if ax is None:
                    fig, ax = plt.subplots(1, figsize=figsize)
                else:
                    fig = plt.gcf()
                if part == 'real':
                    ys = selfz.alphas.real
                elif part == 'imag':
                    ys = selfz.alphas.imag
                elif part == 'norm':
                    ys = np.abs(selfz.alphas)
                else:
                    raise ValueError('Part must be real, imag, or norm.')

                if xs is None:
                    if selfz.contour_type in ['real', 'horizontal', 'sdp']:
                        xs = selfz.Cs.real
                    if selfz.contour_type in ['imaginary', 'vertical']:
                        xs = selfz.Cs.imag
                    if selfz.contour_type in ['circle', 'half_circle']:
                        xs = selfz.Ss

                ax.plot(xs, ys, marker='o', markersize=3, **linekwargs)

                if plot_axis:
                    plt.axhline(0, color='grey', linewidth=1.25)

                if 'label' in linekwargs:
                    plt.legend(fontsize=legend_fontsize)

                return fig, ax

        return Propagator

# --------------------------- Field Plotting ----------------------------------

    def plot_field_1d(self, F, *F_args, xs=None, figsize=(12, 5), part='real',
                      plot_regions=False, plot_axis=True, plot_Rhos=True,
                      hatch='///', contrast=.2, Rho_linewidth=1,
                      fig=None, ax=None, legend_fontsize=12, close_others=True,
                      label=None, axis_linewidth=.5, **lineargs):

        if close_others:
            plt.close('all')

        if fig is None and ax is None:
            fig, ax = plt.subplots(1, figsize=figsize)
        else:
            fig, ax = fig, ax

        if xs is None:
            xs = self.all_Xs

        fs = F(xs, *F_args)

        if part == 'real':
            ys = fs.real
        elif part == 'imag':
            ys = fs.imag
        elif part == 'norm':
            ys = np.abs(fs)
        else:
            raise ValueError('Part must be real, imag or norm.')

        if label is not None:
            ax.plot(xs, ys, label=label, **lineargs)
            plt.legend(fontsize=legend_fontsize)

        else:
            ax.plot(xs, ys, **lineargs)

        if plot_axis:
            near_x_axis = (min(np.abs(ys)) / np.ptp(ys)) <= .1
            if near_x_axis:
                ax.axhline(0, color='lightgray', linewidth=axis_linewidth)

            near_y_axis = (min(np.abs(xs)) / np.ptp(xs)) <= .1
            if near_y_axis:
                ax.axvline(0, color='lightgray', linewidth=axis_linewidth)

        if plot_Rhos:
            Rhos = self.Rhos[1:-1]
            msk = np.where((Rhos <= np.max(xs)) * (Rhos >= np.min(xs)))
            for Rho in Rhos[msk]:
                plt.axvline(Rho, ls=':', c='lightgray', lw=Rho_linewidth)

        if plot_regions:
            Rhos = self.Rhos
            ns = self.ns
            minx, maxx = min(xs), max(xs)
            for i in range(len(Rhos)-1):
                if i == 0:
                    L, R = minx, Rhos[i+1]
                elif i == len(Rhos)-2:
                    L, R = Rhos[i], maxx
                else:
                    L, R = Rhos[i], Rhos[i+1]
                n = ns[i]
                rcolor = 1 - 2 / np.pi * np.arctan(contrast*(n - 1))
                hcolor = max(.9 - 2 / np.pi * np.arctan(contrast*(n - 1)), 0)
                plt.axvspan(L, R, color=str(rcolor),
                            linewidth=0)
                plt.axvspan(L, R, fill=None,
                            linewidth=0,
                            hatch=hatch, color=str(hcolor))
        return fig, ax

    def add_1d_plot(self, F, *F_args, xs=None, part='real', ax=None,
                    single_function=True, legend_fontsize=12, label=None,
                    **lineargs):

        if xs is not None and not single_function:
            raise ValueError('Must use single piecewise function if providing \
x array.')

        if ax is None:
            ax = plt.gca()
        if isinstance(F, Iterable):
            single_function = False
        else:
            single_function = True

        if not single_function:
            for f, Xs in zip(F, self.Xs):
                fs = f(Xs, *F_args)
                if part == 'real':
                    ys = fs.real
                elif part == 'imag':
                    ys = fs.imag
                elif part == 'norm':
                    ys = np.abs(fs)
                else:
                    raise ValueError('Part must be real, imag or norm.')
                ax.plot(Xs, ys, **lineargs)

        else:
            if xs is not None:
                all_Xs = xs
            else:
                all_Xs = self.all_Xs
            fs = F(all_Xs, *F_args)
            if part == 'real':
                ys = fs.real
            elif part == 'imag':
                ys = fs.imag
            elif part == 'norm':
                ys = np.abs(fs)
            else:
                raise ValueError('Part must be real, imag or norm.')
        if label is not None:
            ax.plot(all_Xs, ys, label=label, **lineargs)
            plt.legend(fontsize=legend_fontsize)
        else:
            ax.plot(all_Xs, ys, **lineargs)

    def plot_field_2d(self, F, *F_args,
                      xs=None, zs=None,
                      zmin=0, zmax=4, zref=100,
                      levels=40, part='real',
                      figsize=None, maxdim=9,
                      cmap='viridis', equal=True, plot_Rhos=True,
                      Rho_lineargs={'lw': 1, 'ls': ':', 'c': 'k'},
                      colorbar_orientation=None, colorbar=True, pad=.05,
                      shrink=.9, colorbar_frac=.1, colorbar_aspect=15,
                      colorbar_nticks=5, anchor=(.5, .5),
                      colorbar_format='%.2f',
                      **contourargs):
        '''Plot field F as filled contour plot.

        If no xs are provided, plots entire region defined by SlabExact class.

        Note that we avoid applying meshgrid to xs and zs until after calling
        function. Since fields are sums of products of single functions of xs
        and single functions of zs it would be wasteful to call it on the meshed
        arrays.'''

        plt.close('all')

        if zs is None:
            zs = np.linspace(zmin, zmax, zref)
        zmin, zmax = np.min(zs), np.max(zs)

        if xs is None:
            xs = self.all_Xs

        lenzs, lenxs = np.ptp(zs), np.ptp(xs)

        if figsize is None:
            figsize = 1 / max(lenzs, lenxs) * np.array([lenxs, lenzs]) * maxdim
        else:
            figsize = figsize

        fig, ax = plt.subplots(1, figsize=figsize)

        Fs = F(xs, zs, *F_args)

        Xs, Zs = np.meshgrid(xs, zs)

        if part == 'real':
            Ys = Fs.real
        elif part == 'imag':
            Ys = Fs.imag
        elif part == 'norm':
            Ys = np.abs(Fs)
        else:
            raise ValueError('Part must be real, imag or norm.')

        axmap = ax.contourf(Xs, Zs,  Ys, levels=levels, cmap=cmap,
                            **contourargs)
        if equal:
            ax.set_aspect('equal')

        Rhos = self.Rhos[1:-1]

        if plot_Rhos:
            msk = np.where((Rhos <= np.max(xs)) * (Rhos >= np.min(xs)))
            for Rho in Rhos[msk]:
                plt.axvline(Rho, linestyle=Rho_lineargs['ls'],
                            linewidth=Rho_lineargs['lw'],
                            color=Rho_lineargs['c'])
        if colorbar:
            if colorbar_orientation is None:
                if lenzs >= lenxs:
                    colorbar_orientation = 'horizontal'
                else:
                    colorbar_orientation = 'vertical'
            else:
                colorbar_orientation = colorbar_orientation
            ticks = np.linspace(np.min(Ys), np.max(Ys), colorbar_nticks)

            plt.colorbar(axmap, pad=pad, shrink=shrink,
                         orientation=colorbar_orientation,
                         anchor=anchor, fraction=colorbar_frac,
                         aspect=colorbar_aspect,
                         ticks=ticks, format=colorbar_format)
        return fig, ax

    def add_2d_plot(self, F, *F_args, ax=None,
                    xs=None, zs=None,
                    zmin=0, zmax=10, zref=100,
                    part='real', cmap='viridis',
                    levels=40, plot_Rhos=True,
                    Rho_lineargs={'lw': 1, 'ls': ':', 'c': 'k'},
                    colorbar=False, colorbar_orientation=None, pad=.05,
                    shrink=1, colorbar_frac=.1, colorbar_aspect=15,
                    anchor=(.5, .5), colorbar_nticks=5, colorbar_format='%.2f',
                    **contourargs):

        if ax is None:
            ax = plt.gca()

        if zs is None:
            zs = np.linspace(zmin, zmax, zref)

        if xs is None:
            xs = self.all_Xs

        Fs = F(xs, zs, *F_args)

        if part == 'real':
            Ys = Fs.real
        elif part == 'imag':
            Ys = Fs.imag
        elif part == 'norm':
            Ys = np.abs(Fs)
        else:
            raise ValueError('Part must be real, imag or norm.')

        Xs, Zs = np.meshgrid(xs, zs)
        lenzs, lenxs = np.ptp(Zs), np.ptp(Xs)

        axmap = ax.contourf(Xs, Zs,  Ys, levels=levels, cmap=cmap,
                            **contourargs)

        if plot_Rhos:
            for Rho in self.Rhos[1:-1]:
                ax.axvline(Rho, linestyle=Rho_lineargs['ls'],
                           linewidth=Rho_lineargs['lw'],
                           color=Rho_lineargs['c'])
        if colorbar:
            if colorbar_orientation is None:
                if lenzs >= lenxs:
                    colorbar_orientation = 'horizontal'
                else:
                    colorbar_orientation = 'vertical'
            else:
                colorbar_orientation = colorbar_orientation
            ticks = np.linspace(np.min(Ys), np.max(Ys), colorbar_nticks)

            plt.colorbar(axmap, pad=pad, shrink=shrink,
                         orientation=colorbar_orientation,
                         anchor=anchor, fraction=colorbar_frac,
                         aspect=colorbar_aspect, ax=ax,
                         ticks=ticks, format=colorbar_format)

    def plot_field_2d_surface(self, F, *F_args, xs=None, zs=None,
                              zmin=0, zmax=4, zref=100, part='real',
                              figsize=(10, 5), cmap='viridis',
                              azim=-90, elev=55, roll=0, zoom=2.5,
                              rstride=4, cstride=4, z_lim_factor=2,
                              colorbar=False, pad=.15, shrink=.85,
                              colorbar_frac=.15, anchor=(.5, .5),
                              orient='vertical', **surfaceargs):
        '''Plot field F as filled contour plot.

        If no xs are provided, plots entire region defined by SlabExact class.

        Note that we avoid applying meshgrid to xs and zs until after calling
        function. Since field is a product of single function of xs and
        single function of zs it would be wasteful to call it on the meshed
        arrays.'''
        plt.close('all')

        if isinstance(F, Iterable):
            single_function = False
        else:
            single_function = True

        if xs is not None and not single_function:
            raise ValueError('Must use single piecewise function if providing \
x array.')

        if zs is None:
            Zs = np.linspace(zmin, zmax, zref)
        else:
            Zs = zs
            zmin, zmax = np.min(Zs), np.max(Zs)

        if xs is None:
            if single_function:
                all_Xs = self.all_Xs  # doesn't duplicate endpoints
            else:
                all_Xs = np.concatenate(self.all_Xs)  # duplicates endpoints
        else:
            all_Xs = xs

        fig, ax = plt.subplots(1, figsize=figsize,
                               subplot_kw={"projection": "3d"})
        if not single_function:
            fs = []
            for f, Xs in zip(F, self.Xs):
                fs.append(f(Xs, Zs, *F_args,))

            fs = np.concatenate(fs, axis=1)
        else:
            fs = F(all_Xs, Zs, *F_args)

        Xg, Zg = np.meshgrid(all_Xs, Zs)

        if part == 'real':
            ys = fs.real
        elif part == 'imag':
            ys = fs.imag
        elif part == 'norm':
            ys = np.abs(fs)
        else:
            raise ValueError('Part must be real, imag or norm.')

        axmap = ax.plot_surface(Xg, Zg,  ys, clip_on=False, cmap=cmap,
                                rstride=rstride, cstride=cstride,
                                **surfaceargs)
        lims = (np.ptp(Xg), np.ptp(Zg), min(np.ptp(Xg), np.ptp(Zg)))
        lenys = np.ptp(ys)
        ax.set_zlim(-z_lim_factor * lenys, z_lim_factor*lenys)
        ax.set_box_aspect(lims, zoom=zoom)
        ax.set_axis_off()
        ax.view_init(elev, azim, roll)

        if colorbar:
            plt.colorbar(axmap, pad=pad, shrink=shrink, orientation=orient,
                         anchor=anchor, fraction=colorbar_frac)
        return fig, ax

# -------------------------- Special Plots ------------------------------------

    def plot_points(self, Cs, m='o', ms=5, ax=None, **kwargs):
        '''Plot provided points in complex plane.'''
        try:
            len(Cs)
            Cs = np.asanyarray(Cs)
        except TypeError:
            Cs = np.asanyarray([Cs])
        if ax is None:
            ax = plt.gca()
        for Z in Cs:
            ax.plot(Z.real, Z.imag, marker=m, markersize=ms, **kwargs)

    def plot_refractive_index(self, figsize=(11, 4), contrast=.2,
                              color='cornflowerblue', label=None,
                              plot_regions=False, hatch='///',
                              plot_Rhos=True, ax=None, part='both',
                              legend_fontsize=12, Rho_linewidth=1,
                              **linekwargs):
        """Plot refractive index profile."""

        if ax is not None:
            fig = plt.gcf()
        else:
            fig, ax = plt.subplots(1, figsize=figsize)

        Rhos = self.Rhos
        if part == 'real':
            ns = self.ns.real
        elif part == 'imag':
            ns = self.ns.imag
        elif part == 'both':
            nsr = self.ns.real
            nsi = self.ns.imag
        else:
            raise ValueError('Part must be real or imag or both.')

        if label is None:
            label = part
            if part == 'both':
                label1 = 'real'
                label2 = 'imag'
        if part != 'both':
            for i, (n, Xs) in enumerate(zip(ns, self.Xs)):
                if i == 0:
                    ax.plot(Xs, n * np.ones_like(Xs), color=color, label=label,
                            **linekwargs)
                ax.plot(Xs, n * np.ones_like(Xs), color=color, **linekwargs)
            for i in range(1, len(Rhos)-1):
                Rho = Rhos[i]
                nl, nr = ns[i-1], ns[i]
                ax.plot([Rho, Rho], [nl, nr], color=color, **linekwargs)
        else:
            for i, (n, Xs) in enumerate(zip(nsr, self.Xs)):
                if i == 0:
                    ax.plot(Xs, n * np.ones_like(Xs), color='C0', label=label1,
                            **linekwargs)
                ax.plot(Xs, n * np.ones_like(Xs), color='C0', **linekwargs)
            for i in range(1, len(Rhos)-1):
                Rho = Rhos[i]
                nl, nr = nsr[i-1], nsr[i]
                ax.plot([Rho, Rho], [nl, nr], color='C0', **linekwargs)

            for i, (n, Xs) in enumerate(zip(nsi, self.Xs)):
                if i == 0:
                    ax.plot(Xs, n * np.ones_like(Xs), color='C1', label=label2,
                            **linekwargs)
                ax.plot(Xs, n * np.ones_like(Xs), color='C1', **linekwargs)
            for i in range(1, len(Rhos)-1):
                Rho = Rhos[i]
                nl, nr = nsi[i-1], nsi[i]
                ax.plot([Rho, Rho], [nl, nr], color='C1', **linekwargs)

        if plot_Rhos:
            for Rho in Rhos[1:-1]:
                plt.axvline(Rho, ls=':', c='lightgray', lw=Rho_linewidth)

        if plot_regions:
            for i in range(len(Rhos)-1):
                n = ns[i]
                rcolor = 1 - 2 / np.pi * np.arctan(contrast*(n - 1))
                hcolor = max(.9 - 2 / np.pi * np.arctan(contrast*(n - 1)), 0)
                plt.axvspan(Rhos[i], Rhos[i+1], color=str(rcolor),
                            linewidth=0)
                plt.axvspan(Rhos[i], Rhos[i+1], fill=None,
                            linewidth=0,
                            hatch=hatch, color=str(hcolor))
        plt.legend(fontsize=legend_fontsize)
        return fig, ax

    def determinant_plot(self, rmin, rmax, imin, imax,
                         plane='Z', derivate=False,
                         rref=200, iref=200, levels=70,
                         log_abs=True, equal=False, grid=True,
                         figsize=(11, 5), cmap='viridis',
                         part='norm', facecolor='gray', field_type='TE',
                         mode_type='guided', plot_sdp=True, sdp_sign=-1,
                         plot_axis=True, axis_linewidth=.7,
                         axis_linecolor='k', Normalizer=None, sign=1,
                         colorbar=True, pad=.02, shrink=1, colorbar_frac=.05,
                         colorbar_aspect=30, anchor=(.5, 1),
                         **contourargs):
        plt.close('all')
        xs = np.linspace(rmin, rmax, num=rref)
        ys = np.linspace(imin, imax, num=iref)
        Xs, Ys = np.meshgrid(xs, ys)
        Cs = Xs + 1j * Ys

        Fs = self.determinant(Cs, field_type=field_type, mode_type=mode_type,
                              plane=plane, Normalizer=Normalizer, sign=sign,
                              derivate=derivate)

        if part == 'real':
            Fs = Fs.real
        elif part == 'imag':
            Fs = Fs.imag
        elif part == 'norm':
            Fs = np.abs(Fs)
        elif part == 'phase':
            Fs = np.angle(Fs)
        else:
            raise ValueError('Part must be real, imag, norm or phase.')

        if log_abs:
            Fs = np.log(np.abs(Fs))

        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot()
        if plot_axis:
            if (imin <= 0) * (imax >= 0):
                ax.axhline(0, color=axis_linecolor, linewidth=axis_linewidth)
            if (rmin <= 0) * (rmax >= 0):
                ax.axvline(0, color=axis_linecolor, linewidth=axis_linewidth)

        ax.grid(grid)
        ax.set_facecolor(facecolor)

        im = ax.contour(xs, ys, Fs, levels=levels, cmap=cmap, **contourargs)

        if plot_sdp:
            if plane == 'Z':
                sdp_points = self.sdp(xs, sdp_sign=sdp_sign, plane=plane)
                sdp_ys = sdp_points.imag
                msk = np.where((sdp_ys < imax) * (sdp_ys > imin))
                ax.plot(xs[msk], sdp_ys[msk])

            elif plane == 'Beta':
                sdp_points = self.sdp(ys, sdp_sign=sdp_sign, plane=plane)
                sdp_xs, sdp_ys = sdp_points.real, sdp_points.imag
                if (rmin <= self.K0 * self.n0) * (rmax >= self.K0 * self.n0):
                    ax.plot(sdp_xs, sdp_ys)

            elif plane == 'Psi':
                sdp_points = self.sdp(ys, sdp_sign=sdp_sign, plane=plane)
                sdp_xs, sdp_ys = sdp_points.real, sdp_points.imag
                msk = np.where((sdp_xs < rmax) * (sdp_xs > rmin))
                ax.plot(sdp_xs[msk], sdp_ys[msk])
            else:
                raise ValueError("Plane must be 'Z', 'Beta' or 'Psi'.")

        if equal:
            ax.axis('equal')

        if colorbar:
            plt.colorbar(im, pad=pad, shrink=shrink, orientation='vertical',
                         anchor=anchor, fraction=colorbar_frac,
                         aspect=colorbar_aspect)

        return fig, ax

    def spectral_integrand_plot(self, rmin, rmax, imin, imax,
                                rref=100, iref=100, levels=100,
                                f0=None,  s=None, x=0, z=0,
                                Lx=None, Rx=None,
                                field_type='TE',
                                class_A_only=False,
                                class_B_only=False,
                                plot_sdp=True, sdp_sign=-1,
                                grid=True, facecolor='grey',
                                figsize=(11, 5), width='1250px',
                                log_abs=True, part='norm',
                                max_z=None, min_z=None,
                                vmin=-4, vmax=9, cmap='viridis',
                                plane='Z', Normalizer=None,
                                colorbar=True, pad=.02, shrink=1,
                                colorbar_frac=.05, colorbar_aspect=30,
                                anchor=(.5, 1), det_power=0, **args):
        plt.close('all')
        fig, ax = plt.subplots(1, figsize=figsize)

        xs, ys = np.linspace(rmin, rmax, rref), np.linspace(imin, imax, iref)

        Xs, Ys = np.meshgrid(xs, ys)
        Cs = Xs + 1j * Ys

        if class_A_only and class_B_only:
            raise ValueError('At most one of class_A_only and class_B_only can \
be set to True.')

        if class_A_only:
            alpha_A = self.radiation_transform(Cs, f0=f0, s=s, sign=1,
                                               Lx=Lx, Rx=Rx, plane=plane,
                                               field_type=field_type,
                                               Normalizer=Normalizer)
            alpha_B = None

        elif class_B_only:
            alpha_B = self.radiation_transform(Cs, f0=f0, s=s, sign=-1,
                                               Lx=Lx, Rx=Rx, plane=plane,
                                               field_type=field_type,
                                               Normalizer=Normalizer)
            alpha_A = None
        else:
            alpha_A = self.radiation_transform(Cs, f0=f0, s=s, sign=1,
                                               Lx=Lx, Rx=Rx, plane=plane,
                                               field_type=field_type,
                                               Normalizer=Normalizer)
            alpha_B = self.radiation_transform(Cs, f0=f0, s=s, sign=-1,
                                               Lx=Lx, Rx=Rx, plane=plane,
                                               field_type=field_type,
                                               Normalizer=Normalizer)

        fs = self.spectral_integrand(Cs, f0=f0, s=s, x=x, z=z,
                                     Lx=Lx, Rx=Rx, plane=plane,
                                     alpha_A=alpha_A,
                                     alpha_B=alpha_B,
                                     class_A_only=class_A_only,
                                     class_B_only=class_B_only,
                                     field_type=field_type,
                                     Normalizer=Normalizer,
                                     det_power=det_power)
        if part == 'real':
            fs = fs.real
        elif part == 'imag':
            fs = fs.imag
        elif part == 'norm':
            fs = np.abs(fs)
        else:
            raise ValueError('Part must be real, imag or norm.')
        if max_z is not None:
            vmax = max_z
            mask = np.where(fs > max_z)
            fs[mask] = max_z
        if min_z is not None:
            vmin = min_z
            mask = np.where(fs < min_z)
            fs[mask] = min_z
        if log_abs:
            fs = np.log(np.abs(fs))
        axmap = ax.contour(Xs, Ys, fs, levels=levels,
                           vmin=vmin, vmax=vmax,
                           cmap=cmap, **args)

        if plot_sdp:
            if plane == 'Z':
                sdp_vals = self.sdp(xs, sdp_sign=sdp_sign, plane=plane)
                sdp_xs, sdp_ys = sdp_vals.real, sdp_vals.imag
                msk = np.where((sdp_ys <= imax) * (sdp_ys >= imin))
                ax.plot(sdp_xs[msk], sdp_ys[msk], **args)
            elif plane == 'Beta':
                sdp_points = self.sdp(ys, sdp_sign=sdp_sign, plane=plane)
                sdp_xs, sdp_ys = sdp_points.real, sdp_points.imag
                if (rmin <= self.K0 * self.n0) * (rmax >= self.K0 * self.n0):
                    ax.plot(sdp_xs, sdp_ys)
            elif plane == 'Psi':
                sdp_ys = ys
                sdp_xs = sdp_sign * np.sign(sdp_ys) * \
                    np.arccos(1 / np.cosh(sdp_ys))
                msk = np.where((sdp_xs <= rmax) * (sdp_xs >= rmin))
                ax.plot(sdp_xs[msk], sdp_ys[msk], **args)
            else:
                raise ValueError("Plane must be 'Z', 'Beta' or 'Psi'.")

        ax.grid(grid)
        ax.set_facecolor(facecolor)
        if s is not None:
            plt.title('Spectral Integrand Plot for x = %.2f, s = %.2f, \
z = %.2f' % (x, s, z), fontsize=12)
        else:
            plt.title('Spectral Integrand Plot for x = %.2f, z = %.2f' % (x, z),
                      fontsize=12)
        if colorbar:
            plt.colorbar(axmap, pad=pad, shrink=shrink,
                         orientation='vertical',
                         anchor=anchor, fraction=colorbar_frac,
                         aspect=colorbar_aspect, ax=ax)

    def spectral_integrand_surface_plot(self, rmin, rmax, imin, imax,
                                        rref=100, iref=100,
                                        f0=None,  s=None, x=0, z=0,
                                        Lx=None, Rx=None,
                                        field_type='TE',
                                        class_A_only=False,
                                        class_B_only=False,
                                        max_z=None, min_z=None,
                                        det_power=0,
                                        width='1250px', plane='Z',
                                        log_abs=True, part='norm',
                                        cmap='viridis',
                                        Normalizer=None,
                                        colorbar_aspect=30,
                                        figsize=(10, 5),
                                        azim=-90, elev=55, roll=0, zoom=2.5,
                                        rstride=1, cstride=1, z_lim_factor=2,
                                        colorbar=False, pad=.15, shrink=.85,
                                        colorbar_frac=.15, anchor=(.5, .5),
                                        orient='vertical',
                                        **surfaceargs):
        '''Plot field F as filled contour plot.

        If no xs are provided, plots entire region defined by SlabExact class.

        Note that we avoid applying meshgrid to xs and zs until after calling
        function. Since field is a product of single function of xs and
        single function of zs it would be wasteful to call it on the meshed
        arrays.'''
        plt.close('all')

        fig, ax = plt.subplots(1, figsize=figsize,
                               subplot_kw={"projection": "3d"})
        xs, ys = np.linspace(rmin, rmax, rref), np.linspace(imin, imax, iref)

        Xs, Ys = np.meshgrid(xs, ys)
        Cs = Xs + 1j * Ys

        if class_A_only and class_B_only:
            raise ValueError('At most one of class_A_only and class_B_only can \
be set to True.')

        if class_A_only:
            alpha_A = self.radiation_transform(Cs, f0=f0, s=s, sign=1,
                                               Lx=Lx, Rx=Rx, plane=plane,
                                               field_type=field_type,
                                               Normalizer=Normalizer)
            alpha_B = None

        elif class_B_only:
            alpha_B = self.radiation_transform(Cs, f0=f0, s=s, sign=-1,
                                               Lx=Lx, Rx=Rx, plane=plane,
                                               field_type=field_type,
                                               Normalizer=Normalizer)
            alpha_A = None
        else:
            alpha_A = self.radiation_transform(Cs, f0=f0, s=s, sign=1,
                                               Lx=Lx, Rx=Rx, plane=plane,
                                               field_type=field_type,
                                               Normalizer=Normalizer)
            alpha_B = self.radiation_transform(Cs, f0=f0, s=s, sign=-1,
                                               Lx=Lx, Rx=Rx, plane=plane,
                                               field_type=field_type,
                                               Normalizer=Normalizer)

        fs = self.spectral_integrand(Cs, f0=f0, s=s, x=x, z=z,
                                     Lx=Lx, Rx=Rx, plane=plane,
                                     alpha_A=alpha_A,
                                     alpha_B=alpha_B,
                                     class_A_only=class_A_only,
                                     class_B_only=class_B_only,
                                     field_type=field_type,
                                     Normalizer=Normalizer,
                                     det_power=det_power)
        if part == 'real':
            fs = fs.real
        elif part == 'imag':
            fs = fs.imag
        elif part == 'norm':
            fs = np.abs(fs)
        else:
            raise ValueError('Part must be real, imag or norm.')
        if max_z is not None:
            mask = np.where(fs > max_z)
            fs[mask] = max_z
        if min_z is not None:
            mask = np.where(fs < min_z)
            fs[mask] = min_z
        if log_abs:
            fs = np.log(np.abs(fs))

        axmap = ax.plot_surface(Xs, Ys, fs, clip_on=False, cmap=cmap,
                                rstride=rstride, cstride=cstride,
                                **surfaceargs)

        lims = (np.ptp(Xs), np.ptp(Ys), min(np.ptp(Xs), np.ptp(Ys)))
        fs_width = np.ptp(fs)
        ax.set_zlim(-z_lim_factor * fs_width, z_lim_factor*fs_width)
        ax.set_box_aspect(lims, zoom=zoom)
        ax.set_axis_off()
        ax.view_init(elev, azim, roll)

        if colorbar:
            plt.colorbar(axmap, pad=pad, shrink=shrink, orientation=orient,
                         anchor=anchor, fraction=colorbar_frac)
        return fig, ax

# --------------------------- Animations --------------------------------------

    def animate_field_1d(self, F, name, figsize=(12, 5), xs=None,
                         fps=32, secs=4, part='real', color='blue',
                         plot_Rhos=True,  plot_axis=True, contrast=.2,
                         plot_regions=False, hatch='///', **lineargs):
        'Animate field F and save as name.mp4.'
        plt.close('all')

        fig, ax = plt.subplots(1, figsize=figsize)
        camera = Camera(fig)
        N = int(fps * secs)

        if xs is None:
            xs = self.all_Xs
        fs = F(xs)

        Rhos = self.Rhos

        # Animate
        for k in range(N):
            ys = fs * np.exp(-1j * k/N * 2*np.pi)

            if part == 'real':
                ys = ys.real
            elif part == 'imag':
                ys = ys.imag
            elif part == 'norm':
                ys = np.abs(ys)
            else:
                raise ValueError('Part must be real, imag or norm.')
            ax.plot(xs, ys, color=color, **lineargs)

            if plot_Rhos:
                for Rho in Rhos[1:-1]:
                    plt.axvline(Rho, linestyle=':', color='lightgray')

            if plot_axis:
                plt.axhline(0, color='lightgray')
                plt.axvline(0, color='lightgray')

            if plot_regions:
                ns = self.ns
                for i in range(len(Rhos)-1):
                    n = ns[i]
                    rcolor = 1 - 2 / np.pi * np.arctan(contrast*(n - 1))
                    hcolor = max(.9 - 2 / np.pi*np.arctan(contrast*(n - 1)), 0)
                    plt.axvspan(Rhos[i], Rhos[i+1], color=str(rcolor),
                                linewidth=0)
                    plt.axvspan(Rhos[i], Rhos[i+1], fill=None,
                                linewidth=0,
                                hatch=hatch, color=str(hcolor))
            camera.snap()

        animation = camera.animate(blit=True)
        animation.save(name + '.mp4', fps=fps)

    def animate_field_2d(self, F, name, fps=25, secs=2,
                         zmin=0, zmax=4, zref=100,
                         xs=None, zs=None, levels=30,
                         part='real', figsize=None, maxdim=9,
                         equal=True, plot_Rhos=True,
                         product_func=True,
                         colorbar=False, pad=.05, shrink=.9, orient=None,
                         colorbar_frac=.1, colorbar_aspect=15, anchor=(.5, .5),
                         color_min=None, color_max=None,
                         **contourargs):
        'Animate field F as filled contour plot and save as name.mp4.'
        plt.close('all')

        N = int(fps * secs)

        if zs is None:
            zs = np.linspace(zmin, zmax, zref)

        if xs is None:
            xs = self.all_Xs  # doesn't duplicate endpoints

        lenzs, lenxs = np.ptp(zs), np.ptp(xs)

        if figsize is None:
            figsize = 1 / max(lenzs, lenxs) * np.array([lenxs, lenzs]) * maxdim
        else:
            figsize = figsize

        fig, ax = plt.subplots(1, figsize=figsize)

        if equal:
            ax.set_aspect('equal')

        if colorbar:
            if orient is None:
                if lenzs >= lenxs:
                    orient = 'horizontal'
                else:
                    orient = 'vertical'
            else:
                orient = orient

        camera = Camera(fig)

        Xs, Zs = np.meshgrid(xs, zs)
        if product_func:
            Fs = F(xs, zs)
        else:
            Fs = F(Xs, Zs)

        # Animate
        for k in range(N):
            Ys = Fs * np.exp(-1j * k/N * 2*np.pi)

            if part == 'real':
                Ys = Ys.real
            elif part == 'imag':
                Ys = Ys.imag
            elif part == 'norm':
                Ys = np.abs(Ys)
            else:
                raise ValueError('Part must be real, imag or norm.')

            if color_max is None or color_min is None and k == 0:

                axmap = ax.contourf(Xs, Zs,  Ys, levels=levels,
                                    vmin=color_min, vmax=color_max,
                                    **contourargs)

            if plot_Rhos:
                for Rho in self.Rhos[1:-1]:
                    plt.axvline(Rho, linestyle=':', linewidth=1,
                                color='k')
            if colorbar and k == 0:
                plt.colorbar(axmap, pad=pad, shrink=shrink, orientation=orient,
                             anchor=anchor, fraction=colorbar_frac,
                             aspect=colorbar_aspect)
            camera.snap()

        animation = camera.animate(blit=True)
        animation.save(name + '.mp4', fps=fps)

    def animate_field_2d_surface(self, F, name, *F_args,
                                 fps=25, secs=2,
                                 xs=None, zs=None,
                                 zmin=0, zmax=4, zref=100,
                                 part='real',
                                 figsize=(10, 5), cmap='viridis',
                                 azim=-90, elev=55, roll=0, zoom=2.5,
                                 rstride=4, cstride=4,
                                 maxdim=9, z_lim_factor=2,
                                 equal=True,
                                 product_func=True,
                                 color_min=None, color_max=None,
                                 dpi=100, **surfaceargs):
        'Animate field F as 3d surface plot and save as name.mp4.'
        plt.close('all')

        if isinstance(F, Iterable):
            single_function = False
        else:
            single_function = True

        N = int(fps * secs)

        if xs is not None and not single_function:
            raise ValueError('Must use single piecewise function if providing \
x array.')

        if zs is None:
            Zs = np.linspace(zmin, zmax, zref)
        else:
            Zs = zs
            zmin, zmax = np.min(Zs), np.max(Zs)

        if xs is None:
            if single_function:
                all_Xs = self.all_Xs  # doesn't duplicate endpoints
            else:
                all_Xs = np.concatenate(self.all_Xs)  # duplicates endpoints
        else:
            all_Xs = xs

        lenzs, lenxs = np.ptp(Zs), np.ptp(all_Xs)

        if figsize is None:
            figsize = 1 / max(lenzs, lenxs) * np.array([lenxs, lenzs]) * maxdim
        else:
            figsize = figsize

        fig, ax = plt.subplots(1, figsize=figsize,
                               subplot_kw={"projection": "3d"})

        camera = Camera(fig)

        if not single_function:
            Xgs, Zgs = [], []
            fs = []
            for f, Xs in zip(F, self.Xs):
                Xg, Zg = np.meshgrid(Xs, Zs)
                Xgs.append(Xg)
                Zgs.append(Zg)
                if product_func:
                    fs.append(f(Xs, Zs, *F_args))
                else:
                    fs.append(f(Xg, Zg, *F_args))

            Xg = np.concatenate(Xgs, axis=1)
            Zg = np.concatenate(Zgs, axis=1)
            fs = np.concatenate(fs, axis=1)
        else:
            Xg, Zg = np.meshgrid(all_Xs, Zs)
            if product_func:
                fs = F(all_Xs, Zs, *F_args)
            else:
                fs = F(Xg, Zg, *F_args)

        lims = (np.ptp(Xg), np.ptp(Zg), min(np.ptp(Xg), np.ptp(Zg)))
        ax.set_box_aspect(lims, zoom=zoom)
        ax.set_axis_off()
        ax.view_init(elev, azim, roll)

        # Animate
        for k in range(N):
            ys = fs * np.exp(-1j * k/N * 2*np.pi)

            if part == 'real':
                ys = ys.real
            elif part == 'imag':
                ys = ys.imag
            elif part == 'norm':
                ys = np.abs(ys)
            else:
                raise ValueError('Part must be real, imag or norm.')

            if k == 0:
                lenys = np.ptp(ys)
                if lenys > 0:
                    ax.set_zlim(-z_lim_factor * lenys, z_lim_factor*lenys)

            ax.plot_surface(Xg, Zg, ys, clip_on=False, cmap=cmap,
                            rstride=rstride, cstride=cstride,
                            vmin=color_min, vmax=color_max,
                            **surfaceargs)

            camera.snap()

        animation = camera.animate(blit=True)
        animation.save(name + '.mp4', fps=fps, dpi=dpi)

    def radiation_mode_beta_animation(self, name, beta_range=None,
                                      imag_type=False, field_type='TE',
                                      figsize=(12, 5), fps=32, secs=4,
                                      part='real', color='blue', c0=1,
                                      sign='+1',
                                      plot_Rhos=True, plot_axis=True,
                                      plot_regions=False, hatch='///',
                                      contrast=.2, single_function=True,
                                      **lineargs):
        'Animate radiation field as function of beta and save as name.mp4.'
        plt.close('all')

        fig, ax = plt.subplots(1, figsize=figsize)
        camera = Camera(fig)
        N = fps * secs

        if beta_range is None:
            beta_range = [-self.K_low, self.K_low]

        beta_range = np.linspace(beta_range[0], beta_range[1], N)

        if imag_type:
            beta_range = np.array(beta_range, dtype=complex)
            beta_range *= 1j

        # Animate
        for beta in beta_range:
            F = self.fields(beta, c0=c0, field_type=field_type,
                            mode_type='radiation',
                            single_function=single_function,
                            sign=sign)

            if not single_function:
                fs = []
                for f, Xs in zip(F, self.Xs):
                    fs.append(f(Xs))
                ys = np.concatenate(fs)
                all_X = np.concatenate(self.Xs)
            else:
                ys = F(self.all_Xs)
                all_X = self.all_Xs

            if part == 'real':
                ys = ys.real
            elif part == 'imag':
                ys = ys.imag
            elif part == 'norm':
                ys = np.abs(ys)
            else:
                raise ValueError('Part must be real, imag or norm.')
            ax.plot(all_X, ys, color=color, **lineargs)
            if plot_Rhos:
                Rhos = self.Rhos[1:-1]
                for Rho in Rhos:
                    plt.axvline(Rho, linestyle=':', color='lightgray')
            if plot_axis:
                plt.axhline(0, color='lightgray')
                plt.axvline(0, color='lightgray')
            if plot_regions:
                ns = self.ns
                for i in range(len(Rhos)-1):
                    n = ns[i]
                    rcolor = 1 - 2 / np.pi * np.arctan(contrast*(n - 1))
                    hcolor = max(.9 - 2 / np.pi*np.arctan(contrast*(n - 1)), 0)
                    plt.axvspan(Rhos[i], Rhos[i+1], color=str(rcolor),
                                linewidth=0)
                    plt.axvspan(Rhos[i], Rhos[i+1], fill=None,
                                linewidth=0,
                                hatch=hatch, color=str(hcolor))
            camera.snap()

        animation = camera.animate(blit=True)
        animation.save(name + '.mp4', fps=fps)

    def radiation_mode_Z_animation(self, name, Z_range=None,
                                   field_type='TE',
                                   figsize=(12, 5), fps=32, secs=4,
                                   part='real', color='blue', c0=1,
                                   sign='+1', paper_method=False,
                                   plot_Rhos=True, plot_axis=True,
                                   plot_regions=False, hatch='///',
                                   contrast=.2, single_function=True,
                                   **lineargs):
        '''Animate radiation field as function of Z and save as name.mp4.'''
        plt.close('all')

        fig, ax = plt.subplots(1, figsize=figsize)
        camera = Camera(fig)
        N = fps * secs

        if Z_range is None:
            Z_range = [-2*self.Z_evanescent, 2*self.Z_evanescent]

        Z_range = np.linspace(Z_range[0], Z_range[1], N)

        # Animate
        for Z in Z_range:
            F = self.fields_Z(Z, c0=c0, field_type=field_type,
                              mode_type='radiation',
                              single_function=single_function,
                              sign=sign, paper_method=paper_method)

            if not single_function:
                fs = []
                for f, Xs in zip(F, self.Xs):
                    fs.append(f(Xs))
                ys = np.concatenate(fs)
                all_X = np.concatenate(self.Xs)
            else:
                ys = F(self.all_Xs)
                all_X = self.all_Xs

            if part == 'real':
                ys = ys.real
            elif part == 'imag':
                ys = ys.imag
            elif part == 'norm':
                ys = np.abs(ys)
            else:
                raise ValueError('Part must be real, imag or norm.')
            ax.plot(all_X, ys, color=color, **lineargs)
            if plot_Rhos:
                Rhos = self.Rhos[1:-1]
                for Rho in Rhos:
                    plt.axvline(Rho, linestyle=':', color='lightgray')
            if plot_axis:
                plt.axhline(0, color='lightgray')
                plt.axvline(0, color='lightgray')
            if plot_regions:
                ns = self.ns
                for i in range(len(Rhos)-1):
                    n = ns[i]
                    rcolor = 1 - 2 / np.pi * np.arctan(contrast*(n - 1))
                    hcolor = max(.9 - 2 / np.pi*np.arctan(contrast*(n - 1)), 0)
                    plt.axvspan(Rhos[i], Rhos[i+1], color=str(rcolor),
                                linewidth=0)
                    plt.axvspan(Rhos[i], Rhos[i+1], fill=None,
                                linewidth=0,
                                hatch=hatch, color=str(hcolor))
            camera.snap()

        animation = camera.animate(blit=True)
        animation.save(name + '.mp4', fps=fps)

    def animate_propagation_constants(self, rmin, rmax, imin, imax,
                                      wlmin=1e-6, wlmax=2e-6, Nwl=5,
                                      rref=50, iref=50, levels=50,
                                      name='prop_const_animation',
                                      variable='Psi', cmap='viridis',
                                      field_type='TE', mode_type='leaky',
                                      fps=25, figsize=(10, 6),
                                      colorbar=False, pad=.05, shrink=.9,
                                      orient='vertical', colorbar_frac=.1,
                                      colorbar_aspect=15, anchor=(.5, .5),
                                      color_min=None, color_max=None,
                                      facecolor='grey', grid=True,
                                      bitrate=-1, dpi=300, Z_type='standard',
                                      **contourargs):
        '''Animate propagation constants in complex plane as wavelength changes
        and save as name.mp4.'''

        if variable not in ['Psi', 'beta', 'Z']:
            raise ValueError("Variable must be 'Psi', 'beta' or 'Z'.")
        plt.close('all')

        xs, ys = np.linspace(rmin, rmax, rref), np.linspace(imin, imax, iref)
        Xs, Ys = np.meshgrid(xs, ys)
        Cs = Xs + 1j * Ys

        fig, ax = plt.subplots(1, figsize=figsize)
        camera = Camera(fig)

        wls = np.linspace(wlmin, wlmax, Nwl)

        # Animate
        for k, wl in enumerate(wls):
            self.wl = wl
            K = self.K0 * self.n0

            if variable == 'Psi':
                Fs = self.determinant_Z(K * np.sin(Cs), field_type=field_type,
                                        mode_type=mode_type)

                def sdp(y, c=1):
                    return np.arccos(c / np.cosh(y))
                sdp_ys = np.linspace(0, imax, 101)
                sdp_xs = sdp(sdp_ys)

            elif variable == 'Z':
                Fs = self.determinant_Z(Cs, field_type=field_type,
                                        mode_type=mode_type)

                def sdp(x):
                    KD = self.K0 * self.n0
                    return (KD * x) / np.sqrt(KD**2 + x**2)

                sdp_xs = np.linspace(0, rmax, 101)
                sdp_ys = sdp(sdp_xs)
                msk = np.where((sdp_ys < imax)*(sdp_ys > imin))
                sdp_xs, sdp_ys = sdp_xs[msk], sdp_ys[msk]

            else:
                Fs = self.determinant(Cs, field_type=field_type,
                                      mode_type=mode_type,
                                      Ztype_far_left=Z_type,
                                      Ztype_far_right=Z_type)

                def sdp(y):
                    KD = self.K0 * self.n0
                    return KD * np.ones_like(y)

                sdp_ys = np.linspace(imin, imax, 2)
                sdp_xs = sdp(sdp_ys)

            Zs = np.log(np.abs(Fs))

            axmap = ax.contour(Xs, Ys,  Zs, levels=levels,
                               vmin=color_min, vmax=color_max,
                               cmap=cmap, **contourargs)

            sign = 1
            if mode_type == 'guided' and variable == 'Z':
                sign = -1

            ax.plot(sdp_xs, sign * sdp_ys, 'r', lw=1)
            ax.grid(grid)
            ax.set_facecolor(facecolor)

            if colorbar and k == 0:
                plt.colorbar(axmap, pad=pad, shrink=shrink, orientation=orient,
                             anchor=anchor, fraction=colorbar_frac,
                             aspect=colorbar_aspect)
            camera.snap()

        animation = camera.animate(blit=True)
        animation.save(name + '.mp4', fps=fps, bitrate=bitrate, dpi=dpi)

# --------------------------- Interactive Plots  -------------------------------

    def interactive_determinant_plot(self, rmin, rmax, imin, imax,
                                     minwl=1e-6, maxwl=4e-6, Nwl=1000,
                                     rref=100, iref=100, levels=70,
                                     field_type='TE', mode_type='guided',
                                     plot_sdp=True, sdp_sign=-1,
                                     plane='Z', grid=True, facecolor='grey',
                                     figsize=(10, 6), width='1250px',
                                     plot_axis=True, **args):
        plt.close('all')
        fig, ax = plt.subplots(1, figsize=figsize)

        xs, ys = np.linspace(rmin, rmax, rref), np.linspace(imin, imax, iref)

        Xs, Ys = np.meshgrid(xs, ys)
        Cs = Xs + 1j * Ys

        def det_plot(wl, plot_sdp=plot_sdp):
            self.wl = wl
            ax.clear()

            fs = self.determinant(Cs, field_type=field_type,
                                  mode_type=mode_type, plane=plane)
            Vals = np.log(np.abs(fs))
            ax.contour(Xs, Ys, Vals, levels=levels, vmin=-4, vmax=9, **args)

            if plot_sdp:
                if plane == 'Z':
                    sdp_vals = self.sdp(xs, sdp_sign=sdp_sign, plane=plane)
                    sdp_ys = sdp_vals.imag
                    msk = np.where((sdp_ys < imax) * (sdp_ys > imin))
                    ax.plot(xs[msk], sdp_ys[msk], **args)
                elif plane == 'Beta':
                    sdp_vals = self.sdp(ys, sdp_sign=sdp_sign, plane=plane)
                    ax.plot(sdp_vals.real, sdp_vals.imag, **args)
                elif plane == 'Psi':
                    sdp_points = self.sdp(ys, sdp_sign=sdp_sign, plane=plane)
                    sdp_xs, sdp_ys = sdp_points.real, sdp_points.imag
                    msk = np.where((sdp_xs < rmax) * (sdp_xs > rmin))
                    ax.plot(sdp_xs[msk], sdp_ys[msk])
                else:
                    raise ValueError("Plane must be 'Z', 'Beta' or 'Psi'.")

            ax.grid(grid)
            ax.set_facecolor(facecolor)
            if plot_axis:
                ax.axhline(0, color='k', linewidth=.5)
                ax.axvline(0, color='k', linewidth=.5)
            plt.title('Wavelength Dependent Determinant Plot\n\
Current wavelength: %.6e\n' % (wl), fontsize=12)

        step = (maxwl - minwl) / Nwl

        interactive_plot = interactive(det_plot,
                                       wl=FloatSlider(min=minwl, max=maxwl,
                                                      step=step, value=minwl,
                                                      readout_format='.5e',
                                                      layout=Layout(
                                                          width='90%')))
        output = interactive_plot.children[-1]
        output.layout.width = '1250px'
        return interactive_plot

    def interactive_spectral_integrand_1d(self, contour, f0=None,
                                          Lx=None, Rx=None,
                                          min_x=None, max_x=None, Nx=1000,
                                          min_s=None, max_s=None, Ns=1000,
                                          min_z=0, max_z=15, Nz=1000,
                                          field_type='TE', Normalizer=None,
                                          class_A_only=False,
                                          class_B_only=False,
                                          figsize=(11, 4), width='1250px',
                                          part='both', plot_x_axis=True,
                                          plot_y_axis=True,
                                          plot_Z_evanescent=True,
                                          legend_fontsize=12,
                                          **line_args):
        plt.close('all')
        fig, ax = plt.subplots(1, figsize=figsize)

        Cs, contour_type = contour['Cs'], contour['contour_type']

        if contour_type in ['real', 'horizontal', 'sdp']:
            xs = Cs.real
            if max(xs) < self.Z_evanescent.real:
                plot_Z_evanescent = False
        elif contour_type in ['imaginary', 'vertical']:
            plot_Z_evanescent = False
            xs = Cs.imag
        else:
            plot_Z_evanescent = False
            xs = contour['Ss']

        if min_x is None:
            min_x = self.Rhos[0]
        if max_x is None:
            max_x = self.Rhos[-1]

        step_x = (max_x - min_x) / Nx
        step_z = (max_z - min_z) / Nz

        x_slider = FloatSlider(min=min_x, max=max_x, step=step_x,
                               value=(max_x + min_x)/2,
                               readout_format='3f', layout=Layout(width='90%'))

        z_slider = FloatSlider(min=min_z, max=max_z, step=step_z,
                               value=min_z,
                               readout_format='.3f', layout=Layout(width='90%'))

        if f0 is None:

            if min_s is None:
                min_s = self.Rhos[0]
            if max_s is None:
                max_s = self.Rhos[-1]

            step_s = (max_s - min_s) / Ns

            s_slider = FloatSlider(min=min_s, max=max_s, step=step_s,
                                   value=(max_x + min_x)/2,
                                   readout_format='3f',
                                   layout=Layout(width='90%'))

            def int_plot(x, z, s):
                ax.clear()
                fs = self.spectral_integrand(Cs, f0=None, s=s, x=x, z=z,
                                             Lx=Lx, Rx=Rx,
                                             field_type=field_type,
                                             Normalizer=Normalizer,
                                             )
                if part == 'real':
                    fs = fs.real
                elif part == 'imag':
                    fs = fs.imag
                elif part == 'both':
                    fs1 = fs.real
                    fs2 = fs.imag
                elif part == 'norm':
                    fs = np.abs(fs)
                else:
                    raise ValueError('Part must be real,imag, both, or norm.')

                if part != 'both':
                    ax.plot(xs, fs, label=part, **line_args)
                else:
                    ax.plot(xs, fs1, label='real', **line_args)
                    ax.plot(xs, fs2, label='imag', **line_args)

                if plot_x_axis:
                    plt.axhline(0, color='grey', linewidth=1.25)

                if plot_y_axis:
                    plt.axvline(0, color='grey', linewidth=1.25)

                if plot_Z_evanescent:
                    plt.axvline(self.Z_evanescent.real, color='grey',
                                linestyle=':', linewidth=1.25)

                plt.title('$x, s, z$ Dependent 1d Spectral Integrand Plot\n\
    Current x: %.3f, Current s: %.3f, Current z: %.3f' % (x, s, z), fontsize=12)
                plt.legend(fontsize=legend_fontsize)

            interactive_plot = interactive(int_plot, x=x_slider, z=z_slider,
                                           s=s_slider)
        else:

            if Lx is None:
                Lx = self.Rhos[0]
            if Rx is None:
                Rx = self.Rhos[-1]

            if class_A_only and class_B_only:
                raise ValueError('At most one of class_A_only and class_B_only \
can be set to True.')

            if class_A_only:
                alpha_A = self.radiation_transform(Cs, f0=f0, sign='+1',
                                                   Lx=Lx, Rx=Rx,
                                                   field_type=field_type,
                                                   Normalizer=Normalizer)
                alpha_B = None

            elif class_B_only:
                alpha_B = self.radiation_transform(Cs, f0=f0, sign='-1',
                                                   Lx=Lx, Rx=Rx,
                                                   field_type=field_type,
                                                   Normalizer=Normalizer)
                alpha_A = None
            else:
                alpha_A = self.radiation_transform(Cs, f0=f0, sign='+1',
                                                   Lx=Lx, Rx=Rx,
                                                   field_type=field_type,
                                                   Normalizer=Normalizer)
                alpha_B = self.radiation_transform(Cs, f0=f0, sign='-1',
                                                   Lx=Lx, Rx=Rx,
                                                   field_type=field_type,
                                                   Normalizer=Normalizer)

            def int_plot(x, z):
                ax.clear()
                fs = self.spectral_integrand(Cs, f0, x=x, z=z,
                                             Lx=Lx, Rx=Rx,
                                             alpha_A=alpha_A,
                                             alpha_B=alpha_B,
                                             class_A_only=class_A_only,
                                             class_B_only=class_B_only,
                                             field_type=field_type,
                                             Normalizer=Normalizer,
                                             )
                if part == 'real':
                    fs = fs.real
                elif part == 'imag':
                    fs = fs.imag
                elif part == 'both':
                    fs1 = fs.real
                    fs2 = fs.imag
                elif part == 'norm':
                    fs = np.abs(fs)
                else:
                    raise ValueError('Part must be real,imag, both, or norm.')

                if part != 'both':
                    ax.plot(xs, fs, label=part, **line_args)
                else:
                    ax.plot(xs, fs1, label='real', **line_args)
                    ax.plot(xs, fs2, label='imag', **line_args)

                if plot_x_axis:
                    plt.axhline(0, color='grey', linewidth=1.25)

                if plot_y_axis:
                    plt.axvline(0, color='grey', linewidth=1.25)

                if plot_Z_evanescent:
                    plt.axvline(self.Z_evanescent.real, color='grey',
                                linestyle=':', linewidth=1.25)

                plt.title('$x, z$ Dependent 1d Spectral Integrand Plot\n\
    Current x: %.3f, Current z: %.3f' % (x, z), fontsize=12)
                plt.legend(fontsize=legend_fontsize)

            interactive_plot = interactive(int_plot, x=x_slider, z=z_slider)
        output = interactive_plot.children[-1]
        output.layout.width = '1250px'
        return interactive_plot

    def interactive_spectral_integrand_2d(self, rmin, rmax, imin, imax,
                                          f0=None, Lx=None, Rx=None,
                                          min_x=None, max_x=None, Nx=1000,
                                          min_s=None, max_s=None, Ns=1000,
                                          min_z=0, max_z=15, Nz=1000,
                                          min_val=None, max_val=None,
                                          rref=100, iref=100, levels=100,
                                          field_type='TE', Normalizer=None,
                                          class_A_only=False,
                                          class_B_only=False,
                                          plot_sdp=True, sdp_sign=-1,
                                          plot_axis=True, axis_linewidth=.7,
                                          axis_linecolor='k',
                                          grid=True, facecolor='grey',
                                          figsize=(11, 5), width='1250px',
                                          log_abs=True, part='norm',
                                          vmin=-4, vmax=9, cmap='viridis',
                                          plane='Z', **args):

        plt.close('all')
        fig, ax = plt.subplots(1, figsize=figsize)

        xs, ys = np.linspace(rmin, rmax, rref), np.linspace(imin, imax, iref)

        Xs, Ys = np.meshgrid(xs, ys)
        Cs = Xs + 1j * Ys

        if min_x is None:
            min_x = self.Rhos[0]
        if max_x is None:
            max_x = self.Rhos[-1]

        step_x = (max_x - min_x) / Nx
        step_z = (max_z - min_z) / Nz

        x_slider = FloatSlider(min=min_x, max=max_x, step=step_x,
                               value=(max_x + min_x)/2,
                               readout_format='3f', layout=Layout(width='90%'))

        z_slider = FloatSlider(min=min_z, max=max_z, step=step_z,
                               value=min_z,
                               readout_format='.3f', layout=Layout(width='90%'))

        if f0 is None:
            if min_s is None:
                min_s = self.Rhos[0]
            if max_s is None:
                max_s = self.Rhos[-1]

            step_s = (max_s - min_s) / Ns

            s_slider = FloatSlider(min=min_s, max=max_s, step=step_s,
                                   value=(max_x + min_x)/2,
                                   readout_format='3f',
                                   layout=Layout(width='90%'))

            def int_plot(x, z, s, det_power=0, plot_sdp=plot_sdp,
                         log_abs=log_abs):
                ax.clear()
                fs = self.spectral_integrand(Cs, f0=None, x=x, z=z, s=s,
                                             Lx=Lx, Rx=Rx, plane=plane,
                                             class_A_only=False,
                                             class_B_only=False,
                                             field_type=field_type,
                                             Normalizer=Normalizer,
                                             det_power=det_power)
                if part == 'real':
                    fs = fs.real
                elif part == 'imag':
                    fs = fs.imag
                elif part == 'norm':
                    fs = np.abs(fs)
                else:
                    raise ValueError('Part must be real, imag or norm.')
                if max_val is not None:
                    vmax = max_val
                    mask = np.where(fs > max_val)
                    fs[mask] = max_val
                if min_val is not None:
                    vmin = min_val
                    mask = np.where(fs < min_val)
                    fs[mask] = min_val
                if log_abs:
                    fs = np.log(np.abs(fs))

                if plot_axis:
                    if (imin <= 0) * (imax >= 0):
                        ax.axhline(0, color=axis_linecolor,
                                   linewidth=axis_linewidth)
                    if (rmin <= 0) * (rmax >= 0):
                        ax.axvline(0, color=axis_linecolor,
                                   linewidth=axis_linewidth)
                ax.grid(grid)
                ax.set_facecolor(facecolor)

                ax.contour(Xs, Ys, fs, levels=levels,
                           vmin=vmin, vmax=vmax,
                           cmap=cmap, **args)

                if plot_sdp:
                    if plane == 'Z':
                        sdp_vals = self.sdp(xs, sdp_sign=sdp_sign, plane=plane)
                        sdp_xs, sdp_ys = sdp_vals.real, sdp_vals.imag
                        msk = np.where((sdp_ys < imax) * (sdp_ys > imin))
                    elif plane == 'Beta':
                        sdp_points = self.sdp(ys, sdp_sign=sdp_sign,
                                              plane=plane)
                        sdp_xs, sdp_ys = sdp_points.real, sdp_points.imag
                        if (rmin <= self.K0*self.n0)*(rmax >= self.K0*self.n0):
                            ax.plot(sdp_xs, sdp_ys)
                    elif plane == 'Psi':
                        sdp_ys = ys
                        sdp_xs = sdp_sign * np.sign(sdp_ys) * \
                            np.arccos(1 / np.cosh(sdp_ys))
                        msk = np.where((sdp_xs <= rmax) * (sdp_xs >= rmin))
                    else:
                        raise ValueError("Plane must be 'Z', 'Beta' or 'Psi'.")

                    ax.plot(sdp_xs[msk], sdp_ys[msk], **args)

                plt.title('$x, s, z$ Dependent Spectral Integrand Plot\n\
    Current x: %.3f, Current s: %.3f, Current z: %.3f' % (x, s, z), fontsize=12)

            interactive_plot = interactive(int_plot, x=x_slider, z=z_slider,
                                           s=s_slider)

        else:
            if Lx is None:
                Lx = self.Rhos[0]
            if Rx is None:
                Rx = self.Rhos[-1]

            if class_A_only and class_B_only:
                raise ValueError('At most one of class_A_only and class_B_only \
can be set to True.')

            if class_A_only:
                alpha_A = self.radiation_transform(Cs, f0=f0, sign='+1',
                                                   Lx=Lx, Rx=Rx, plane=plane,
                                                   field_type=field_type,
                                                   Normalizer=Normalizer)
                alpha_B = None

            elif class_B_only:
                alpha_B = self.radiation_transform(Cs, f0=f0, sign='-1',
                                                   Lx=Lx, Rx=Rx, plane=plane,
                                                   field_type=field_type,
                                                   Normalizer=Normalizer)
                alpha_A = None
            else:
                alpha_A = self.radiation_transform(Cs, f0=f0, sign='+1',
                                                   Lx=Lx, Rx=Rx, plane=plane,
                                                   field_type=field_type,
                                                   Normalizer=Normalizer)
                alpha_B = self.radiation_transform(Cs, f0=f0, sign='-1',
                                                   Lx=Lx, Rx=Rx, plane=plane,
                                                   field_type=field_type,
                                                   Normalizer=Normalizer)

            def int_plot(x, z, det_power=0, plot_sdp=plot_sdp, log_abs=log_abs):
                ax.clear()
                fs = self.spectral_integrand(Cs, f0=f0, x=x, z=z,
                                             Lx=Lx, Rx=Rx, plane=plane,
                                             alpha_A=alpha_A,
                                             alpha_B=alpha_B,
                                             class_A_only=class_A_only,
                                             class_B_only=class_B_only,
                                             field_type=field_type,
                                             Normalizer=Normalizer,
                                             det_power=det_power)
                if part == 'real':
                    fs = fs.real
                elif part == 'imag':
                    fs = fs.imag
                elif part == 'norm':
                    fs = np.abs(fs)
                else:
                    raise ValueError('Part must be real, imag or norm.')
                if log_abs:
                    fs = np.log(np.abs(fs))

                if plot_axis:
                    if (imin <= 0) * (imax >= 0):
                        ax.axhline(0, color=axis_linecolor,
                                   linewidth=axis_linewidth)
                    if (rmin <= 0) * (rmax >= 0):
                        ax.axvline(0, color=axis_linecolor,
                                   linewidth=axis_linewidth)

                ax.grid(grid)
                ax.set_facecolor(facecolor)

                ax.contour(Xs, Ys, fs, levels=levels,
                           # vmin=vmin, vmax=vmax,
                           cmap=cmap, **args)

                if plot_sdp:
                    if plane == 'Z':
                        sdp_vals = self.sdp(xs, sdp_sign=sdp_sign, plane=plane)
                        sdp_xs, sdp_ys = sdp_vals.real, sdp_vals.imag
                        msk = np.where((sdp_ys < imax) * (sdp_ys > imin))
                    elif plane == 'Beta':
                        sdp_points = self.sdp(ys, sdp_sign=sdp_sign,
                                              plane=plane)
                        sdp_xs, sdp_ys = sdp_points.real, sdp_points.imag
                        if (rmin <= self.K0*self.n0)*(rmax >= self.K0*self.n0):
                            ax.plot(sdp_xs, sdp_ys)
                    elif plane == 'Psi':
                        sdp_ys = ys
                        sdp_xs = sdp_sign * np.sign(sdp_ys) * \
                            np.arccos(1 / np.cosh(sdp_ys))
                        msk = np.where((sdp_xs <= rmax) * (sdp_xs >= rmin))
                    else:
                        raise ValueError("Plane must be 'Z', 'Beta' or 'Psi'.")
                    ax.plot(sdp_xs[msk], sdp_ys[msk], **args)

                plt.title('$x, z$ Dependent Spectral Integrand Plot\n\
    Current x: %.3f, Current z: %.3f' % (x, z), fontsize=12)

            interactive_plot = interactive(int_plot, x=x_slider, z=z_slider)
        output = interactive_plot.children[-1]
        output.layout.width = '1250px'
        return interactive_plot

    def interactive_radiation_mode_plot(self, minZ=.001, maxZ=1, NZ=200,
                                        xs=None, field_type='TE', sign='+1',
                                        minwl=None, maxwl=None, Nwl=300,
                                        figsize=(11, 5), width='1250px',
                                        paper_method=False, part='real',
                                        ylims=None, plot_Rhos=True,
                                        **line_args):
        plt.close('all')
        fig, ax = plt.subplots(1, figsize=figsize)
        if part not in ['real', 'imag', 'norm']:
            raise ValueError("Part must be 'real', 'imag', or 'norm'.")
        if xs is None:
            xs = self.all_Xs

        if minwl is None or maxwl is None:
            minwl, maxwl = self.wl, self.wl

        def F_plot(Z, wl):
            ax.clear()
            self.wl = wl

            if sign == 'both':
                F1 = self.fields_Z(Z, mode_type='radiation',
                                   field_type=field_type,
                                   sign='1')
                F2 = self.fields_Z(Z, mode_type='radiation',
                                   field_type=field_type,
                                   sign='-1')
                ys1, ys2 = F1(xs), F2(xs)
                if part == 'real':
                    ys1, ys2 = ys1.real, ys2.real
                elif part == 'imag':
                    ys1, ys2 = ys1.imag, ys2.imag
                else:
                    ys1, ys2 = np.abs(ys1), np.abs(ys2)
                ax.plot(xs, ys1, **line_args)
                ax.plot(xs, ys2, **line_args)

            else:
                F = self.fields_Z(Z, mode_type='radiation',
                                  field_type=field_type,
                                  sign=sign)
                ys = F(xs)

                if part == 'real':
                    ys = ys.real
                elif part == 'imag':
                    ys = ys.imag
                else:
                    ys = np.abs(ys)
                ax.plot(xs, ys, **line_args)

            ax.axhline(0, color='grey', linewidth=1.25)
            ax.axvline(0, color='grey', linewidth=1.25)

            if ylims is not None:
                ax.set_ylim(ylims[:])

            if plot_Rhos:
                for Rho in self.Rhos[1:-1]:
                    plt.axvline(Rho, linestyle=':', linewidth=1, color='grey')

            plt.title('Z Dependent Radiation Mode Plot\n$Z = %.4e$\n' % Z,
                      fontsize=12)

        Zstep = (maxZ - minZ) / NZ
        Zslider = FloatSlider(min=minZ, max=maxZ, step=Zstep, value=minZ,
                              readout_format='.3e', layout=Layout(width='90%'))

        wlstep = (maxwl - minwl) / Nwl
        wlslider = FloatSlider(min=minwl, max=maxwl, value=minwl, step=wlstep,
                               readout_format='.3e', layout=Layout(width='90%'))

        interactive_plot = interactive(F_plot, Z=Zslider, wl=wlslider)
        output = interactive_plot.children[-1]
        output.layout.width = '1250px'

        return interactive_plot
