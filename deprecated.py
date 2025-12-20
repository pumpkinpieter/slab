#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 12:59:17 2025

@author: pv
"""


def Zi_from_Z0(self, Z0, ni):
    '''Get Z on region i (with refractive index ni) from Z0.'''
    if ni == self.n0:
        return Z0
    else:
        return np.sqrt((ni ** 2 - self.n0 ** 2) * self.K0 ** 2 + Z0 ** 2,
                       dtype=complex)


def region_index(self, x, continuity='left'):
    '''Return region index in which (non-dimensional) x lies.  Points
    outside of computational domain on left get index j=0, on right
    get index j=len(A.Rhos)-1'''
    for j, Rho in enumerate(self.Rhos[1:]):
        if continuity == 'left':
            if x <= Rho:
                return j
        elif continuity == 'right':
            if x < Rho:
                return j
        else:
            raise ValueError('Continuity must be set to left or right.')
    return j


def CL(self, beta):
    """Get confinement loss from (non-scaled) beta."""
    CL = 20 * beta.imag/np.log(10)
    return CL

    def coefficients(self, C, c0=1, c1=None, field_type='TE',
                     mode_type='guided', sign='+1', paper_method=False,
                     up_to_region=-1, rounding=12, plane='Z'):
        """Return field coefficients given propagation constant Beta."""

        if field_type != 'TE' and field_type != 'TM':
            raise ValueError('Must have field_type either TE or TM.')

        # Single scalar inputs need to be given at least a single dimension
        try:
            len(C)
            C = np.array(C, dtype=complex)
        except ValueError:
            C = np.array([C], dtype=complex)

        # Set up initial vector array M0.
        # This will contain initial vector for each coefficient array,
        # and also be overwritten to store the new coefficients
        # as we apply transfer matrix.
        M0 = np.zeros(C.shape + (2, 1), dtype=complex)

        if mode_type == 'guided':
            check_idx = 1
            M0[..., :, 0] = np.array([0, c0])
            Ztype_far_left = 'imag'
            Ztype_far_right = 'imag'

        elif mode_type == 'leaky':
            check_idx = 0
            M0[..., :, 0] = np.array([c0, 0])
            Ztype_far_left = 'imag'
            Ztype_far_right = 'imag'

        elif mode_type == 'radiation':
            Ztype_far_left = 'standard'
            Ztype_far_right = 'standard'

            A = np.sqrt(c0/(2*np.pi))

            M = self.transmission_matrix(C, plane=plane,
                                         field_type=field_type,
                                         Ztype_far_left=Ztype_far_left,
                                         Ztype_far_right=Ztype_far_right)
            if len(self.ns) > 1:
                # method = radiation_normalization_method
                # N = self.radiation_normalization_class(method)()
                # M0 = N.normalization(Z, sign=sign, ft=field_type)

                if paper_method:

                    A = np.sqrt(c0/(2*np.pi))

                    if c1 is not None and c0 == 0:
                        A = np.sqrt(c1/(2*np.pi))

                    r1 = -M[..., 1, 0] / M[..., 1, 1]
                    FT = field_type
                    detM = self.transmission_determinant(C, plane=plane,
                                                         field_type=FT)
                    t2 = 1 / (M[..., 1, 1] * detM)
                    frac = (-M[..., 1, 0] * detM) / M[..., 0, 1]

                    b = int(sign) * np.sqrt(frac, dtype=complex)

                    C = A / np.sqrt(r1 + b * t2, dtype=complex)

                    M0[..., 0, :] = C[..., np.newaxis]
                    M0[..., 1, :] = C[..., np.newaxis].conjugate()

                else:
                    FT = field_type
                    detM = self.transmission_determinant(C, plane=plane,
                                                         field_type=FT)

                    frac = (-M[..., 1, 0] * detM) / M[..., 0, 1]

                    b = int(sign) * np.sqrt(frac, dtype=complex)

                    C1 = (b.conj() - M[..., 0, 1])
                    C2 = M[..., 0, 0]

                    M0[..., 0, :] = C1[..., np.newaxis]
                    M0[..., 1, :] = C2[..., np.newaxis]

                    factor2 = np.sqrt(M0[..., 0, :] * M0[..., 1, :],
                                      dtype=complex)[..., np.newaxis]

                    M0 *= np.sqrt(1 / (2 * np.pi)) * 1 / factor2

            else:
                if int(sign) == 1:
                    phase = 0
                else:
                    phase = np.pi/2
                phase_term = np.exp(1j*phase)
                C = c0 * phase_term
                M0[..., :, 0] = np.array([C, C.conjugate()], dtype=complex).T
                M0 *= np.sqrt(c0/(2*np.pi))

            if c1 is not None:
                # print('overriding paper, setting using c0 and c1 provided.')
                M0[..., :, 0] = np.array([c0, c1], dtype=complex).T

                if int(sign) == -1:
                    inds = np.arange(len(M.shape))
                    inds[-1], inds[-2] = inds[-2], inds[-1]
                    J = np.array([[0, 1], [-1, 0]])
                    S = np.array([[0, 1], [1, 0]])
                    M0 = 1j*J @ (S + M.transpose(inds) @ S @ M) @ M0

        else:
            raise ValueError('Mode type must be guided, leaky or radiation.')

        Rhos = self.Rhos
        ns = self.ns

        if up_to_region >= 0:
            up_to_region = up_to_region - len(Rhos) + 1

        Coeffs = np.zeros(C.shape + (2, len(Rhos)+up_to_region),
                          dtype=complex)

        # set first vectors in each coefficient array
        Coeffs[..., :, 0] = M0[..., :, 0]

        for i in range(1, len(Rhos)+up_to_region):
            nl, nr = ns[i-1], ns[i]
            Rho = Rhos[i]
            T = self.transfer_matrix(C, Rho, nl, nr,
                                     field_type=field_type,
                                     plane=plane)

            M0 = (T @ M0)  # apply T to vectors
            Coeffs[..., :, i] = M0[..., :, 0]  # update coefficient arrays

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

# ------------------ Working in Z Plane Directly -------------------------

   def transfer_matrix_Z(self, Z0, Rho, n_left, n_right, field_type='TE',
                          direction='LR', stripped=False):
        """Matrix giving coefficients of field in next layer from previous.

        This version takes non-dimensionalized Z0 inputs."""

        Z0 = np.array(Z0, dtype=complex)

        if direction not in ['RL', 'LR']:
            raise ValueError('Direction must be RL or LR.')

        # swap indices to go other direction
        # Note n_left and n_right should still be provided to function in LR
        # direction.
        if direction == 'RL':
            n_right, n_left = n_left, n_right

        M = np.zeros(Z0.shape + (2, 2), dtype=np.complex128)

        Z_left = self.Zi_from_Z0(Z0, n_left)
        Z_right = self.Zi_from_Z0(Z0, n_right)

        Exp_minus = 1j * (Z_right - Z_left) * Rho
        Exp_plus = 1j * (Z_right + Z_left) * Rho

        Ymat = np.zeros_like(M)
        Y = 1 / Z_right
        Ymat[..., 0, :] = np.array([Y.T, Y.T]).T
        Ymat[..., 1, :] = np.array([Y.T, Y.T]).T

        if field_type == 'TM':
            scalar = 1 / (2 * n_left**2)
            Ymat = scalar * Ymat
            A_minus = Z_right * n_left**2 - Z_left * n_right**2
            A_plus = Z_right * n_left**2 + Z_left * n_right**2

        elif field_type == 'TE':
            scalar = 1 / 2
            Ymat = scalar * Ymat
            A_minus = Z_right - Z_left
            A_plus = Z_right + Z_left
        else:
            raise ValueError('Field type must be TE or TM.')

        M[..., 0, :] = np.array([(np.exp(-Exp_minus) * A_plus).T,
                                 (np.exp(-Exp_plus) * A_minus).T]).T

        M[..., 1, :] = np.array([(np.exp(Exp_plus) * A_minus).T,
                                 (np.exp(Exp_minus) * A_plus).T]).T
        if stripped:
            return scalar * M
        return Ymat * M

    def transmission_matrix_Z(self, Z0, field_type='TE', up_to_region=-1,
                              direction='LR', stripped=False):
        """Total product of TE transfer matrices."""

        Z0 = np.array(Z0, dtype=complex)

        if direction not in ['RL', 'LR']:
            raise ValueError('Direction must be RL or LR.')

        T = np.zeros(Z0.shape + (2, 2), dtype=complex)
        T[..., :, :] = np.eye(2, dtype=complex)

        Rhos = self.Rhos
        ns = self.ns

        if up_to_region >= 0:
            up_to_region = up_to_region - len(Rhos) + 1

        enum = range(1, len(Rhos)+up_to_region)

        if direction == 'RL':
            enum = reversed(enum)

        for i in enum:

            nl, nr = ns[i-1], ns[i]
            rho = Rhos[i]

            T = self.transfer_matrix_Z(Z0, rho, nl, nr,
                                       field_type=field_type,
                                       direction=direction,
                                       stripped=stripped) @ T
        return T

    def transmission_determinant_Z(self, Z, field_type='TE',
                                   up_to_region=-1,
                                   direction='LR'):
        Z = np.array(Z, dtype=complex)
        Z0 = self.Zi_from_Z0(Z, ni=self.ns[0])
        Zd = self.Zi_from_Z0(Z, ni=self.ns[up_to_region])
        base = Z0 / Zd
        if field_type == 'TM':
            base *= self.ns[up_to_region]**2 / self.ns[0]**2
        if direction == 'RL':
            base = 1 / base
        return base

    def determinant_Z(self, Z, field_type='TE', mode_type='guided',
                      direction='LR', stripped=False,
                      radiation_normalization_method='ours',
                      sign=1):
        """Eigenvalue function (formerly: determinant of matching matrix, hence
        nomenclature)."""

        if field_type != 'TE' and field_type != 'TM':
            raise ValueError('Field_type must be TE or TM.')

        M = self.transmission_matrix_Z(Z, field_type=field_type,
                                       direction=direction,
                                       stripped=stripped)
        if mode_type == 'guided':
            return M[..., 1, 1]

        elif mode_type == 'leaky':
            return M[..., 0, 0]

        elif mode_type == 'radiation':
            method = radiation_normalization_method
            N = self.radiation_normalization_class(method)()
            return N.pole_locations(Z, sign=sign, ft=field_type)
        else:
            raise ValueError('Mode type must be guided, leaky or radiation.')

    def coefficients_Z(self, Z, c0=1, c1=None, field_type='TE',
                       mode_type='guided', sign='+1', paper_method=False,
                       up_to_region=-1, rounding=12,
                       radiation_normalization_method='ours',
                       ):
        """Return field coefficients given propagation constant Z."""

        if field_type != 'TE' and field_type != 'TM':
            raise ValueError('Must have field_type either TE or TM.')

        # Single scalar inputs need to be given at least a single dimension
        try:
            len(Z)
            Z = np.array(Z, dtype=complex)
        except TypeError:
            Z = np.array([Z], dtype=complex)

        # Set up initial vector array M0.
        # This will contain initial vector for each coefficient array,
        # and also be overwritten to store the new coefficients
        # as we apply transfer matrix.
        M0 = np.zeros(Z.shape + (2, 1), dtype=complex)

        if mode_type == 'guided':
            check_idx = 1
            M0[..., :, 0] = np.array([0, c0])

        elif mode_type == 'leaky':
            check_idx = 0
            M0[..., :, 0] = np.array([c0, 0])

        elif mode_type == 'radiation':

            if len(self.ns) > 1:
                # method = radiation_normalization_method
                # N = self.radiation_normalization_class(method)()
                # M0 = N.normalization(Z, sign=sign, ft=field_type)

                if paper_method:
                    M = self.transmission_matrix_Z(Z, field_type=field_type)

                    A = np.sqrt(c0/(2*np.pi))

                    if c1 is not None and c0 == 0:
                        A = np.sqrt(c1/(2*np.pi))

                    r1 = -M[..., 1, 0] / M[..., 1, 1]
                    FT = field_type
                    detM = self.transmission_determinant_Z(Z, field_type=FT)
                    t2 = 1 / (M[..., 1, 1] * detM)
                    frac = (-M[..., 1, 0] * detM) / M[..., 0, 1]

                    b = int(sign) * np.sqrt(frac, dtype=complex)

                    C = A / np.sqrt(r1 + b * t2, dtype=complex)

                    M0[..., 0, :] = C[..., np.newaxis]
                    M0[..., 1, :] = C[..., np.newaxis].conjugate()

                else:
                    M = self.transmission_matrix_Z(Z, field_type=field_type)

                    FT = field_type
                    detM = self.transmission_determinant_Z(Z,
                                                           field_type=FT)

                    frac = (-M[..., 1, 0] * detM) / M[..., 0, 1]

                    b = int(sign) * np.sqrt(frac, dtype=complex)

                    C1 = (b.conj() - M[..., 0, 1])
                    C2 = M[..., 0, 0]

                    M0[..., 0, :] = C1[..., np.newaxis]
                    M0[..., 1, :] = C2[..., np.newaxis]

                    factor2 = np.sqrt(M0[..., 0, :] * M0[..., 1, :],
                                      dtype=complex)[..., np.newaxis]

                    M0 *= np.sqrt(1 / (2 * np.pi)) * 1 / factor2

            else:
                if int(sign) == 1:
                    phase = 0
                else:
                    phase = np.pi/2
                phase_term = np.exp(1j*phase)
                C = c0 * phase_term
                M0[..., :, 0] = np.array([C, C.conjugate()], dtype=complex).T
                M0 *= np.sqrt(c0/(2*np.pi))

            if c1 is not None:
                M = self.transmission_matrix_Z(Z, field_type=field_type)

                # print('overriding paper, setting using c0 and c1 provided.')
                M0[..., :, 0] = np.array([c0, c1], dtype=complex).T

                if int(sign) == -1:
                    inds = np.arange(len(M.shape))
                    inds[-1], inds[-2] = inds[-2], inds[-1]
                    J = np.array([[0, 1], [-1, 0]])
                    S = np.array([[0, 1], [1, 0]])
                    M0 = 1j*J @ (S + M.transpose(inds) @ S @ M) @ M0

                # factor = (np.linalg.norm((M @ M0)[..., :, 0],
                #                          axis=1)**2
                #           + np.linalg.norm((M0)[..., :, 0],
                #                            axis=1)**2)**.5

                # M0 *= np.sqrt(2/np.pi)
        else:
            raise ValueError('Mode type must be guided, leaky or radiation.')

        Rhos = self.Rhos
        ns = self.ns

        if up_to_region >= 0:
            up_to_region = up_to_region - len(Rhos) + 1

        Coeffs = np.zeros(Z.shape + (2, len(Rhos)+up_to_region),
                          dtype=complex)

        # set first vectors in each coefficient array
        Coeffs[..., :, 0] = M0[..., :, 0]

        for i in range(1, len(Rhos)+up_to_region):
            nl, nr = ns[i-1], ns[i]
            Rho = Rhos[i]
            T = self.transfer_matrix_Z(Z, Rho, nl, nr,
                                       field_type=field_type)

            M0 = (T @ M0)  # apply T to vectors
            Coeffs[..., :, i] = M0[..., :, 0]  # update coefficient arrays

        if len(Z) == 1:
            Coeffs = Coeffs[0]
        # Round to avoid false blowup, noted in guided modes.
        # Fundamental had lower error and rounding=16 worked, but HOMs
        # had more noise and required rounding=12
        Coeffs = np.round(Coeffs, decimals=rounding)

        # Check for correct coefficients if mode type is guided or leaky
        if mode_type in ['guided', 'leaky'] and len(Z) == 1:
            if Coeffs.T[-1, check_idx] != 0:
                warn(message='Provided mode type %s, but coefficients in outer \
region do not align with this. User may wish to check supplied \
propagation constant and/or rounding parameter.' % mode_type)

        return Coeffs

    def regional_field_Z(self, Z, index, coeffs):
        """Return field on one region of fiber (for non-unified fields)."""

        A, B = coeffs[:]
        n = self.ns[index]
        Beta = self.Beta_from_Zi(Z, ni=n)

        def F(xs, zs=None):

            try:
                len(xs)
                xs = np.array(xs)
            except TypeError:
                xs = np.array([xs])

            if len(xs.shape) > 1:
                raise ValueError('Please provide single dimension arrays for \
xs and zs to levarage product nature of fields.')

            ys = (A * np.exp(1j*Z*xs) + B * np.exp(-1j*Z*xs))

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

    def fields_Z(self, Z, c0=1, c1=None, field_type='TE',
                 mode_type='guided', sign='+1', single_function=True,
                 rounding=12, paper_method=False, user_coeffs=None,
                 radiation_normalization_method='ours'):
        '''Return fields.'''

        # Give Beta at least one dimension (for vectorization)
        try:
            len(Z)
            Z = np.array(Z)
        except TypeError:
            Z = np.array([Z])

        if user_coeffs is not None:
            M = user_coeffs
        else:
            method = radiation_normalization_method
            M = self.coefficients_Z(Z, c0=c0, c1=c1, mode_type=mode_type,
                                    field_type=field_type,  sign=sign,
                                    rounding=rounding,
                                    paper_method=paper_method,
                                    radiation_normalization_method=method).T

        Beta = self.Beta_from_Zi(Z, ni=self.n0)
        Zs = np.array([self.Zi_from_Z0(Z, n) for n in self.ns])

        # Get list of functions, one for each region
        Fs = []
        for i in range(len(self.Rhos)-1):
            Fs.append(self.regional_field_Z(Zs[i], i, M[i]))

        if not single_function:
            # Return list of functions corresponding to regions of slab
            return Fs

        else:

            # Return single piecewise defined function
            def F(xs, zs=0):

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

                try:
                    len(zs)
                    zs = np.array(zs)
                except TypeError:
                    zs = np.array([zs])
                if len(zs.shape) > 1:
                    raise ValueError('Please provide single dimension \
arrays for xs and zs to levarage product nature of fields.')
                ys = np.outer(np.exp(1j * Beta * zs), ys)
                if len(zs) == 1:
                    # Reduce dimension for only 1 z value to facilitate plotting
                    ys = ys[0]
                return ys

            return F

# ----------------------- Integration and Normalization -----------------------

    def integrate_region(self, Beta, region_index, region_coeffs,
                         field_type='TE', Ztype='standard', mode_type='guided'):

        A, B = region_coeffs[:]
        K0 = self.K0
        n = self.ns[region_index]
        L, R = self.Rhos[region_index], self.Rhos[region_index+1]

        if Ztype == 'standard':
            Z = np.sqrt(K0**2 * n**2 - Beta**2, dtype=complex)
        elif Ztype == 'imag':
            Z = 1j * np.sqrt(Beta**2 - K0**2 * n**2, dtype=complex)
        else:
            raise ValueError('Ztype must be standard or imag.')

        if np.abs(Z.imag) <= 1e-10 and Z.real != 0:
            lsum = (np.abs(A)**2 + np.abs(B)**2) * (L - R)
            rsum = A * B.conjugate() * (np.exp(2*1j*Z*R) - np.exp(2*1j*Z*L)) +\
                A.conjugate() * B * (np.exp(-2*1j*Z*R) - np.exp(-2*1j*Z*L))
            return lsum + 1/(2*1j*Z) * rsum
        elif np.abs(Z.real) <= 1e-10 and Z.imag != 0:
            rsum = A * B.conjugate() * (np.exp(2*1j*Z*R) - np.exp(2*1j*Z*L)) +\
                A.conjugate() * B * (np.exp(-2*1j*Z*R) - np.exp(-2*1j*Z*L))
            return lsum + 1/(2*1j*Z) * rsum
        else:
            pass

    def integrate(self, L, R, field_type='TE', mode_type='guided'):

        locations = self.condition_list([L, R])
        if L is None and R is None:
            pass
        elif L is None and R is not None:
            pass
        else:
            pass

        return locations
    

    def evaluate_fields_Z(self, Z, x, z=0, field_type='TE',
                          mode_type='radiation', sign='+1',
                          paper_method=False):
        '''Return value of field with propagation constant Zj(Z) at x, z=0.
        Vectorized for Z.'''
        j = self.region_index(x)
        if np.isnan(j):
            raise ValueError('Source point is outside computational domain.')
        Zj = np.array(self.Zi_from_Z0(Z, self.ns[j]))
        Beta = self.Beta_from_Zi(Z, ni=self.n0)
        Cs = self.coefficients_Z(Z, up_to_region=j, mode_type=mode_type,
                                 field_type=field_type, sign=sign,
                                 paper_method=paper_method)
        return (Cs[..., 0, j] * np.exp(1j * Zj * x) +
                Cs[..., 1, j] * np.exp(-1j * Zj * x)) * np.exp(1j * Beta * z)

    def evaluate_dFdZ_approx(self, Z_base, x, z=0, dZ=1e-6, field_type='TE',
                             sign='+1', paper_method=False, order=2):

        FZ0 = self.evaluate_fields_Z(Z_base, x, z=z, field_type=field_type,
                                     mode_type='radiation',
                                     paper_method=paper_method)

        FdZ = self.evaluate_fields_Z(Z_base + dZ, x, z=z, field_type=field_type,
                                     mode_type='radiation',
                                     paper_method=paper_method)

        if order == 2:
            FdZ2 = self.evaluate_fields_Z(Z_base + 2 * dZ, x, z=z,
                                          field_type=field_type,
                                          mode_type='radiation',
                                          paper_method=paper_method)
            return (-3 * FZ0 + 4 * FdZ - FdZ2) / (2 * dZ)

        elif order == 1:
            return (FdZ - FZ0) / dZ

        else:
            raise NotImplementedError('Only orders 1 and 2 implemented.')

    def radiation_poles_as_zeros(self, Zs, field_type='TE', sign=None,
                                 paper_method=False):
        '''Find the pole locations (as zeros) of the extension of the radiation
        modes to the complex plane.

        This function finds pole locations for both classes of radiation modes
        simultaneously by default.  To find them only for a particular class
        pass 1 or -1 to the sign argument.'''

        if sign is None:
            Cs1 = self.coefficients_Z(Zs, field_type=field_type,
                                      mode_type='radiation',
                                      paper_method=paper_method,
                                      sign=1,
                                      up_to_region=0,
                                      )[..., 0]

            Cs2 = self.coefficients_Z(Zs, field_type=field_type,
                                      mode_type='radiation',
                                      paper_method=paper_method,
                                      sign=-1,
                                      up_to_region=0,
                                      )[..., 0]

            sum_abs1 = np.abs(Cs1[..., 0])**2 + np.abs(Cs1[..., 1])**2
            sum_abs2 = np.abs(Cs2[..., 0])**2 + np.abs(Cs2[..., 1])**2
            return 1 / (sum_abs1 * sum_abs2)

        elif sign == 1:
            Cs1 = self.coefficients_Z(Zs, field_type=field_type,
                                      mode_type='radiation',
                                      paper_method=paper_method,
                                      sign=1,
                                      up_to_region=0,
                                      )[..., 0]
            sum_abs1 = np.abs(Cs1[..., 0])**2 + np.abs(Cs1[..., 1])**2
            return 1 / sum_abs1

        elif sign == -1:
            Cs2 = self.coefficients_Z(Zs, field_type=field_type,
                                      mode_type='radiation',
                                      paper_method=paper_method,
                                      sign=-1,
                                      up_to_region=0,
                                      )[..., 0]

            sum_abs2 = np.abs(Cs2[..., 0])**2 + np.abs(Cs2[..., 1])**2
            return 1 / sum_abs2

        else:
            raise ValueError("Sign must be None, 1 or -1.")
