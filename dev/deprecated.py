#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 15:31:21 2025

@author: pv
"""

# Tried to do a box contour, but doesn't work for integration techniques
# would need paths to be C1 at interface.
# Deprecated 2/23/25

  def box_contour(self, x_left, x_right, y_bottom, y_top,
                   N_left, N_right=None, N_bottom=None, N_top=None,
                   orientation='cw'):
       if N_right is None:
            N_right = N_left
        if N_bottom is None:
            N_bottom = N_left
        if N_top is None:
            N_top = N_left

        len_horiz, len_vert = abs(x_right - x_left), abs(y_top - y_bottom)

        if orientation == 'cw':

            B = self.horizontal_contour(y_bottom, x_right, x_left, N_bottom,
                                        s_start=0, s_end=len_horiz)

            L = self.vertical_contour(x_left, y_bottom, y_top, N_left,
                                      s_start=len_horiz,
                                      s_end=(len_horiz+len_vert))

            T = self.horizontal_contour(y_top, x_left, x_right, N_top,
                                        s_start=(len_horiz+len_vert),
                                        s_end=(2*len_horiz+len_vert))

            R = self.vertical_contour(x_right, y_top, y_bottom, N_right,
                                      s_start=(2*len_horiz+len_vert),
                                      s_end=(2*len_horiz+2*len_vert))

            Zs = np.concatenate((B['Zs'][:-1], L['Zs'][:-1], T['Zs'][:-1],
                                 R['Zs']))
            Ss = np.concatenate((B['Ss'][:-1], L['Ss'][:-1], T['Ss'][:-1],
                                 R['Ss']))
            dZdS = np.concatenate((B['dZdS'][:-1], L['dZdS'][:-1],
                                   T['dZdS'][:-1], R['dZdS']))

        elif orientation == 'ccw':

            B = self.horizontal_contour(y_bottom, x_left, x_right, N_bottom,
                                        s_start=0, s_end=len_horiz)

            R = self.vertical_contour(x_right, y_bottom, y_top, N_right,
                                      s_start=len_horiz,
                                      s_end=(len_horiz+len_vert))

            T = self.horizontal_contour(y_top, x_right, x_left, N_top,
                                        s_start=(len_horiz+len_vert),
                                        s_end=(2*len_horiz+len_vert))

            L = self.vertical_contour(x_left, y_top, y_bottom, N_left,
                                      s_start=(2*len_horiz+len_vert),
                                      s_end=(2*len_horiz+2*len_vert))

            Zs = np.concatenate((B['Zs'][:-1], R['Zs'][:-1], T['Zs'][:-1],
                                 L['Zs']))
            Ss = np.concatenate((B['Ss'][:-1], R['Ss'][:-1], T['Ss'][:-1],
                                 L['Ss']))
            dZdS = np.concatenate((B['dZdS'][:-1], R['dZdS'][:-1],
                                   T['dZdS'][:-1], L['dZdS']))

        else:
            raise ValueError('Orientation must be cw or ccw.')

        return {'Zs': Zs, 'dZdS': dZdS, 'Ss': Ss}

# Original (non-vectorized) scipy integrator for Propagator class
# Deprecated 2/22/2025


def integrator(selfz, Zs, field_type='TE', Lx=-np.inf,
               Rx=np.inf, sign='+1',
               c0=1, c1=None, paper_method=False, conjugate=False,
               **intargs):
    '''
    Return transform coefficients (alphas) from Zs using scipy
    integration techniques.
    '''

    # Check if field is stored (we still recalculate)
    if Zs in selfz.Zs:
        idx = np.argwhere(Zs == selfz.Zs)[0][0]
        F = selfz.Fs[idx]

    else:
        F = self.fields_Z(Zs, mode_type='radiation',
                          field_type=selfz.field_type,
                          sign=selfz.sign,
                          paper_method=paper_method,
                          c0=c0, c1=c1)

    def integrand(x, conj=True):
        if conj:
            out = selfz.f0(x) * F(x).conjugate()
        else:
            out = selfz.f0(x) * F(x)
        return out

    Int = integrate.quad(integrand, Lx, Rx, args=(conjugate),
                         complex_func=True,
                         **intargs)[0]
    return 1 / 2 * Int


# Our own version of transform from Propagator class
# Deprecated 2/8/2025


def quad_transform(selfz, Zs, field_type='TE', Lx=self.Rhos[0],
                   Rx=self.Rhos[-1], Nx=10, sign='+1',
                   c0=1, c1=None,
                   paper_method=True,
                   pts_per_wl=6, baseN=10, **intargs):
    '''
    Return transform coefficients (alphas) from Zs using simple
    quadrature.

    Parameters
    ----------
    Zs : function
        Input Z propagation constants.
    field_type : str, optional
        Type of radiation field for expansion, either transverse
        electric (TE) or transverse magnetic (TM). The default is
        'TE'.
    sign : str, optional
        Determines even (+1) or odd (-1_ radiation field for
        expansion. The default is '+1'.
    phase : float, optional
        Sets phase of radiation field manually, overriding sign
        keyword.  Resulting radiation modes are typically neither
        even nor odd. The default is None.
    Lx : float or np.inf, optional
        Left x limit for integration. The default is self.Rhos[-1].
    Rx : float or np.inf, optional
        Right x limit for integration. The default is self.Rhos[-1].

    Returns
    -------
    alphas
        Coefficients for radiation field expansion.

    '''
    try:
        len(Zs)
        Zs = np.array(Zs)
    except TypeError:
        Zs = np.array([Zs])

    # Check if field is stored (we still recalculate)
    if Zs in selfz.Zs:
        idx = np.argwhere(Zs == selfz.Zs)[0][0]
        F = selfz.Fs[idx]
    else:
        F = self.fields_Z(Zs, mode_type='radiation',
                          field_type=selfz.field_type,
                          sign=selfz.sign,
                          paper_method=paper_method,
                          c0=c0, c1=c1)

    # Perform quadrature (could add better method, Simpson's etc).

    Xs = np.linspace(Lx, Rx, Nx)
    dX = Xs[1] - Xs[0]
    Int = sum([selfz.f0(x)*F(x).conjugate()*dX for x in Xs])

    return 1 / 2 * Int


# Matrices and Deteriminant (Eigenvalue) Functions in Beta Plane
# Deprecated 2/21/25


    def transfer_matrix(self, Beta, Rho, n_left, n_right, field_type='TE',
                        Ztype_left='standard', Ztype_right='standard',
                        direction='LR'):
        """Matrix giving coefficients of field in next layer from previous.

        This version takes scaled beta inputs and forms Z_left and Z_right
        from them.  This allows one to choose between equivalent forms for
        Z which satisfy Z^2 = K^2 - Beta^2, which can move branch cuts in the
        determinant and help visualize things in the complex beta plane.

        However, as integration for propagating the radiation plane takes place
        over the positive Z axis, this version may eventually be deprecated or
        modified to only take Beta as input when forced by user via keyword
        argument."""

        Beta = np.array(Beta, dtype=complex)

        if direction not in ['RL', 'LR']:
            raise ValueError('Direction must be RL or LR.')

        # swap indices to go other direction
        # Note n_left and n_right should still be provided to function in LR
        # direction.
        if direction == 'RL':
            n_right, n_left = n_left, n_right

        M = np.zeros(Beta.shape + (2, 2), dtype=np.complex128)

        K0 = self.K0

        if Ztype_left == 'standard':
            Z_left = np.sqrt(K0**2 * n_left**2 - Beta**2, dtype=complex)
        elif Ztype_left == 'imag':
            Z_left = 1j * np.sqrt(Beta**2 - K0**2 * n_left**2, dtype=complex)
        else:
            raise ValueError('Ztype must be standard or imag.')

        if Ztype_right == 'standard':
            Z_right = np.sqrt(K0**2 * n_right**2 - Beta**2, dtype=complex)
        elif Ztype_right == 'imag':
            Z_right = 1j * np.sqrt(Beta**2 - K0**2 * n_right**2, dtype=complex)
        else:
            raise ValueError('Ztype must be standard or imag.')

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

    def transmission_matrix(self, Beta, field_type='TE',
                            Ztype_far_left='imag', Ztype_far_right='imag',
                            up_to_region=-1, direction='LR'):
        """Total product of TE transfer matrices."""

        Beta = np.array(Beta, dtype=complex)

        if direction not in ['RL', 'LR']:
            raise ValueError('Direction must be RL or LR.')

        T = np.zeros(Beta.shape + (2, 2), dtype=complex)
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

            if i == 1:
                L, R = Ztype_far_left, 'standard'

            elif i == len(Rhos) - 2:
                L, R = 'standard', Ztype_far_right

            else:
                L, R = 'standard', 'standard'

            T = self.transfer_matrix(Beta, rho, nl, nr,
                                     field_type=field_type,
                                     Ztype_left=L, Ztype_right=R,
                                     direction=direction) @ T
        return T

    def determinant(self, Beta, field_type='TE', mode_type='guided',
                    indices=None, Ztype_far_left='imag',
                    Ztype_far_right='imag', direction='LR'):
        """Eigenvalue function (determinant of matching matrix)."""

        if field_type != 'TE' and field_type != 'TM':
            raise ValueError('Must have field_type either TE or TM.')

        if indices is not None:
            indices = indices
            Ztype_far_left = Ztype_far_left
            Ztype_far_right = Ztype_far_right
            warn('Providing indices to determinant overrides mode type and \
passes Ztype_far_left and Ztype_far_right keywords to transmission matrix. This\
 allows for exploration, but may put branch cuts in unexpected places.')

        else:
            if mode_type == 'guided':
                indices = [1, 1]

            elif mode_type == 'leaky':
                indices = [0, 0]
            else:
                raise ValueError('Mode type must be guided or leaky to plot\
     determinant. For radiation modes use transmission_matrix_TE/TM')

        T = self.transmission_matrix(Beta, field_type=field_type,
                                     Ztype_far_left=Ztype_far_left,
                                     Ztype_far_right=Ztype_far_right,
                                     direction=direction)

        return T[..., indices[0], indices[1]]

    def transmission_determinant(self, beta, field_type='TE',
                                 up_to_region=-1,
                                 direction='LR'):
        '''Determinant of transmission matrix.'''
        beta = np.array(beta, dtype=complex)
        Z0 = self.Z_from_Beta(beta, n=self.ns[0])
        Zd = self.Z_from_Beta(beta, n=self.ns[up_to_region])
        base = Z0 / Zd
        if field_type == 'TM':
            base *= self.ns[up_to_region]**2 / self.ns[0]**2
        if direction == 'RL':
            base = 1 / base
        return base

# ---------- Building Fields from Propagation Constants in Beta Plane ---------

    def coefficients(self, Beta, c0=1, c1=None, field_type='TE',
                     mode_type='guided', sign='+1', paper_method=True,
                     up_to_region=-1, rounding=12):
        """Return field coefficients given propagation constant Beta."""

        if field_type != 'TE' and field_type != 'TM':
            raise ValueError('Must have field_type either TE or TM.')

        # Single scalar inputs need to be given at least a single dimension
        try:
            len(Beta)
            Beta = np.array(Beta, dtype=complex)
        except TypeError:
            Beta = np.array([Beta], dtype=complex)

        # Set up initial vector array M0.
        # This will contain initial vector for each coefficient array,
        # and also be overwritten to store the new coefficients
        # as we apply transfer matrix.
        M0 = np.zeros(Beta.shape + (2, 1), dtype=complex)

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
            # Beta_one_sided = self.Beta_from_Z(self.Z_one_sided)

            # if Beta >= Beta_one_sided:
            #     one_sided = True
            # else:
            one_sided = False

            A = np.sqrt(c0/(2*np.pi))

            if len(self.ns) > 1:
                # print('Multilayer case.')
                M = self.transmission_matrix(Beta, field_type=field_type,
                                             Ztype_far_left=Ztype_far_left,
                                             Ztype_far_right=Ztype_far_right)
                if paper_method:
                    r1 = -M[..., 1, 0] / M[..., 1, 1]

                    if one_sided:
                        # print('one sided rad mode')
                        phase_term = np.sqrt(r1.conjugate(), dtype=complex)
                        C = A * phase_term
                        M0[..., :, 0] = np.array(
                            [C, C.conjugate()], dtype=complex).T
                    else:
                        # Check if Ztypes will affect determinant below
                        FT = field_type
                        detM = self.transmission_determinant(Beta,
                                                             field_type=FT)

                        frac = (-M[..., 1, 0] * detM) / M[..., 0, 1]

                        b = int(sign) * np.sqrt(frac, dtype=complex)

                        t2 = 1 / M[..., 1, 1]

                        phase_term = 1 / np.sqrt(r1 + b * t2, dtype=complex)

                        C = A * phase_term
                        M0[..., :, 0] = np.array(
                            [C, C.conjugate()], dtype=complex).T

                else:
                    r1 = -M[..., 1, 0] / M[..., 1, 1]

                    if one_sided:
                        # print('one sided rad mode')
                        phase_term = np.sqrt(r1.conjugate(), dtype=complex)
                        C = A * phase_term
                        M0[..., :, 0] = np.array(
                            [C, C.conjugate()], dtype=complex).T
                    else:
                        # Check if Ztypes will affect determinant below
                        FT = field_type
                        detM = self.transmission_determinant(Beta,
                                                             field_type=FT)

                        frac = (-M[..., 1, 0] * detM) / M[..., 0, 1]

                        b = int(sign) * np.sqrt(frac, dtype=complex)

                        factor = M[..., 0, 0].conj() + b - M[..., 0, 1].conj()

                        M0[..., :, 0] = (np.array([b.conj() - M[..., 0, 1],
                                                  M[..., 0, 0]])*factor).T

                        factor2 = np.sqrt(4 * M0[..., 0, 0] * M0[..., 1, 0],
                                          dtype=complex)
                        M0 *= np.sqrt(2/np.pi) * 1 / factor2

            else:
                # print('Single layer case.')
                if int(sign) == 1:
                    phase = 0
                else:
                    phase = np.pi/2
                phase_term = np.exp(1j*phase)
                C = c0 * phase_term
                M0[..., :, 0] = np.array([C, C.conjugate()], dtype=complex).T

            if c1 is not None:
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

        Coeffs = np.zeros(Beta.shape + (2, len(Rhos)+up_to_region),
                          dtype=complex)

        # set first vectors in each coefficient array
        Coeffs[..., :, 0] = M0[..., :, 0]

        for i in range(1, len(Rhos)+up_to_region):
            nl, nr = ns[i-1], ns[i]
            Rho = Rhos[i]
            if i == 1:
                L, R = Ztype_far_left, 'standard'
            elif i == len(Rhos)-2:
                L, R = 'standard', Ztype_far_right
            else:
                L, R = 'standard', 'standard'
            T = self.transfer_matrix(Beta, Rho, nl, nr,
                                     field_type=field_type,
                                     Ztype_left=L,
                                     Ztype_right=R)

            M0 = (T @ M0)  # apply T to vectors
            Coeffs[..., :, i] = M0[..., :, 0]  # update coefficient arrays

        if len(Beta) == 1:
            Coeffs = Coeffs[0]
        # Round to avoid false blowup, noted in guided modes.
        # Fundamental had lower error and rounding=16 worked, but HOMs
        # had more noise and required rounding=12
        Coeffs = np.round(Coeffs, decimals=rounding)

        # Check for correct coefficients if mode type is guided or leaky
        if mode_type in ['guided', 'leaky'] and len(Beta) == 1:
            if Coeffs.T[-1, check_idx] != 0:
                warn(message='Provided mode type %s, but coefficients in outer \
region do not align with this. User may wish to check supplied \
propagation constant and/or rounding parameter.' % mode_type)

        return Coeffs

    def regional_field(self, Beta, index, coeffs, Ztype='standard'):
        """Return field on one region of fiber (for non-unified fields)."""

        A, B = coeffs[:]
        K0 = self.K0
        n = self.ns[index]

        if Ztype == 'standard':
            Z = np.sqrt(K0**2 * n**2 - Beta**2, dtype=complex)
        elif Ztype == 'imag':
            Z = 1j * np.sqrt(Beta**2 - K0**2 * n**2, dtype=complex)
        else:
            raise ValueError('Ztype must be standard or imag.')

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

    def fields(self, Beta, c0=1, c1=None, field_type='TE',
               mode_type='guided', sign='+1', single_function=True,
               rounding=12, paper_method=True):
        '''Return fields.'''

        # Give Beta at least one dimension (for vectorization)
        try:
            len(Beta)
            Beta = np.array(Beta)
        except TypeError:
            Beta = np.array([Beta])

        M = self.coefficients(Beta, c0=c0, c1=c1, mode_type=mode_type,
                              field_type=field_type,  sign=sign,
                              rounding=rounding,
                              paper_method=paper_method).T

        if mode_type == 'guided':
            Ztype_far_left = 'imag'
            Ztype_far_right = 'imag'

        elif mode_type == 'leaky':
            Ztype_far_left = 'imag'
            Ztype_far_right = 'imag'

        elif mode_type == 'radiation':
            Ztype_far_left = 'standard'
            Ztype_far_right = 'standard'

        Ztypes = ['standard' for i in range(len(self.Rhos)-1)]
        Ztypes[0], Ztypes[-1] = Ztype_far_left, Ztype_far_right

        # Get list of functions, one for each region
        Fs = []
        for i in range(len(self.Rhos)-1):
            Fs.append(self.regional_field(Beta, i, M[i], Ztype=Ztypes[i]))

        if not single_function:
            # Return list of functions corresponding to regions of slab
            return Fs

        else:

            # Return single piecewise defined function
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

    def evaluate_fields(self, Beta, x, z=0, field_type='TE',
                        mode_type='radiation', sign='+1',
                        paper_method=True):
        '''Return value of field with propagation constant Beta at x, z.'''
        j = self.region_indices(x)
        if np.isnan(j):
            raise ValueError('Source point is outside computational domain.')
        Zj = np.array(self.Z_from_Beta(Beta, n=self.ns[j]))
        Cs = self.coefficients(Beta, up_to_region=j, mode_type=mode_type,
                               field_type=field_type, sign=sign,
                               paper_method=paper_method)
        return (Cs[..., 0, j] * np.exp(1j * Zj * x) +
                Cs[..., 1, j] * np.exp(-1j * Zj * x)) * np.exp(1j * Beta * z)
