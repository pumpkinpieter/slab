#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:50:29 2025

@author: pv
"""

            def propagate(selfz, xs, zs, method='simpsons'):
                '''
                Propagate radiation field of input function selfz.f0 (or exact
                transform).

                Option to only return evanescent or propagating portions of the
                field.

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
                Nz = len(selfz.Zs)
                alphas, Fs, dZdS = selfz.alphas, selfz.Fs, selfz.dZdS
                dS = selfz.dS

                if method == 'left_endpoint':
                    ys = np.sum([alphas[i] * Fs[i](xs, zs) * dZdS[i] *
                                 dS[i] for i in range(Nz-1)])

                elif method == 'right_endpoint':
                    ys = np.sum([alphas[i+1] * Fs[i+1](xs, zs) * dZdS[i+1] *
                                 dS[i] for i in range(Nz-1)])

                elif method == 'trapezoid':
                    ys = alphas[0] * Fs[0](xs, zs) * dZdS[0] * dS[0]
                    ys += np.sum([alphas[i] * Fs[i](xs, zs) * dZdS[i] *
                                 (dS[i] + dS[i-1]) for i in range(1, Nz-1)])
                    ys += alphas[-1] * Fs[-1](xs, zs) * dZdS[-1] * dS[-1]
                    ys *= 1 / 2

                elif method == 'simpsons':
                    if len(selfz.Zs) % 2 != 1:
                        raise ValueError('Contour must have odd number of \
points points (even number of intervals) for Simpsons rule.')

                    set_dS = set(np.round(selfz.dS, decimals=13))
                    if len(set_dS) != 1:
                        raise ValueError('Simpsons rule requires equally \
spaced spaced intervals.')

                    dS = set_dS.pop()
                    upper = int((len(selfz.Zs)-1)/2) + 1
                    ys = alphas[0] * Fs[0](xs, zs) * dZdS[0]
                    ys += 4 * np.sum([alphas[2*i-1] * Fs[2*i-1](xs, zs) *
                                      dZdS[2*i-1] for i in range(1, upper)])
                    ys += 2 * np.sum([alphas[2*i] * Fs[2*i](xs, zs) *
                                      dZdS[2*i] for i in range(1, upper-1)])
                    ys += alphas[-1] * Fs[-1](xs, zs) * dZdS[-1]
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
                    x, z = ind_var, [slice_at]
                    ys = np.zeros((len(z), len(x)), dtype=complex)

                elif constant_variable == 'x':
                    x, z = [slice_at], ind_var
                    ys = np.zeros((len(z), len(x)), dtype=complex)

                else:
                    raise TypeError('Constant variable must be x or z.')

                Nz = len(selfz.Zs)
                alphas, Fs, dZdS = selfz.alphas, selfz.Fs, selfz.dZdS
                dS = selfz.dS

                if method == 'left_endpoint':
                    ys = np.sum([alphas[i] * Fs[i](x, zs=z) * dZdS[i] *
                                 dS[i] for i in range(Nz-1)])

                elif method == 'right_endpoint':
                    ys = np.sum([alphas[i+1] * Fs[i+1](x, zs=z) * dZdS[i+1] *
                                 dS[i] for i in range(Nz-1)])

                elif method == 'trapezoid':
                    ys = alphas[0] * Fs[0](x, zs=z) * dZdS[0] * dS[0]
                    ys += np.sum([alphas[i] * Fs[i](x, zs=z) * dZdS[i] *
                                 (dS[i] + dS[i-1]) for i in range(1, Nz-1)])
                    ys += alphas[-1] * Fs[-1](x, zs=z) * dZdS[-1] * dS[-1]
                    ys *= 1 / 2

                elif method == 'simpsons':
                    if len(selfz.Zs) % 2 != 1:
                        raise ValueError('Contour must have odd number of \
points points (even number of intervals) for Simpsons rule.')

                    set_dS = set(np.round(selfz.dS, decimals=13))
                    if len(set_dS) != 1:
                        raise ValueError('Simpsons rule requires equally \
spaced spaced intervals.')

                    dS = set_dS.pop()
                    upper = int((len(selfz.Zs)-1)/2) + 1
                    ys = alphas[0] * Fs[0](x, zs=z) * dZdS[0]
                    ys += 4 * np.sum([alphas[2*i-1] * Fs[2*i-1](x, zs=z) *
                                      dZdS[2*i-1] for i in range(1, upper)])
                    ys += 2 * np.sum([alphas[2*i] * Fs[2*i](x, zs=z) *
                                      dZdS[2*i] for i in range(1, upper-1)])
                    ys += alphas[-1] * Fs[-1](x, zs=z) * dZdS[-1]
                    ys *= 1/3 * dS

                if constant_variable == 'z':
                    return ys[0]
                else:
                    return ys[:, 0]


   def Zi_from_Z0(self, Z0, ni):
        '''Get Z on region i (with refractive index ni) from Z0.'''
        Z0, ni = np.array(Z0), np.array(ni)
        return np.sqrt((ni[np.newaxis]**2 - self.n0**2) * self.K0**2 +
                       Z0[..., np.newaxis]**2, dtype=complex)

    def region_indices(self, xs):
        '''Return region indices in which (non-dimensional) xs lie.  Indexing
        is based on left-continuity, so if x is in (rho_i, rho_i+1) it gets
        index i.  Points outside of domain on left get index 0, on right get
        index len(Rhos)-1.'''
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

        return idx

    def ns_from_xs(self, xs):
        '''Return refractive indices from region in which (non-dimensional) xs
        lie.'''
        return self.ns[self.region_indices(xs)]


