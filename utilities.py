#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:16:45 2025

@author: pv
"""
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn


def plot_complex(f, rmin, rmax, imin, imax, fargs=(), fkwargs={}, rref=100,
                 iref=100, levels=70, log_abs=True, equal=False, grid=True,
                 colorbar=True,  figsize=(11, 5), cmap='viridis', part='norm',
                 facecolor='gray', close_others=True, pad=.02, shrink=1,
                 colorbar_frac=.05, colorbar_aspect=30, anchor=(.5, 1),
                 max_value=None, min_value=None, **contourargs):
    '''Plot complex valued function f on window in complex plane defined by
    rmin, rmax, imin, imax.'''
    if close_others:
        plt.close('all')
    xs = np.linspace(rmin, rmax, num=rref)
    ys = np.linspace(imin, imax, num=iref)
    Xs, Ys = np.meshgrid(xs, ys)
    Zs = Xs + 1j * Ys
    Fs = f(Zs, *fargs, **fkwargs)

    if part == 'real':
        F = Fs.real

    elif part == 'imag':
        F = Fs.imag
    elif part == 'norm':
        F = np.abs(Fs)
    elif part == 'phase':
        F = np.angle(Fs)
    else:
        raise ValueError('Part must be real, imag, norm or phase.')

    if log_abs:
        if part == 'phase':
            print('Taking log and absolute value of phase. May wish to set \
log_abs=False.')
        F = np.log(np.abs(F))

    if max_value is not None:
        msk = np.where(F > max_value)
        F[msk] = np.abs(max_value) * F[msk] / np.abs(F[msk])
    if min_value is not None:
        msk = np.where(F < min_value)
        F[msk] = np.abs(min_value) * F[msk] / np.abs(F[msk])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    ax.grid(grid)
    ax.set_facecolor(facecolor)
    im = ax.contour(xs, ys, F, levels=levels, **contourargs)

    if equal:
        ax.axis('equal')

    if colorbar:
        plt.colorbar(im, pad=pad, shrink=shrink,
                     orientation='vertical',
                     anchor=anchor, fraction=colorbar_frac,
                     aspect=colorbar_aspect, ax=ax)


def plot_complex_surface(f, rmin, rmax, imin, imax,
                         rref=100, iref=100, fargs=(), fkwargs={},
                         log_abs=False, equal=False, grid=True,
                         colorbar=True,  figsize=(11, 5), cmap='viridis',
                         part='norm', facecolor='gray', z_lims=None,
                         elev=50, roll=0, azim=-90, zoom=1.5, axis_off=True,
                         rstride=2, cstride=2, levels=None, max_value=None,
                         min_value=None, return_vals=False,
                         **contourargs):
    '''Plot complex valued function f on window in complex plane defined by
    rmin, rmax, imin, imax.'''
    plt.close('all')
    xs = np.linspace(rmin, rmax, num=rref)
    ys = np.linspace(imin, imax, num=iref)
    Xs, Ys = np.meshgrid(xs, ys)
    Zs = Xs + 1j * Ys
    Fs = f(Zs, *fargs, **fkwargs)

    if part == 'real':
        F = Fs.real
    elif part == 'imag':
        F = Fs.imag
    elif part == 'norm':
        F = np.abs(Fs)
    elif part == 'phase':
        F = np.angle(Fs)
    else:
        raise ValueError('Part must be real, imag, norm or phase.')

    if log_abs:
        F = np.log(np.abs(F))

    if max_value is not None:
        msk = np.where(F > max_value)
        F[msk] = np.abs(max_value) * F[msk] / np.abs(F[msk])
    if min_value is not None:
        msk = np.where(F < min_value)
        F[msk] = np.abs(min_value) * F[msk] / np.abs(F[msk])

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(projection='3d')
    lims = (np.ptp(xs), np.ptp(ys), min(np.ptp(xs), np.ptp(ys)))
    ax.set_box_aspect(lims, zoom=zoom)

    if z_lims is not None:
        z_min, z_max = z_lims[:]
        ax.set_zlim(z_min, z_max)
    if axis_off:
        ax.set_axis_off()

    ax.view_init(elev, azim, roll)

    im = ax.plot_surface(Xs, Ys, F, clip_on=False, cmap=cmap,
                         rstride=rstride, cstride=cstride,
                         **contourargs)
    if equal:
        ax.axis('equal')

    if colorbar:
        plt.colorbar(im, pad=.05, orientation='vertical',
                     anchor=(.5, 1), fraction=.15)
    if return_vals:
        return F


def plot_complex_contour(f, contour, fkwargs={}, rref=20, iref=20,
                         levels=25, log_abs=False, grid=True,
                         colorbar=True,  figsize=(11, 5), part='both',
                         legend_fontsize=12, gridlw=.1, max_value=None,
                         min_value=None, **lineargs):
    '''Plot complex valued function f on contour in complex plane.'''
    plt.close('all')
    Zs, Ss = contour['Zs'], contour['Ss']
    contour_type = contour['contour_type']

    if contour_type in ['real', 'horizontal', 'sdp']:
        xs = Zs.real
    if contour_type in ['imaginary', 'vertical']:
        xs = Zs.imag
    if contour_type in ['circle', 'half_circle']:
        xs = Ss
    Fs = f(Zs, **fkwargs)

    if part == 'real':
        F = Fs.real
    elif part == 'imag':
        F = Fs.imag
    elif part == 'both':
        Fr = Fs.real
        Fi = Fs.imag
    elif part == 'norm':
        F = np.abs(Fs)
    elif part == 'phase':
        F = np.angle(Fs)
    else:
        raise ValueError('Part must be real, imag, norm or phase.')

    if log_abs:
        if part == 'phase':
            print('Taking log and absolute value of phase. May wish to set \
log_abs=False.')
        if part == 'both':
            Fr = np.log(np.abs(Fr))
            Fi = np.log(np.abs(Fi))
        else:
            F = np.log(np.abs(F))

    if max_value is not None:
        msk = np.where(F > max_value)
        F[msk] = np.abs(max_value) * F[msk] / np.abs(F[msk])
    if min_value is not None:
        msk = np.where(F < min_value)
        F[msk] = np.abs(min_value) * F[msk] / np.abs(F[msk])

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot()
    ax.grid(grid, lw=gridlw)
    if part == 'both':
        ax.plot(xs, Fr, label='real', **lineargs)
        ax.plot(xs, Fi, label='imag', **lineargs)
    else:
        ax.plot(xs, Fr, label=part, **lineargs)
    plt.legend(fontsize=legend_fontsize)


def plotlogf(f, rmin, rmax, imin, imax, fkwargs={}, title='',
             levels=25, truncate=False, h=2, equal=False,
             colorbar=True, rref=20, iref=20, figsize=(12, 6),
             log_off=False, loop=False, three_D=False,
             cmap='viridis', phase=False, xy_plane=False,
             elev=50, roll=0, azim=-90, zoom=1.5,
             abs_off=False, part='norm', facecolor='gray',
             rstride=2, cstride=2, axis_off=True, grid=True,
             z_lims=None, **contourargs):
    """Create contour plot of complex function f on given range."""

    plt.close('all')

    if abs_off and not log_off:
        log_off = True
        warn("Setting log_off=True since you are using abs_off=True.")

    Xr = np.linspace(rmin, rmax, num=rref)
    Xi = np.linspace(imin, imax, num=iref)
    xr, xi = np.meshgrid(Xr, Xi)

    if not xy_plane:
        zs = xr + 1j * xi
    if not loop:
        if not xy_plane:
            fx = f(zs, **fkwargs)
        else:
            fx = f(xr, xi, **fkwargs)

    else:
        if not xy_plane:
            fx = np.zeros_like(zs)
            for i in range(zs.shape[0]):
                for j in range(zs.shape[1]):
                    fx[i, j] = f(zs[i, j], **fkwargs)
        else:
            fx = np.zeros_like(xr)
            for i in range(zs.shape[0]):
                for j in range(zs.shape[1]):
                    fx[i, j] = f(xr[i, j], xi[i, j], **fkwargs)

    if truncate:
        fx[np.abs(fx) > h] = h  # ignore large values to focus on roots

    if phase:
        if xy_plane:
            raise ValueError('Working with xy_plane=True; output assumed real, \
hence has no phase.  If complex output desired set xy_plane=False (default).')
        ys = np.angle(fx)
    else:
        if xy_plane:
            ys = fx
        else:
            if part == 'real':
                ys = fx.real
                if not abs_off:
                    ys = np.abs(ys)
            elif part == 'imag':
                ys = fx.imag
                if not abs_off:
                    ys = np.abs(ys)
            elif part == 'norm':
                ys = np.abs(fx)
            else:
                raise ValueError("Must choose norm, real or imag part.")
            # ys = np.abs(fx)

    if not log_off and phase:
        raise ValueError("Need log_off when phase is turned on.")

    if not log_off:
        ys = np.log(np.abs(ys))  # abs only needed if xy_plane=True

    fig = plt.figure(figsize=figsize)

    if three_D:
        ax = fig.add_subplot(projection='3d')
        lims = (np.ptp(Xr), np.ptp(Xi), min(np.ptp(Xr), np.ptp(Xi)))
        ax.set_box_aspect(lims, zoom=zoom)
        if z_lims is not None:
            z_min, z_max = z_lims[:]
            ax.set_zlim(z_min, z_max)
        if axis_off:
            ax.set_axis_off()
        ax.view_init(elev, azim, roll)

        im = ax.plot_surface(xr, xi, ys, clip_on=False, cmap=cmap,
                             rstride=rstride, cstride=cstride,
                             **contourargs)

    else:
        ax = fig.add_subplot()
        ax.grid(grid)
        ax.set_facecolor(facecolor)
        im = ax.contour(xr, xi, ys, levels=levels, **contourargs)

    if equal:
        ax.axis('equal')

    plt.title(title)
    if colorbar:
        plt.colorbar(im, pad=.05, orientation='vertical',
                     anchor=(.5, 1), fraction=.15)
    # return fig, ax


def plotlogf_real(f, x_min, x_max, fkwargs={}, n=1000, figsize=(10, 4),
                  level=np.e, log_off=False, truncate=False, height=2,
                  bounds=None, loop=False, part='norm', abs_off=False,
                  return_xyf=False, grid=False, plot_axis=True):
    """Plot (possibly complex) f on given real valued input range."""
    plt.close('all')
    if abs_off and not log_off:
        log_off = True
        warn("Setting log_off=True since you are using abs_off=True.")

    xs = np.linspace(x_min, x_max, num=n)
    if loop:
        fx = np.zeros_like(xs, dtype=complex)
        for i in range(len(xs)):
            fx[i] = f(xs[i], **fkwargs)
        if part == 'real':
            fx = fx.real
        elif part == 'imag':
            fx = fx.imag
        elif part == 'norm':
            fx = np.abs(fx)
        else:
            raise ValueError("Must choose norm, real or imag part.")
    else:
        fx = np.array(f(xs, **fkwargs), dtype=complex)
        if part == 'real':
            fx = fx.real
        elif part == 'imag':
            fx = fx.imag
        elif part == 'norm':
            fx = np.abs(fx)
        else:
            raise ValueError("Must choose norm, real or imag part.")

    fig, ax = plt.subplots(1, figsize=figsize)

    if log_off:
        if truncate:
            fx[np.abs(fx) > height] = height  # truncate to height
        if abs_off:
            ys = fx
            # ax.plot([xs[0], xs[-1]], [0, 0], linewidth=1.5, color='k')
        else:
            ys = np.abs(fx)
        ax.plot(xs, ys)
    else:
        ys = np.zeros_like(fx, dtype=float)
        ys[np.abs(fx) >= level] = np.log(np.abs(fx[np.abs(fx) >= level]))
        ys[np.abs(fx) < level] = np.abs(
            fx[np.abs(fx) < level]) * (np.log(level) / level)
        if truncate:
            ys[np.abs(ys) > height] = height
        ax.plot(xs, ys)

    bottom, top = plt.ylim()
    if not abs_off:
        ax.set_ylim(0, top)

    if bounds is not None:
        l, r = bounds[:]
        ax.plot([l, l], [0, top], color='orange', linestye=':', linewidth=.75)
        ax.plot([r, r], [0, top], color='orange', linestye=':', linewidth=.75)

    ax.grid(grid)

    if plot_axis:
        plt.axhline(0, color='lightgray')
        plt.axvline(0, color='lightgray')

    if return_xyf:
        return xs, ys, fx

    return fig, ax


def plotlogf_imag(f, im_min, im_max, fkwargs={}, n=1000, figsize=(10, 4),
                  level=np.e, log_off=False, truncate=False, height=2,
                  bounds=None, loop=False, part='real', abs_off=False,
                  grid=False, return_xyf=False, plot_x_axis=True,
                  plot_y_axis=True):
    """Plot (possibly complex) f on given pure imaginary valued input range."""

    if abs_off and not log_off:
        raise ValueError("Need to set log_off=True if using abs_off=True.")

    xs = np.linspace(im_min, im_max, num=n)
    ims = 1j * xs

    if loop:
        fx = np.zeros_like(xs, dtype=complex)
        for i in range(len(xs)):
            fx[i] = f(ims[i], **fkwargs)
        if part == 'real':
            fx = fx.real
        elif part == 'imag':
            fx = fx.imag
        elif part == 'norm':
            fx = np.abs(fx)
        else:
            raise ValueError("Must choose norm, real or imag part.")
    else:
        fx = np.array(f(ims, **fkwargs), dtype=complex)
        if part == 'real':
            fx = fx.real
        elif part == 'imag':
            fx = fx.imag
        elif part == 'norm':
            fx = np.abs(fx)
        else:
            raise ValueError("Must choose norm, real or imag part.")

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    if log_off:
        if truncate:
            fx[np.abs(fx) > height] = height  # truncate to height
        if abs_off:
            ys = fx
            ax.plot([xs[0], xs[-1]], [0, 0], linewidth=1.5, color='k')
        else:
            ys = np.abs(fx)
        ax.plot(xs, ys)
    else:
        ys = np.zeros_like(fx, dtype=float)
        ys[np.abs(fx) >= level] = np.log(np.abs(fx[np.abs(fx) >= level]))
        ys[np.abs(fx) < level] = np.abs(
            fx[np.abs(fx) < level]) * (np.log(level) / level)
        if truncate:
            ys[np.abs(ys) > height] = height
        ax.plot(xs, ys)

    bottom, top = plt.ylim()
    if not abs_off:
        ax.set_ylim(0, top)

    if plot_x_axis:
        ax.axhline(0, linewidth=.7, color='gray')
    if plot_y_axis:
        ax.axvline(0, linewidth=.7, color='gray')
    if bounds is not None:
        l, r = bounds[:]
        ax.plot([l, l], [0, top], color='orange', linestye=':', linewidth=.75)
        ax.plot([r, r], [0, top], color='orange', linestye=':', linewidth=.75)
    ax.grid(grid)
    if return_xyf:
        return xs, ys, fx
    return fig, ax
