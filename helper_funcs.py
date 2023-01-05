import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pickle
import numba as nb
import spherical
import numba_progress
from scipy import stats
from scipy.integrate import romberg
from scipy import interpolate
from astropy.constants import c
from astropy.units import u
import pyccl as ccl
import pywigxjpf as wig
import pywigxjpf_ffi
from pywigxjpf_ffi import ffi, lib
import math

from tqdm.notebook import tqdm
from spherical import Wigner3j
from numba import jit, njit, prange, set_num_threads
from numba_progress import ProgressBar
from numba.core.typing import cffi_utils as cffi_support
cffi_support.register_module(pywigxjpf_ffi)

nb_wig3jj = pywigxjpf_ffi.lib.wig3jj

def mask_map(map_file, mask_file):
    mask = hp.read_map(mask_file).astype(np.bool_)
    masked_map = hp.ma(map_file)
    masked_map.mask = np.logical_not(mask)
    
    return masked_map

def plot_cl(ell, cls_list, labels=None, normalize=False, figsize=(12, 6), xscale='log', 
                    yscale='log', xlabel="multipole $\ell$", ylabel="$C_{\ell}$", 
                    ylim=[1e-11,1e1], title="Power Spectrum"):
    '''
    Plots the Angular Power Spectrum

    Parameters
    ------------
    l : multipole moments
    cl : cl's -- variance of alms
    normalize : default to False, if True: l(l+1)C_l / 2*pi normalization
    figsize : defaults to (12,6)
    xscale : defaults to log-scale
    yscale : defaults to log-scale
    xlabel : xlabel
    ylabel : ylabel
    title : title
    '''    
    plt.figure(figsize=figsize)

    if normalize:
        if len(np.shape(cls_list)) > 1:
            for cl,label in zip(cls_list, labels):
                plt.plot(ell, (ell * (ell + 1) * cl)/(2*np.pi), label=label)
            plt.legend(frameon=False, loc='upper right', fontsize=14)
        else:
            plt.plot(ell, (ell * (ell + 1) * cls_list)/(2*np.pi))
        plt.ylabel("$\ell(\ell+1)C_{\ell}/2\pi$", fontsize=18)
    else:
        if len(np.shape(cls_list)) > 1:
            for cl,label in zip(cls_list, labels):
                plt.plot(ell, cl, label=label)
            plt.legend(frameon=False, loc='upper right', fontsize=14)
        else:
            plt.plot(ell, cls_list)
        plt.ylabel(ylabel, fontsize=18)
    
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.ylim(ylim)
    plt.xlabel(xlabel, fontsize=18)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=18)
    plt.grid()

def plot_pk(k, pk_lin, pk_nl=None, figsize=(12,6), xscale='log', 
                        yscale='log', xlabel='$k\quad[Mpc^{-1}]$', 
                        ylabel='$P(k)\quad[{\\rm Mpc}]^3$', title="Power Spectrum"):
    '''
    Plots the Power Spectrum in k-space

    Parameters
    ------------
    k : k-modes
    pk_lin : linear power spectrum P(k)
    pk_nl : non-linear power spectrum
    figsize : defaults to (12,6)
    xscale : defaults to log-scale
    yscale : defaults to log-scale
    xlabel : xlabel
    ylabel : ylabel
    title : title
    '''

    plt.figure(figsize=figsize)
    plt.plot(k, pk_lin, 'b-')

    if pk_nl is not None:
        plt.plot(k, pk_nl, 'r-')

    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.show()

def smooth_map(map_file, deg=1., norm=None):
    smoothed_map = hp.smoothing(map_file, fwhm=np.radians(deg))
    # hp.mollview(smoothed_map, title="Map smoothed {:.2f} deg".format(deg), norm=norm)
    return smoothed_map
    
def find_stats(array, hp_map=False):
    shape = np.shape(array)
    mean = np.mean(array)
    median = np.median(array)
    std = np.std(array)
    iqr = stats.iqr(array)
    max_val = np.max(array)
    min_val = np.min(array)
    
    if hp_map:
        nside = hp.get_nside(array)
        pix_area = hp.nside2pixarea(nside)

        print("Shape: {} \nNSide: {} \nMean: {:.3e} \nMedian: {:.3e} \nStDev: {:.3e} \nIQR: {:.3e} \nMax Val: {:.3e} \nMin Val: {:.3e} \nPixel Area: {:.3e}".format(shape, nside, mean, median, std, iqr, max_val, min_val, pix_area))
    else:
        print("Shape: {} \nMean: {:.3e} \nMedian: {:.3e} \nStDev: {:.3e} \nIQR: {:.3e} \nMax Val: {:.3e} \nMin Val: {:.3e}".format(shape, mean, median, std, iqr, max_val, min_val))
        
def downsize_map(map_link, nside_out=2048, order_out='r'):
    # Reading the map
    input_map = np.fromfile(map_link, dtype='<f')
    
    if order_out == 'r':
        # Reorder from nested scheme to ring scheme
        rmap = hp.reorder(input_map, n2r = True)
    elif order_out == 'n':
        # Reorder map from ring scheme to nested scheme
        rmap = hp.reorder(input_map, r2n = True)

    # print npix,nside before downgrading
    print("npix before dg:",hp.get_nside(rmap),"| nside before dg:",hp.get_map_size(rmap))

    # Downsize it
    dg_map = hp.ud_grade(rmap, nside_out=nside_out, order_out=order_out)

    # print npix,nside after downgrading
    print("npix after dg:",hp.get_nside(dg_map),"| nside after dg:",hp.get_map_size(dg_map))

    hp.write_map(map_link, dg_map)
    print("saved successfully!")

def sort_alms(alms, lmax):
    '''
    Sorts healpix alm's by \ell instead of m given a fortran90 array output from hp.map2alm.

    Parameters
    ----------
    alms : fortran90 healpix alm array from hp.map2alm
    num_ls : number of l's in alms array
    
    Returns
    ----------
    sorted_alms : alm dictionary keyed by \ell values with numpy arrays consisting of the corresponding m values.
    '''
    start = 0
    sorted_alms = {}

    for l in range(lmax):
        sorted_alms[l] = np.zeros(2*l+1, dtype=np.cdouble)

    for m in range(lmax):
        num_ms = lmax - m
        ms = alms[start:num_ms+start]
        start = num_ms + start
        m_sign = (-1)**m
        for l in range(num_ms):
            idx = m + l
            sorted_alms[idx][m] = ms[l]
            if m != 0:
                sorted_alms[idx][-m] = m_sign * np.conj(ms[l])
    
    return sorted_alms

def compute_cls(sorted_alms):
    '''
    Computes the normalized C_l's given inputted alm's

    Parameters
    -----------
    sorted_alms : alm dictionary keyed by \ell values with numpy arrays consisting of the corresponding m values.

    Returns
    -----------
    cls : numpy array of length sorted_alms.keys() consisting of the properly normalized C_l's for each l value.
    '''
    cls = np.zeros(len(sorted_alms))

    for l in sorted_alms:
        cls[l] = 1/(2*l+1) * np.sum(np.abs(sorted_alms[l])**2)

    return cls

def find_z_cdm(z, cosmo, pkg = 'ccl', interp = 'z2cdm', tol=1e-6):

    c_km = c.to(u.km/u.s).value

    if pkg == 'ccl':
        def _invH(z):
            return c_km/(ccl.h_over_h0(cosmo, 1/(1+z)) * cosmo["H0"])
    elif pkg == 'astropy':
        def _invH(z):
            return c_km/cosmo.H(z).value
    else:
        print("Please input either 'ccl' or 'astropy' for pkg parameter.")

    zs = np.logspace(-2, 3, 1000)
    dcm = np.zeros_like(zs)

    for i,z in enumerate(zs):
        dcm[i] = romberg(_invH, 0, z, tol=tol)[0]

    if interp == 'z2cdm':
        cdm_from_z = interpolate.interp1d(zs, dcm, bounds_error=False, fill_value=0)
        return cdm_from_z
    elif interp == 'cdm2z':
        z_from_cdm = interpolate.interp1d(dcm, zs, bounds_error=False, fill_value=0)
        return z_from_cdm
    else:
        print("Please input either 'z2cdm' or 'cdm2z' for interp parameter.")

def check_alm_nan(alms):
    nan_count = 0
    nan_index = []
    for l in tqdm(alms.keys()):
        for alm in alms[l]:
            if np.isnan(np.abs(alm)):
                nan_count += 1
                nan_index.append(l)
    if (nan_count and len(nan_index)) > 0:
        print("Nan count:",nan_count)
        print("Nan indicies:",nan_index)
    else:
        print("all alm values are real")

# pywigxjpf
def get_w3j(j1, j2, j3, m1, m2, m3):
    return nb_wig3jj(2*j1, 2*j2, 2*j3, 2*m1, 2*m2, 2*m3)

@njit(parallel=True)
def compute_bispec_wig(l1, l2, l3, alms_l1, alms_l2, alms_l3, num_threads=16):

    set_num_threads(num_threads)

    wig.wig_table_init((max(l1, l2, l3)+1)*2, 3)
    wig.wig_temp_init((max(l1, l2, l3)+1)*2)

    assert (l1 + l2 + l3) % 2 == 0, "even parity not satisfied" # even parity
    assert np.abs(l1-l2) <= l3, "LHS of triangle inequality not satisfied" # triangle inequality LHS
    assert l3 <= l1+l2, "RHS of triangle inequality not satisfied" # triangle inequality RHS

    bispec_sum = 0

    # alms_l1, alms_l2, alms_l3 = alms_l1.real, alms_l2.real, alms_l3.real

    for m1 in prange(-l1, l1+1):
        for m2 in range(-l2, l2+1):
            m3 = -(m1 + m2) # condition that m1 + m2 + m3 == 0 fully determines m3
            w3j = nb_wig3jj(2*l1, 2*l2, 2*l3, 2*m1, 2*m2, 2*m3)
            if w3j != 0:
                exp_alms = alms_l1[m1] * alms_l2[m2] * alms_l3[m3]
                bispec_sum += w3j * np.abs(exp_alms)

    norm_factor = ((l1*2+1) * (l2*2+1) * (l3*2+1))/(4*np.pi) * (nb_wig3jj(2*l1, 2*l2, 2*l3, 0, 0, 0))**2

    wig.wig_temp_free()
    wig.wig_table_free()

    return np.sqrt(norm_factor) * bispec_sum

# @njit(parallel=True)
# def compute_bispec_jit(l1, l2, l3, alms_l1, alms_l2, alms_l3, num_threads=16):

#     set_num_threads(num_threads)

#     assert (l1 + l2 + l3) % 2 == 0, "even parity not satisfied" # even parity
#     assert np.abs(l1-l2) <= l3, "LHS of triangle inequality not satisfied" # triangle inequality LHS
#     assert l3 <= l1+l2, "RHS of triangle inequality not satisfied" # triangle inequality RHS

#     bispec_sum = 0

#     # alms_l1, alms_l2, alms_l3 = np.abs(alms_l1), np.abs(alms_l2), np.abs(alms_l3)
#     # alms_l1, alms_l2, alms_l3 = alms_l1.real, alms_l2.real, alms_l3.real

#     for m1 in prange(-l1, l1+1):
#         for m2 in range(-l2, l2+1):
#             m3 = -(m1 + m2) # condition that m1 + m2 + m3 == 0 fully determines m3
#             w3j = Wigner3j(l1, l2, l3, m1, m2, m3)
#             if w3j != 0:
#                 exp_alms = alms_l1[m1] * alms_l2[m2] * alms_l3[m3]
#                 bispec_sum += w3j * np.abs(exp_alms)
    
#     norm_factor = ((l1*2+1) * (l2*2+1) * (l3*2+1))/(4*np.pi) * (Wigner3j(l1, l2, l3, 0, 0, 0))**2

#     return np.sqrt(norm_factor) * bispec_sum

def split_bls(bls, get='both'):
    sorted_bls = sorted(bls.items()) # sorted by key, return a list of tuples
    ell_triplets, bls_vals = zip(*sorted_bls) # unpack a list of pairs into two tuples
    ell_triplets, bls_vals = np.array(ell_triplets), np.array(bls_vals)
    if get == 'both':
        return ell_triplets, bls_vals
    elif get == 'ells':
        return ell_triplets
    elif get == 'bls':
        return bls_vals
    else:
        return None

def bispec_range_eq(input_map, ells=np.arange(1, 500, 2), num_threads=32):

    lmax = np.max(ells)
    alms = hp.map2alm(input_map)
    sorted_alms = sort_alms(alms, lmax+1)

    bls = {}
    
    for l in tqdm(ells, "Looping over even equilateral ell-triplets"):
    # for l in ells:
        bls[(l, l, l)] = compute_bispec_wig(l, l, l, sorted_alms[l], sorted_alms[l], sorted_alms[l], num_threads=num_threads)
    
    return bls

def create_eq_ell_triplets(lmax):
    
    ells = np.arange(lmax+1)
    ell_triplets = np.zeros_like(ells, dtype=tuple)

    for l in ells:
        ell_triplets[l] = ((l, l, l))
    
    return ell_triplets

def get_bls_extrema(bls_gauss, bls_nongauss, xlmin=None, xlmax=None):
    if xlmin is not None and xlmax is not None:
        gauss_min = np.min(split_bls(bls_gauss, get='bls')[xlmin:xlmax])
        nongauss_min = np.min(split_bls(bls_nongauss, get='bls')[xlmin:xlmax])
        gauss_max = np.max(split_bls(bls_gauss, get='bls')[xlmin:xlmax])
        nongauss_max = np.max(split_bls(bls_nongauss, get='bls')[xlmin:xlmax])
    elif xlmin is not None and xlmax is None:
        gauss_min = np.min(split_bls(bls_gauss, get='bls')[xlmin:])
        nongauss_min = np.min(split_bls(bls_nongauss, get='bls')[xlmin:])
        gauss_max = np.max(split_bls(bls_gauss, get='bls')[xlmin:])
        nongauss_max = np.max(split_bls(bls_nongauss, get='bls')[xlmin:])
    elif xlmin is None and xlmax is not None:
        gauss_min = np.min(split_bls(bls_gauss, get='bls')[:xlmax])
        nongauss_min = np.min(split_bls(bls_nongauss, get='bls')[:xlmax])
        gauss_max = np.max(split_bls(bls_gauss, get='bls')[:xlmax])
        nongauss_max = np.max(split_bls(bls_nongauss, get='bls')[:xlmax])
    else:
        gauss_min = np.min(split_bls(bls_gauss, get='bls'))
        nongauss_min = np.min(split_bls(bls_nongauss, get='bls'))
        gauss_max = np.max(split_bls(bls_gauss, get='bls'))
        nongauss_max = np.max(split_bls(bls_nongauss, get='bls'))
    return gauss_min, nongauss_min, gauss_max, nongauss_max

def plot_bispec_eq(bls_list, labels=None, figsize=(12,6), xlmin=0, xlmax=None, xscale='linear', yscale='linear', xlabel="even multipole triplet $(\ell_1, \ell_2, \ell_3)$", ylabel="$B_{(\ell_1, \ell_2, \ell_3)}$", title="Bispectrum", scheme='e'):
    if scheme == 'e':
        plt.figure(figsize=figsize)
        assert type(bls_list) is list
        for bl,label in zip(bls_list,labels):
            ells, bls = split_bls(bl)
            step = ells[1]-ells[0]
            if xlmax is None:
                xlmax = np.max(ells)
            plt.plot(ells[xlmin//step:xlmax//step+1], bls[xlmin//step:xlmax//step+1], label=label)
        plt.legend(frameon=False, loc='upper right', fontsize=14)
        plt.plot(ells[xlmin//step:xlmax//step+1], np.zeros_like(ells[xlmin//step:xlmax//step+1]),'k--')
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.ylabel(ylabel, fontsize=18)
        plt.xlabel(xlabel, fontsize=18)
        plt.title(title, fontsize=18)
        # ell_window = ell_triplets[xlmin//step:xlmax//step+1]
        # plt.xticks(np.arange(0,len(bls_window)), [str(l) for l in ell_window])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()