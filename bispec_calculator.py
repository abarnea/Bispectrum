import numpy as np
import healpy as hp
import scipy
from numba import jit, njit, prange, set_num_threads
from numba.typed import List, Dict
from tqdm.notebook import tqdm
from helper_funcs import *


def sort_alms(alms, lmax):
    '''
    Sorts healpix alm's by \ell instead of m given a fortran90 array output from hp.map2alm.

    Parameters
    ----------
    alms : fortran90 healpix alm array from hp.map2alm
    num_ls : number of l's in alms array
    
    Returns
    ----------
    sorted_alms : alm dictionary keyed by ell values with numpy arrays 
                    consisting of the corresponding m values.
    '''
    start = 0
    sorted_alms = {}

    for l in range(lmax + 1):
        sorted_alms[l] = np.zeros(2*l+1, dtype=np.cdouble)

    for m in range(lmax + 1):
        num_ms = lmax + 1 - m
        ms = alms[start:num_ms + start]
        start = num_ms + start
        m_sign = (-1)**m
        for l in range(num_ms):
            idx = m + l
            sorted_alms[idx][m] = ms[l]
            if m != 0:
                sorted_alms[idx][-m] = m_sign * np.conj(ms[l])
    
    return sorted_alms


def unsort_alms(sorted_alms):
    """
    Unsorts a sorted_alm dictionary from sorted_alms into a Fortran90
    array that can be processed by HEALPix.

    Inputs:
        sorted_alms (dict[ndarray]) : sorted alms from sort_alms

    Returns:
        (ndarray) : unsorted alms for HEALPix
    """
    ells = np.array(list(sorted_alms.keys()))
    lmax = ells[-1]
    unsorted_alms = np.zeros(np.sum(ells + 1), dtype=np.complex128)

    start = 0
    for m in range(lmax + 1):
        for ell in range(m, lmax + 1):
            unsorted_alms[start + ell] = sorted_alms[ell][m]
        start += lmax - m
    
    return unsorted_alms

@njit
def check_valid_triangle(l1, l2, l3):
    """
    Checks if a set of l1, l2, l3's form a valid triangle, which can be
    confirmed if the l's satisfy the following three metrics:
    1. bispectrum is symmetric under any l1, l2, l3 permutations
    2. even parity selection rule
    3. triangle inequality

    Inputs:
        l1 (int) : l1
        l2 (int) : l2
        l3 (int) : l3
    
    Returns:
        (bool) : returns True if triangle is valid, False otherwise
    """
    permutations = l1 <= l2 <= l3
    even_parity = (l1 + l2 + l3) % 2 == 0
    tri_inequality = np.abs(l1 - l2) <= l3 <= l1 + l2

    return permutations and even_parity and tri_inequality

@njit
def count_valid_configs(i1, i2, i3, num_threads=16):
    """
    Counts the number of valid ell-triplet configurations in a single bin.
    Validity is determined as satisfying the parity condition selection
    rule and the triangle inequality.

    Inputs:
        i1 (int) : bin1
        i2 (int) : bin2
        i3 (int) : bin3
        num_threads (int) : number of threads to parallelize on

    Returns:
        (int) : number of valid configurations of ell-triplets.
    """

    set_num_threads(num_threads)

    configs = 0

    for l1 in i1:
        for l2 in i2:
            for l3 in i3:
                if check_valid_triangle(l1, l2, l3):
                    configs += 1
    
    return configs

def filter_map_binned(sorted_alms, i_bins, nside=1024):
    """
    Map alms that have been binned and filtered.

    Inputs:
        sorted_alms (dict) : dictionary of sorted alms
        lmin (int) : lmin
        lmax (int) : lmax
        nbins (int) : number of bins
    
    Returns:
        (ndarray) : healpix map of filtered alms
    """
    lmax = max(sorted_alms.keys())
    filtered_alms = {}
    for ell in range(lmax + 1):
        filtered_alms[ell] = np.zeros(2 * ell + 1, dtype=np.cdouble)

    for ell in i_bins:
        filtered_alms[ell] = sorted_alms[ell]
    
    return hp.alm2map(unsort_alms(filtered_alms), nside=nside)

def get_three_filtered_maps(sorted_alms, i1, i2, i3, nside=1024):
    map_i1 = filter_map_binned(sorted_alms, i1, nside=nside)
    map_i2 = filter_map_binned(sorted_alms, i2, nside=nside)
    map_i3 = filter_map_binned(sorted_alms, i3, nside=nside)
    return map_i1, map_i2, map_i3

def create_bins(lmin, lmax, nbins=1):
    """
    Creates bins in an ell-range.

    Parameters:
        lmin (int) : min ell value of bins
        lmax (int) : max ell value of bins
        nbins (int) : number of bins
    
    Returns:
        bins (list of ndarray) : all the bins
    """
    bin_range = np.linspace(lmin, lmax, nbins + 1)

    min_val = bin_range[:-1]
    max_val = bin_range[1:]

    bins = []

    for i in range(nbins):
        bins.append(np.arange(min_val[i], max_val[i]))
    
    return bins

def select_bins(lmin, lmax, nbins=1, bins_to_use=[0, 0, 0]):
    bin1, bin2, bin3 = bins_to_use
    bins = create_bins(lmin, lmax, nbins)
    return bins[bin1], bins[bin2], bins[bin3]

def create_bins_and_maps(sorted_alms, lmin, lmax, nbins, bins_to_use=[0, 0, 0]):
    bins = create_bins(lmin, lmax, nbins)
    i1_bin, i2_bin, i3_bin = select_bins(bins, bins_to_use)
    i1_map, i2_map, i3_map = get_three_filtered_maps(sorted_alms, i1_bin, i2_bin, i3_bin)

    return i1_bin, i2_bin, i3_bin, i1_map, i2_map, i3_map


def get_pix_area(map):
    return hp.nside2pixarea(hp.get_nside(map))

def get_pix_areas(map1, map2, map3):
    return get_pix_area(map1), get_pix_area(map2), get_pix_area(map3)

def get_map_sizes(map1, map2, map3):
    return hp.get_map_size(map1). hp.get_map_size(map2), hp.get_map_size(map3)


# @njit(parallel=True)
def compute_binned_bispec(i1, i2, i3, map_i1, map_i2, map_i3, num_threads=16):
    """
    Compute binned bispec

    Parameters:
    """
    xi = count_valid_configs(i1, i2, i3, num_threads=num_threads)
    assert xi != 0

    pixarea = get_pix_area(map_i1)
    B_i = np.sum(map_i1 * map_i2 * map_i3) * pixarea

    return (1 / xi) * B_i


## Filter Method

@njit
def find_valid_configs(i1, i2, i3, num_threads=16):
    """
    Counts the number of valid ell-triplet configurations in a single bin.
    Validity is determined as satisfying the parity condition selection
    rule and the triangle inequality.

    Inputs:
        i1 (int) : bin1
        i2 (int) : bin2
        i3 (int) : bin3
        num_threads (int) : number of threads to parallelize on

    Returns:
        (int) : number of valid configurations of ell-triplets.
    """

    # set_num_threads(num_threads)

    # num_configs = count_valid_configs(i1, i2, i3, num_threads=num_threads)
    
    # lmin1, lmax1 = i1.min(), i1.max()

    valid_configs = List()

    for i, l1 in enumerate(i1):
        for l2 in i2[i:]:
            for l3 in i3[i:]:
                if check_valid_triangle(l1, l2, l3):
                    valid_configs.append((l1, l2, l3))

    return valid_configs

@njit
def compute_bispec_norm_factor(l1, l2, l3):
    return ((l1*2+1) * (l2*2+1) * (l3*2+1))/(4*np.pi) \
                        * (nb_wig3jj(2*l1, 2*l2, 2*l3, 0, 0, 0))**2

@njit(parallel=True)
def compute_bispec(l1, l2, l3, alms_l1, alms_l2, alms_l3, num_threads=16):

    if not check_valid_triangle(l1, l2, l3):
        return None

    bispec_sum = 0
    val_init = (max(l1, l2, l3) + 1) * 2

    set_num_threads(num_threads) # set for Roomba to be 16 threads max

    lib.wig_table_init(val_init, 3)

    lib.wig_temp_init(val_init)
    norm_factor = compute_bispec_norm_factor(l1, l2, l3)
    lib.wig_temp_free()

    if not norm_factor:
        return 0

    for m1 in prange(-l1, l1 + 1):
        lib.wig_temp_init(val_init)
        for m2 in range(-l2, l2 + 1):
            m3 = -(m1 + m2) # noting that m1 + m2 + m3 == 0
            if -l3 <= m3 <= l3:
                w3j = nb_wig3jj(2*l1, 2*l2, 2*l3, 2*m1, 2*m2, 2*m3)
                if w3j:
                    conv = (0 + 1j)**(l1 + l2 + l3)
                    exp_alms = conv * alms_l1[m1] * alms_l2[m2] * alms_l3[m3]
                    bispec_sum += w3j * exp_alms.real
        lib.wig_temp_free()
    
    lib.wig_table_free()

    return get_perm_weighting(l1, l2, l3) * np.sqrt(norm_factor) * bispec_sum

@njit
def get_perm_weighting(l1, l2, l3):
    """
    Finds the permutation weighting scale factor given 3 inputted ell values.

    Inputs:
        l1 (int) : l1
        l2 (int) : l2
        l3 (int) : l3
    
    Returns:
        (int): permutation weighting scale factor
    """
    if l1 != l2 != l3:
        return 6
    elif l1 == l2 == l3:
        return 1
    else:
        return 3

def compute_averaged_bispec(i1, i2, i3, sorted_alms, num_threads=16):
    
    configs = find_valid_configs(i1, i2, i3)

    bls = 0
    for l1, l2, l3 in configs:
        bls += compute_bispec(l1, l2, l3, sorted_alms[l1], sorted_alms[l2], \
                                    sorted_alms[l3], num_threads=num_threads)
    
    return bls / len(configs)


# def compute_averaged_bispec_long(i1, i2, i3, sorted_alms, num_threads=16):

#     count = 0
#     bls = 0
#     for i, l1 in enumerate(i1):
#         for l2 in i2[i:]:
#             for l3 in i3[i:]:
#                 if check_valid_triangle(l1, l2, l3):
#                     bls += compute_bispec(l1, l2, l3, sorted_alms[l1], \
#                         sorted_alms[l2], sorted_alms[l3], num_threads=num_threads)
    
#     return bls / count

def find_bispec_var(l1, l2, l3, binned=False):
    val_init = (max(l1, l2, l3) + 1) * 2
    
    lib.wig_table_init(val_init, 3)
    lib.wig_temp_init(val_init)
    norm_factor = ((l1 * 2 + 1) * (l2 * 2 + 1) * (l3 * 2 + 1))/(4 * np.pi) * (get_w3j(l1, l2, l3, 0, 0, 0))**2
    lib.wig_temp_free()
    lib.wig_table_free()
    
    ells = np.array([l1, l2, l3])
    cls = (ells+0.0)**(-3.)
    if binned:
        return norm_factor * np.prod(cls)
    else:
        return find_g_val(l1, l2, l3) * norm_factor * np.prod(cls)

def find_binned_bispec_var(i1, i2, i3):
    """
    Finds the variance of the binned bispectrum according to Eq.

    Inputs:
        i1 (ndarray) : i1 bins
        i2 (ndarray) : i2 bins
        i3 (ndarray) : i3 bins
    
    Returns:
        (ndarray) : variance of binned bispectrum
    """
    configs = find_valid_configs(i1, i2, i3)
    num_configs = len(configs)

    bls_var = np.zeros(num_configs, dtype='float')
    for i, (l1, l2, l3) in enumerate(configs):
        bls_var[i] += find_bispec_var(l1, l2, l3, binned=True)
    
    g = find_g_val(i1[-1], i2[-1], i3[-1])

    return g/(num_configs**2) * bls_var

@njit
def find_g_val(val1, val2, val3):
    """
    Finds the g-value given 3 inputted ell values or bins.

    Inputs:
        val1 (int) : 
    """
    if val1 == val2 == val3:
        return 6
    elif val1 != val2 != val3:
        return 1
    else:
        return 2