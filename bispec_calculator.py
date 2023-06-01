import numpy as np
import healpy as hp
from numba import njit, prange, set_num_threads
from typing import Tuple, List
# from numba.typed import List, Dict
from tqdm.notebook import tqdm
from helper_funcs import *


def sort_alms(alms: np.ndarray, lmax: int) -> dict:
    '''
    Sorts healpix alm's by \ell instead of m given a fortran90 array output from hp.map2alm.

    Parameters
    ----------
    alms (ndarray) : fortran90 healpix alm array from hp.map2alm
    num_ls (int ): number of l's in alms array
    
    Returns
    ----------
    sorted_alms (dict) : alm dictionary keyed by ell values with numpy arrays
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


def unsort_alms(sorted_alms: dict) -> np.ndarray:
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

def filter_alms(alms, bins):
    """
    Filters alms without using dictionaries.

    Parameters:
        alms (ndarray) : alms
        bins (ndarray) : bins
    
    Returns:
        filtered_alms (ndarray) : filtered alms
    """
    lmax = hp.Alm.getlmax(alms.size)
    filtered_alms = np.zeros_like(alms)
    for ell in bins:
        ms = np.arange(-ell, ell + 1)
        indices = hp.Alm.getidx(lmax, ell, ms).astype(int)
        filtered_alms[indices] = alms[indices]
    return filtered_alms


@njit
def check_valid_triangle(l1: int, l2: int, l3: int) -> bool:
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
def count_valid_configs(i1: np.ndarray,
                        i2: np.ndarray,
                        i3: np.ndarray,
                        num_threads=16) -> int:
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


def filter_map_binned(sorted_alms: dict, bin: np.ndarray, nside=1024) -> np.ndarray:
    """
    Map alms that have been binned and filtered.

    Inputs:
        sorted_alms (dict) : dictionary of sorted alms
        bin (int) : bin to filter
        nside (int) : nside
    
    Returns:
        (ndarray) : healpix map of filtered alms
    """
    lmax = max(sorted_alms.keys())
    filtered_alms = {}
    for ell in range(lmax + 1):
        filtered_alms[ell] = np.zeros(2 * ell + 1, dtype=np.cdouble)

    for ell in bin:
        filtered_alms[ell] = sorted_alms[ell]
    
    return hp.alm2map(unsort_alms(filtered_alms), nside=nside)


def _create_bins(lmin: int, lmax: int, nbins=1) -> List[np.ndarray]:
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


def create_and_select_bins(lmin: int,
                lmax: int,
                nbins=1,
                bins_to_use=[0, 0, 0]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates and selects bins to use from a range of ell-values and a given
    number of bins.

    Parameters:
        lmin (int) : minimum ell value of the bins
        lmax (int) : maximum ell value of the bins
        nbins (int) : number of bins to create
        bins_to_use (List[int, int, int]) : bin numbers to use starting at 0
    
    Returns:
        (ndarray, ndarray, ndarray) : three selected bins
    """
    bin1, bin2, bin3 = bins_to_use
    bins = _create_bins(lmin, lmax, nbins)
    return bins[bin1], bins[bin2], bins[bin3]


def create_bins_and_maps(alms: np.ndarray,
                         lmin: int,
                         lmax: int,
                         nbins=1,
                         bins_to_use=[0, 0, 0]):
    """
    Creates and selects bins from a range of ell-values and a given number of
    bins, then creates maps based on the three selected bins.

    Parameters:
        alms (ndarray) : alms to map
        lmin (int) : minimum ell value of the bins
        lmax (int) : maximum ell value of the bins
        nbins (int) : number of bins to create
        bins_to_use (List[int, int, int]) : bin numbers to use starting at 0
    
    Returns:
        i1_bin (ndarray) : first bin
        i2_bin (ndarray) : second bin
        i3_bin (ndarray) : third bin
        i1_map (ndarray) : first bin map
        i2_map (ndarray) : second bin map
        i3_map (ndarray) : third bin map
    """
    i1_bin, i2_bin, i3_bin = select_bins(lmin, lmax, nbins, bins_to_use)
    i1_map = hp.alm2map(filter_alms(alms, i1_bin))
    i2_map = hp.alm2map(filter_alms(alms, i2_bin))
    i3_map = hp.alm2map(filter_alms(alms, i3_bin))

    return i1_bin, i2_bin, i3_bin, i1_map, i2_map, i3_map


def get_pix_area(map: np.ndarray) -> int:
    """
    Gets the pixel area of a HEALPix map.

    Parameters:
        map (ndarray) : healpix map
    
    Returns:
        (int) : pixel area
    """
    return hp.nside2pixarea(hp.get_nside(map))


# @njit(parallel=True)
def compute_binned_bispec(i1: np.ndarray,
                          i2: np.ndarray,
                          i3: np.ndarray,
                          map_i1: np.ndarray,
                          map_i2: np.ndarray,
                          map_i3: np.ndarray,
                          num_threads=16) -> np.ndarray:
    """
    Compute binned bispec

    Parameters:
        i1 (ndarray) : bin1
        i2 (ndarray) : bin2
        i3 (ndarray) : bin3
        map_i1 (ndarray) : i1 bin map
        map_i2 (ndarray) : i2 bin map
        map_i3 (ndarray) : i3 bin map
        num_threads (int) : number of threads to allocate
    
    Returns:
        (ndarray) : binned bispectrum
    """
    xi = count_valid_configs(i1, i2, i3, num_threads=num_threads)
    
    if xi == 0:
        raise ZeroDivisionError

    pixarea = get_pix_area(map_i1)
    B_i = np.sum(map_i1 * map_i2 * map_i3) * pixarea

    return (1 / xi) * B_i


## Filter Method

@njit
def find_valid_configs(i1: np.ndarray,
                       i2: np.ndarray,
                       i3: np.ndarray, \
                       num_threads=16) -> int:
    """
    Counts the number of valid ell-triplet configurations in a single bin.
    Validity is determined as satisfying the parity condition selection
    rule and the triangle inequality.

    Inputs:
        i1 (ndarray) : bin1
        i2 (ndarray) : bin2
        i3 (ndarray) : bin3
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


def compute_bispec_norm_factor(l1: int, l2: int, l3: int) -> float:
    """
    Computes the normalization factor of the averaged Bispectrum equation.

    Parameters:
        l1 (int) : first ell value
        l2 (int) : second ell value
        l3 (int) : third ell value
    
    Returns:
        (float) : normalization factor
    """
    val_init = (max(l1, l2, l3) + 1) * 2
    
    lib.wig_table_init(val_init, 3)
    lib.wig_temp_init(val_init)
    norm_factor = ((l1*2+1) * (l2*2+1) * (l3*2+1))/(4*np.pi) * (get_w3j(l1, l2, l3, 0, 0, 0))**2
    lib.wig_temp_free()
    lib.wig_table_free()

    return norm_factor
                        


@njit(parallel=True)
def compute_bispec(l1: int,
                   l2: int,
                   l3: int,
                   alms_l1 : dict,
                   alms_l2 : dict,
                   alms_l3 : dict,
                   num_threads=16) -> float:
    """
    Computes the bispectrum value of a set of ell-values and alm values and
    parallelizes computation across 16-threads by default.

    Parameters:
        l1 (int) : first ell value
        l2 (int) : second ell value
        l3 (int) : third ell value
        alms_l1 (dict) : dictionary of l1 alms
        alms_l2 (dict) : dictionary of l2 alms
        alms_l3 (dict) : dictionary of l3 alms
        num_threads (int) : number of threads to parallelize across

    Returns:
        (float) : bispectrum value
    """

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
def get_perm_weighting(l1: int, l2: int, l3: int) -> int:
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

def compute_averaged_bispec(i1: int,
                            i2: int,
                            i3: int,
                            sorted_alms: dict,
                            num_threads=16) -> float:
    """
    Computes the averaged binned bispectrum.

    Parameters:
        i1 (ndarray) : first ell bin
        i2 (ndarray) : second ell bin
        i3 (ndarray) : third ell bin
        sorted_alms (dict) : alm values
        num_threads (int) : number of threads to parallelize computation across
    
    Returns:
        (float) : averaged binned bispectrum value
    """
    
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

def find_bispec_var(l1, l2, l3):
    norm_factor = compute_bispec_norm_factor(l1, l2, l3)
    
    ells = np.array([l1, l2, l3])
    cls = (ells+0.0)**(-3.)

    return find_g_val(l1, l2, l3) * norm_factor * np.prod(cls)


def find_binned_bispec_var(i1: np.ndarray, i2: np.ndarray, i3: np.ndarray) -> np.ndarray:
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
def find_g_val(val1: int, val2: int, val3: int) -> int:
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
