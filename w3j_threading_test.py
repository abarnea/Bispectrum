import matplotlib.pyplot as plt

import numpy as np
import healpy as hp
import pickle
import numba as nb
import pywigxjpf
import numba_progress
import math

from tqdm.notebook import tqdm
from spherical import Wigner3j
from scipy import stats
from pywigxjpf_ffi import ffi, lib
from numba import jit, njit, prange, set_num_threads
from numba_progress import ProgressBar
from numba.core.typing import cffi_utils as cffi_support
cffi_support.register_module(pywigxjpf_ffi)

nb_wig3jj = pywigxjpf_ffi.lib.wig3jj

from helper_funcs import *


@njit(parallel=True)
def compute_bispec_wig(l1, l2, l3, alms_l1, alms_l2, alms_l3, num_threads=16):

    # set_num_threads(num_threads)

    assert (l1 + l2 + l3) % 2 == 0, "even parity not satisfied" # even parity
    assert np.abs(l1-l2) <= l3, "LHS of triangle inequality not satisfied" # triangle inequality LHS
    assert l3 <= l1+l2, "RHS of triangle inequality not satisfied" # triangle inequality RHS

    bispec_sum = 0

    wig.wig_table_init((max(l1, l2, l3)+1)*2, 3)
    wig.wig_temp_init((max(l1, l2, l3)+1)*2)

    # alms_l1, alms_l2, alms_l3 = alms_l1.real, alms_l2.real, alms_l3.real

    for m1 in range(-l1, l1+1):
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