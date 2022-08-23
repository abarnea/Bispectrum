import numpy as np
import astropy
import scipy
from scipy.integrate import romberg
from scipy import interpolate
from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c
from astropy.units import u

c_km = c.to(u.km/u.s).value

def _invH(z):
    return c_km/cosmo.H(z).value

zs = np.logspace(-2, 3, 1000)
dcm = np.zeros_like(zs)

for i,z in enumerate(zs):
    dcm[i] = romberg(_invH, 0, z, tol=1e-6)[0]

z_from_chi = interpolate.interp1d(dcm, zs, bounds_error=False, fill_value = 0)
chi_from_z = interpolate.interp1d(zs, dcm, bounds_error=False, fill_value = 0)

