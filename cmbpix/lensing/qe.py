import numpy as np
from cmbpix.lensing.qe_c import N1_bias_integral_cy_Kesden, N1_bias_integral_cy_Kesden_mc, N1_bias_integral_cy_Hanson

def N0_bias_integral(l1xv, l1yv, Lv, ClTTunlensed, ClTTtotal, l1min = 30, l1max = 8000):
    
    L = np.expand_dims(np.expand_dims(Lv,axis=-1),axis=-1)
    
    Ldotl1 = l1xv*L
    LdotLminusl1 = L*(L-l1xv)
    
    modl1 = np.sqrt(l1xv**2+l1yv**2)
    modLminusl1 = np.sqrt((L-l1xv)**2 + l1yv**2)
    
    maskl1 = np.ones(np.shape(modl1),dtype = int)
    maskl1[np.where(modl1 < l1min)] = 0
    maskl1[np.where(modl1 > l1max)] = 0

    maskLminusl1 = np.ones(np.shape(modLminusl1),dtype = int)
    maskLminusl1[np.where(modLminusl1 < l1min)] = 0
    maskLminusl1[np.where(modLminusl1 > l1max)] = 0
    
    integrand = maskl1 * maskLminusl1 \
        * ( Ldotl1*ClTTunlensed[np.rint(modl1).astype(int)] + LdotLminusl1*ClTTunlensed[np.rint(modLminusl1).astype(int)] ) \
        * ( Ldotl1*ClTTunlensed[np.rint(modl1).astype(int)] + LdotLminusl1*ClTTunlensed[np.rint(modLminusl1).astype(int)] ) \
        * 1./(2. * ClTTtotal[np.rint(modl1).astype(int)] * ClTTtotal[np.rint(modLminusl1).astype(int)]) \
    
    deltal1x = l1xv[0,1]-l1xv[0,0]
    deltal1y = l1yv[1,0]-l1yv[0,0]
    
    integrand*= deltal1x*deltal1y/((2*np.pi)**2)
    
    return np.sum(integrand, axis=(-2,-1))

def N1Kesden(Ls, uCl, tCl, Clpp, lmin=2, lmax=3000, dl=100, n_samps=0, version=0):
    """
    Compute the N1 lensing bias for the power spectrum of the lensing potential
    using the Kesden et al. (2002) estimator.

    Parameters
    ----------
    Ls : array_like
        Multipoles at which to compute the lensing bias.
    uCl : array_like
        Power spectrum of the lensed temperature field.
    tCl : array_like
        Power spectrum of the observed temperature field: T+noise+fg.
    Clpp : array_like
        Power spectrum of the lensing potential.
    lmin : int, default=2
        Minimum multipole of the QE filters.
    lmax : int, default=3000
        Maximum multipole of the QE filters.
    dl : int, default=100
        Integral bin width.
    n_samps : int, default=0
        Number of samples to use in the Monte Carlo integration. If zero, the
        integral is computed using the midpoint rule.

    Returns
    -------
    N1 : array_like
        N1 lensing bias for the power spectrum of the lensing potential.
    """
    N1 = np.zeros(np.size(Ls))
    if version == 0:
        N1func = N1_bias_integral_cy_Kesden
    elif version == 1 and n_samps > 0:
        N1func = N1_bias_integral_cy_Kesden_mc
    elif version == 2:
        N1func = N1_bias_integral_cy_Hanson
    for iL, LL in enumerate(Ls):
        if n_samps > 0:
            N1[iL] = N1func(LL, uCl, tCl, Clpp, lmin, lmax, dl, n_samps)
        else:
            N1[iL] = N1func(LL, uCl, tCl, Clpp, lmin, lmax, dl)
    return N1 / ((2.0*np.pi)**4)