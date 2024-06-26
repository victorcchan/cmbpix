import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap, utils as putils
#from cmbpix.utils import *
from orphics.maps import binned_power, FourierCalc
from orphics.stats import bin2D
import symlens as s
from scipy.interpolate import CubicSpline
from cmbpix.lensing.SCALE_c import Psi_and_A_cy, Psi_and_A_cy_mc

def ApplyFilter(map_in, Fl):
    return enmap.ifft(enmap.fft(map_in)*Fl).real

def WienerFilter(map_in, ells, uCls, lCls, Nls, lmin, lmax, grad=False):
    cs = CubicSpline(ells, uCls / (lCls + Nls))
    lmap = map_in.modlmap()
    fl = cs(lmap)
    fl[0,0] = 0.
    fl[(lmap < lmin) | (lmap > lmax)] = 0.
    if grad:
        ly, lx = map_in.lmap()
        return ApplyFilter(map_in, 1j*fl*ly), ApplyFilter(map_in, 1j*fl*lx)
    else:
        return ApplyFilter(map_in, fl)

def InvVarFilter(map_in, ells, Cls, Nls, lmin, lmax, grad=False):
    cs = CubicSpline(ells, 1./(Cls + Nls))
    lmap = map_in.modlmap()
    fl = cs(lmap)
    fl[0,0] = 0.
    fl[(lmap < lmin) | (lmap > lmax)] = 0.
    if grad:
        ly, lx = map_in.lmap()
        return ApplyFilter(map_in, 1j*fl*ly), ApplyFilter(map_in, 1j*fl*lx)
    else:
        return ApplyFilter(map_in, fl)
    
def CalcBias(shape, wcs, ells, ldT, lmin, lmax, lbin, lCls, Nls, w, sg, Clpp, plots=False):
    # shape,wcs = enmap.geometry(shape=(side,side),res=res*putils.arcmin,pos=(0,0))
    # shape, wcs = map_in.shape, map_in.wcs
    lmap = enmap.modlmap(shape, wcs)
    el = np.arange(0, ldT, 1, dtype=np.float64)
    prefactor = (s.l1*s.l1-s.Ldl1)*(s.l1*s.l1-s.Ldl1)
    spectra = s.e('uC_T_T_l1')*s.e('uC_T_T_l1')/s.e('tC_T_T_l1') * s.e('uC_T_T_l2')*s.e('uC_T_T_l2')/s.e('tC_T_T_l2')
    AL_integrand = prefactor*spectra
    feed_dict = {}
    feed_dict['uC_T_T'] = s.interp(el,lCls[:ldT])(lmap)
    feed_dict['tC_T_T'] = s.interp(el,lCls[:ldT])(lmap)+(w*np.pi/180./60.)**2./s.gauss_beam(lmap,sg)**2.
    tellmin = 0 ; tellmax = ldT
    xmask = s.mask_kspace(shape,wcs,lmin=tellmin,lmax=tellmax)
    integral = s.integrate(shape,wcs,feed_dict,AL_integrand,xmask=xmask,ymask=xmask,physical_units = False).real * enmap.pixsize(shape,wcs)**0.5 / (np.prod(shape[-2:])**0.5)
    AL = 1./integral
    ells_small_scale = np.arange(lmin,lmax,1, dtype=np.float64)
    ctt_small_scale = lCls[lmin:lmax]
    ctt_obs_small_scale = (lCls+Nls)[lmin:lmax]
    # prefactor_small_scale = (s.l1*s.l1)*(s.l1*s.l1)*s.L/s.L
    prefactor_small_scale = (s.l1*s.l1-s.Ldl1)*(s.l1*s.l1)
    spectra_small_scale = 1./s.e('tC_T_T_l1') * 1./s.e('tC_T_T_l2')
    integrand_small_scale = prefactor_small_scale*spectra_small_scale
    feed_dict_small_scale = {}
    feed_dict_small_scale['uC_T_T'] = s.interp(ells_small_scale,ctt_small_scale)(lmap)
    feed_dict_small_scale['tC_T_T'] = s.interp(ells_small_scale,ctt_small_scale)(lmap)+(w*np.pi/180./60.)**2./s.gauss_beam(lmap,sg)**2.
    ## Set ell range according to g filter
    tellmin_small_scale = lmin ; tellmax_small_scale = lmax
    ymask = s.mask_kspace(shape,wcs,lmin=tellmin_small_scale,lmax=tellmax_small_scale)
    small_scale_integral = small_scale_integral = s.integrate(shape,wcs,feed_dict_small_scale,integrand_small_scale,xmask=ymask,ymask=ymask,physical_units = False).real * enmap.pixsize(shape,wcs)**0.5 / (np.prod(shape[-2:])**0.5)
    AL /= small_scale_integral
    binner = s.bin2D(lmap,lbin)
    cents,AL1d = binner.bin(AL)
    prefactorB = (s.l1*s.l1-s.Ldl1)*(s.l1*s.l1-s.Ldl1)
    spectraB = s.e('uC_T_T_l1')/s.e('tC_T_T_l1') * s.e('uC_T_T_l2')/s.e('tC_T_T_l2') * ( s.e('uC_T_T_l1')+s.e('uC_T_T_l2') )
    BL_integrand = prefactorB*spectraB
    BL = s.integrate(shape,wcs,feed_dict,BL_integrand,xmask=xmask,ymask=xmask).real
    # prefactorB_small_scale = (s.l1*s.l1)*(s.l1*s.l1)*s.L/s.L
    prefactorB_small_scale = (s.l1*s.l1-s.Ldl1)*(s.l1*s.l1)
    spectraB_small_scale = s.e('uC_T_T_l1')/s.e('tC_T_T_l1') * s.e('uC_T_T_l2')/s.e('tC_T_T_l2') * ( s.e('uC_T_T_l1')+s.e('uC_T_T_l2') )
    BL_integrand_small_scale = prefactorB_small_scale*spectraB_small_scale
    B_small_scale_integral = s.integrate(shape,wcs,feed_dict_small_scale,BL_integrand_small_scale,xmask=ymask,ymask=ymask,physical_units = False).real * enmap.pixsize(shape,wcs)**0.5 / (np.prod(shape[-2:])**0.5)
    BL *= B_small_scale_integral
    binner = s.bin2D(lmap,lbin)
    cents,BL1d = binner.bin(BL)
    BL1d *= AL1d
    BL1d *= Clpp[cents.astype(int)]
    if plots:
        f, axs = plt.subplots(2,1,figsize=(8,12))
        axs[0].semilogy(cents, AL1d)
        axs[0].set(ylabel='A(L)', xlabel='L')
        axs[1].semilogy(cents, BL1d)
        axs[1].set(ylabel='B(L)', xlabel='L')
        plt.show()
    return cents, AL1d, BL1d

def CalcExp(shape, wcs, lCls, w, sg, Clpp, ldT, lmin, lmax, lbin, plots=False):
    # shape, wcs = map_in.shape, map_in.wcs
    lmap = enmap.modlmap(shape, wcs)
    ells = np.arange(0.,ldT,1, dtype=np.float64)
    ctt = lCls[:ldT]
    cphiphi = Clpp[lmin:lmax]
    ells_small_scale = np.arange(lmin,lmax,1, dtype=np.float64)
    ctt_small_scale = lCls[lmin:lmax]
    # prefactor_small_scale = (s.l1*s.l1)*(s.l1*s.l1)*s.L/s.L
    prefactor_small_scale = (s.l1*s.l1-s.Ldl1)*(s.l1*s.l1)
    spectra_small_scale = 1./s.e('tC_T_T_l1') * 1./s.e('tC_T_T_l2')
    spectra_numerator = s.e('uC_phi_phi_l1')/s.e('tC_T_T_l1') * 1./s.e('tC_T_T_l2')
    integrand_small_scale = prefactor_small_scale*spectra_small_scale
    integrand_numerator = prefactor_small_scale*spectra_numerator
    feed_dict_small_scale = {}
    feed_dict_small_scale['uC_T_T'] = s.interp(ells_small_scale,ctt_small_scale)(lmap)
    feed_dict_small_scale['uC_phi_phi'] = s.interp(ells_small_scale,cphiphi)(lmap)
    feed_dict_small_scale['tC_T_T'] = s.interp(ells_small_scale,ctt_small_scale)(lmap)+(w*np.pi/180./60.)**2./s.gauss_beam(lmap,sg)**2.
    tellmin = lmin ; tellmax = lmax 
    xmask = s.mask_kspace(shape,wcs,lmin=tellmin,lmax=tellmax)
    small_scale_integral = s.integrate(shape,wcs,feed_dict_small_scale,integrand_small_scale,xmask=xmask,ymask=xmask,physical_units = False).real * enmap.pixsize(shape,wcs)**0.5 / (np.prod(shape[-2:])**0.5)
    numerator_integral = s.integrate(shape,wcs,feed_dict_small_scale,integrand_numerator,xmask=xmask,ymask=xmask,physical_units = False).real * enmap.pixsize(shape,wcs)**0.5 / (np.prod(shape[-2:])**0.5)
    Phi_theory = numerator_integral/small_scale_integral
    binner = s.bin2D(lmap,lbin)
    cents,hat_Phi = binner.bin(Phi_theory)
    if plots:
        plt.figure(figsize=(8,6))
        plt.semilogy(cents, hat_Phi)
        plt.ylabel(r'$\Phi(L)$')
        plt.xlabel(r'$L$')
        plt.show()
    return hat_Phi

def l2integral(l1xv, l1yv, Lv, l2xv, l2yv, ClTTunlensed, ClTTtotal, 
    Clphiphi = None, l2min = 30, l2max = 3000):
    
    #Add axes in order to vectorize integral
    l1x = np.expand_dims(np.expand_dims(np.expand_dims(l1xv,axis=0),axis=-1),
        axis=-1)
    l1y = np.expand_dims(np.expand_dims(np.expand_dims(l1yv,axis=0),axis=-1),
        axis=-1)
    L = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(Lv,axis=-1),
        axis=-1),axis=-1),axis=-1)
    
    ## Testing:
    #print(np.shape(l1x))
    #print(np.shape(l1y))
    #print(np.shape(L))
    
    l2dotl2minusl1 = l2xv*(l2xv-l1x)+l2yv*(l2yv-l1y)
    Lminusl2dotl1minusl2 = (L-l2xv)*(l1x-l2xv) + (-l2yv)*(l1y-l2yv)
    l2dotl2minusL = l2xv*(l2xv-L)+l2yv*l2yv
    modl2 = np.sqrt(l2xv**2+l2yv**2)
    modLminusl2 = np.sqrt((L-l2xv)**2 + l2yv**2)
    modl1minusl2 = np.sqrt((l1x-l2xv)**2 + (l1y-l2yv)**2)
    
    maskl2 = np.ones(np.shape(modl2),dtype = int)
    maskl2[np.where(modl2 < l2min)] = 0
    maskl2[np.where(modl2 > l2max)] = 0

    maskLminusl2 = np.ones(np.shape(modLminusl2),dtype = int)
    maskLminusl2[np.where(modLminusl2 < l2min)] = 0
    maskLminusl2[np.where(modLminusl2 > l2max)] = 0
    
    hfilter = maskl2*ClTTunlensed[np.rint(modl2).astype(int)]/ClTTtotal[np.rint(modl2).astype(int)] \
            * maskLminusl2*ClTTunlensed[np.rint(modLminusl2).astype(int)]/ClTTtotal[np.rint(modLminusl2).astype(int)]
    
    integrand = l2dotl2minusl1*Lminusl2dotl1minusl2*l2dotl2minusL \
                * 2. * hfilter * ClTTunlensed[np.rint(modl2).astype(int)] * ClTTunlensed[np.rint(modLminusl2).astype(int)]
    if Clphiphi is not None:
        integrand *= Clphiphi[np.rint(modl1minusl2).astype(int)]
    
    deltal2x = l2xv[0,1]-l2xv[0,0]
    deltal2y = l2yv[1,0]-l2yv[0,0]
    
    integrand*= deltal2x*deltal2y/((2*np.pi)**2)
    
    ## Testing:
    #print(np.shape(integrand))
    
    return np.sum(integrand, axis=(-2,-1))

def l1integral(l1xv, l1yv, Lv, l2xv, l2yv, ClTTunlensed, ClTTtotal, 
    Clphiphi = None, l1min = 6000, l1max = 8000, l2min = 30, l2max = 3000):

    #Add axes in order to vectorize integral
    L = np.expand_dims(np.expand_dims(Lv,axis=-1),axis=-1)
    
    ## Testing:
    #print(np.shape(l1x))
    #print(np.shape(l1y))
    #print(np.shape(L))
    
    l1dotl1minusL = l1xv*(l1xv-L)+l1yv*l1yv
    modl1 = np.sqrt(l1xv**2+l1yv**2)
    modLminusl1 = np.sqrt((L-l1xv)**2 + l1yv**2)
    
    maskl1 = np.ones(np.shape(modl1),dtype = int)
    maskl1[np.where(modl1 < l1min)] = 0
    maskl1[np.where(modl1 > l1max)] = 0

    maskLminusl1 = np.ones(np.shape(modLminusl1),dtype = int)
    maskLminusl1[np.where(modLminusl1 < l1min)] = 0
    maskLminusl1[np.where(modLminusl1 > l1max)] = 0
    
    gfilter = maskl1*1./ClTTtotal[np.rint(modl1).astype(int)] \
            * maskLminusl1*1./ClTTtotal[np.rint(modLminusl1).astype(int)]
    
    integrand = l1dotl1minusL * gfilter 
    
    integrand *= l2integral(l1xv = l1xv, 
                   l1yv = l1yv, 
                   Lv = Lv, 
                   l2xv = l2xv, 
                   l2yv = l2yv, 
                   ClTTunlensed = ClTTunlensed, 
                   ClTTtotal = ClTTtotal, 
                   Clphiphi = Clphiphi, 
                   l2min = l2min, 
                   l2max = l2max)
   
    deltal1x = l1xv[0,1]-l1xv[0,0]
    deltal1y = l1yv[1,0]-l1yv[0,0]
    
    integrand*= deltal1x*deltal1y/((2*np.pi)**2)
    
    ## Testing:
    #print(np.shape(integrand))
    
    return np.sum(integrand, axis=(-2,-1))

def CalcBiasExp(uCl, tCl, Clpp, l1min, l1max, l2min, l2max, Lv, fCl=None, 
    dl1=75, dl2=100, useMC=True, useC=True):
    """Return the normalization AL and expected Psi_L for the given spectra.

    Return the numerically integrated normalization factors AL as well as 
    expected Psi_L outputs for SCALE. Numerical accuracy can be tuned with 
    the dl1 and dl2 parameters. A Cython implementation is enabled by default 
    with the useC parameter; otherwise, a pure Python implementation is used.

    Parameters
    ----------
    uCl: 1d-array
        The expected unlensed CMB temperature power spectrum.
    tCl: 1d-array
        A fiducial total, observed CMB temperature power spectrum.
    Clpp: 1d-array
        A fiducial lensing potential (phiphi) power spectrum.
    l1min: int
        The lower limit to the small-scale integral, corrsponding to l1.
    l1max: int
        The upper limit to the small-scale integral, corresponding to l1.
    l2min: int
        The lower limit to the large-scale integral, corresponding to l2.
    l2max: int
        The upper limit to the large-scale integral, corresponding to l2.
    Lv: 1d-array
        The Lcheck values (centered in bins) at which to evaluate Al, Psi_L.
    fCl: 1d-array, default=None
        If given, replaces two factors of uCl with fCl. Physically, this 
        corresponds to the filter (fCl) cosmology being different from that 
        of the true (uCl) cosmology.
    dl1: int
        The integration step size for the l1 integral. 
        (Bigger effect on accuracy)
    dl2: int
        The integration step size for the l2 integral.
    useMC: bool, default=True
        If True, use Monte Carlo integration. Else, check useC.
    useC: bool, default=True
        If True, use the Cython implementation (~4*Ncores times faster). 
        Else, use the pure Python implementation. Does nothing if useMC=True.
    
    Returns
    -------
    ALv: 1d-array
        The normalization factors evaluated at each Lv.
    PsiLv: 1d-array
        The expected SCALE Psi_Lcheck evaluated at each Lv.
    """


    ## Do each L individually to save on memory usage
    ## Loop is a bit slower than vectorized calculation, but memory savings is factor of nL

    ALv = np.zeros(np.shape(Lv)[0])
    PsiLv = np.zeros(np.shape(Lv)[0])
    if fCl is None:
        fCl = uCl

    if useMC:
        for iL, LL in enumerate(Lv):
            PsiLv[iL], ALv[iL] = Psi_and_A_cy_mc(LL, uCl, fCl, tCl, Clpp, l1min, l1max, 
                l2min, l2max, 200000, 1) # 200k achieves ~1% error
        ALv *= (2*np.pi)**4
        PsiLv /= (2*np.pi)**4
    elif useC:
        for iL, LL in enumerate(Lv):
            PsiLv[iL], ALv[iL] = Psi_and_A_cy(LL, uCl, fCl, tCl, Clpp, l1min, l1max, 
                l2min, l2max, dl1, dl2)
        ALv *= (2*np.pi)**4
        PsiLv /= (2*np.pi)**4
    else:
        nl1x, nl1y = (2*l1max//dl1, 2*l1max//dl1)
        l1x = np.linspace(-l1max, l1max, nl1x)
        l1y = np.linspace(-l1max, l1max, nl1y)
        l1xv, l1yv = np.meshgrid(l1x, l1y, sparse=True)  # make sparse output arrays

        ## Turnover of Psi(L) seems to depend on choice of sampling here
        ## Increase for better precision at cost of speed and memory usage
        nl2x, nl2y = (2*l2max//dl2, 2*l2max//dl2)
        l2x = np.linspace(-l2max, l2max, nl2x)
        l2y = np.linspace(-l2max, l2max, nl2y)
        l2xv, l2yv = np.meshgrid(l2x, l2y, sparse=True)  # make sparse output arrays
        for iL, LL in enumerate(Lv):
            ALv[iL] = 1./l1integral(l1xv, l1yv, [LL], l2xv, l2yv, 
                uCl, tCl, Clphiphi = None, 
                l1min = l1min, l1max = l1max, l2min = l2min, l2max = l2max)
            PsiLv[iL] = l1integral(l1xv, l1yv, [LL], l2xv, l2yv, 
                uCl, tCl, Clphiphi = Clpp, 
                l1min = l1min, l1max = l1max, l2min = l2min, l2max = l2max)*ALv[iL]
    return ALv, PsiLv

def SCALE(map_in, map_delens=None, l1min=6000, l1max=10000, l2min=0, l2max=3000, 
           DLv=50, uCl=None, lCl=None, Nl=None, Clpp=None, w=0., b=0., 
           compute_bias=False):
    """Return the SCALE cross-spectrum Psi_Lcheck for the given map_in.

    Perform the SCALE estimator for small-scale CMB lensing on map_in. 
    In general, this function pre-processes map_in according to the other 
    parameters in two different ways, and computes the cross-spectrum of 
    these two products. Multipole limits for filtering are set by l1min, 
    l1max, l2min, and l2max. Options are available to provide fiducial 
    power spectra for filtering. Also numerically computes the normalization 
    factor A_Lcheck and the expected Psi_Lcheck from theory if compute_bias 
    is selected.

    Parameters
    ----------
    map_in: nd_map (pixell.enmap)
        The _flat-sky_ CMB map to apply the SCALE method to.
    map_delens: nd_map (pixell.enmap), default=None
        A delensed or unlensed version of map_in. If given, this map is 
        used to compute the lambda (corresponding to large-scales) map.
    l1min: int, default=6000
        The lower limit to the high-pass filter, corrsponding to l1.
    l1max: int, default=10000
        The upper limit to the high-pass filter, corresponding to l1.
    l2min: int, default=0
        The lower limit to the low-pass filter, corresponding to l2.
    l2max: int, default=3000
        The upper limit to the low-pass filter, corresponding to l2.
    DLv: int, default=50
        The size of Lcheck bins in the output.
    uCl: 1d-array, default=None
        A fiducial unlensed CMB temperature power spectrum for filtering.
        If None, no filter is applied.
    lCl: 1d-array, default=None
        A fiducial lensed CMB temperature power specturm for filtering.
        If None, no filter is applied.
    Nl: 1d-array, default=None
        A fiducial CMB temperature noise spectrum for filtering.
        If None, a set of Nls are computed with w, sg.
    Clpp: 1d-array, default=None
        A fiducial lensing potential (phiphi) power spectrum. Required if 
        compute_bias=True.
    w: float, default=0.
        The white-noise level (in uK-arcmin) for computation of Nl if no 
        Nls are given.
    b: float, default=0.
        The beam FWHM (in arcmin) for computation of Nl if no Nls are 
        given.
    compute_bias: bool, default=False
        If True, return also the numerically computed normalization factors 
        A_Lcheck as well as the expected Psi_Lcheck spectrum. This option 
        requires inputs for uCl, lCl, and Clpp.
    
    Returns
    -------
    Lv: 1d-array
        The centers of the Lcheck bins (set by DLv) of the other outputs.
    CLvls: 1d-array
        The un-normalized C_Lcheck^{lambda,sigma} cross-spectrum of map_in.
    ALv: 1d-array
        The normalization for CLls such that Psi_Lcheck = AL*CLls. 
        Only returned if compute_bias=True.
    PsiLv: 1d-array
        The expected theory values for Psi_Lcheck at the same each bin center.
        Only returned if compute_bias=True.
    """
    if uCl is None:
        uCl = np.ones(l1max)
    if lCl is None:
        lCl = np.ones(l1max)
    ell = np.arange(len(lCl), dtype=np.float64)
    if Nl is None:
        Nl = (w*np.pi/180./60.)**2. / np.exp(-ell*(ell+1)*(b*np.pi/180./60. / np.sqrt(8.*np.log(2)))**2)
    shape, wcs = map_in.shape, map_in.wcs
    lmap = map_in.modlmap()
    if map_delens is not None:
        Tlp = WienerFilter(map_delens, ell, uCl, uCl, Nl, lmin=l2min, lmax=l2max, grad=True)
    else:
        Tlp = WienerFilter(map_in, ell, uCl, lCl, Nl, lmin=l2min, lmax=l2max, grad=True)
    lam = Tlp[0]**2 + Tlp[1]**2
    Thp = InvVarFilter(map_in, ell, lCl, Nl, lmin=l1min, lmax=l1max, grad=True)
    sig = Thp[0]**2 + Thp[1]**2
    fc = FourierCalc(shape, wcs)
    p2d,_,_ = fc.power2d(lam, sig) # 2D cross-spectrum between lambda & sigma maps
    nLv = l1max // DLv
    bins = np.arange(0, nLv*DLv+1, DLv)
    binner = bin2D(lmap,bins)
    Lv, CLvls = binner.bin(p2d)
    if compute_bias:
        ALv, PsiLv = CalcBiasExp(uCl, lCl+Nl, Clpp, l1min, l1max, l2min, l2max, Lv[Lv<l2max])
        return Lv, CLvls, ALv, PsiLv
    else:
        return Lv, CLvls, None, None
    
def SCALEerror(Lv, CLv, ALv, fsky=1., dLv=1., normed=False):
    """Return the minimum expected variance per mode of the SCALE spectrum.

    Parameters
    ----------
    Lv: 1d-array
        The centers of the Lcheck bins of the other inputs.
    CLv: 1d-array
        The un-normalized C_Lcheck^{lambda,sigma} cross-spectrum of map_in.
    ALv: 1d-array
        The normalization for CLls such that Psi_Lv = ALv*CLls.
    fsky: float, default=1.
        The fraction of the sky covered by the map.
    dLv: float or array, default=1.
        The size of the Lcheck bins.
    normed: bool, default=False
        If True, assumes that CLv is normalized (actually PsiLv)

    Returns
    -------
    dPsiLv: 1d-array
        The minimum expected variance per mode of the SCALE spectrum.
    """
    if normed:
        PsiLv = CLv
    else:
        PsiLv = CLv*ALv
    return np.sqrt((PsiLv**2 + 4*ALv) / (fsky*dLv*(2*Lv+1)))
