import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap, utils as putils
#from cmbpix.utils import *
from orphics.maps import binned_power, FourierCalc
from orphics.stats import bin2D
import symlens as s
from scipy.interpolate import CubicSpline

def ApplyFilter(map_in, Fl):
    return enmap.ifft(enmap.fft(map_in)*Fl).real

def WienerFilter(map_in, ells, uCls, lCls, Nls, lmin, lmax, grad=False):
    cs = CubicSpline(ells, uCls / (lCls + Nls))
    lmap = map_in.modlmap()
    fl = cs(lmap)
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

def CalcBiasExp(shape, wcs, uCls, tCls, Clpp, l1min, l1max, l2min, l2max, lbin):
    nl1x, nl1y = (200, 200)
    l1x = np.linspace(-l1max, l1max, nl1x)
    l1y = np.linspace(-l1max, l1max, nl1y)
    l1xv, l1yv = np.meshgrid(l1x, l1y, sparse=True)  # make sparse output arrays

    ## Turnover of Phi(L) seems to depend on choice of sampling here
    ## Increase for better precision at cost of speed and memory usage
    nl2x, nl2y = (40, 40)
    l2x = np.linspace(-l2max, l2max, nl2x)
    l2y = np.linspace(-l2max, l2max, nl2y)
    l2xv, l2yv = np.meshgrid(l2x, l2y, sparse=True)  # make sparse output arrays

    # nL = l2max // lbins
    # Lmax = l2max
    # Lmin = l2min
    # Lv = np.linspace(Lmin, Lmax, nL)
    Lv = lbin


    ## Do each L individually to save on memory usage
    ## Loop is a bit slower than vectorized calculation, but memory savings is factor of nL

    AL = np.zeros(np.shape(Lv)[0])
    Phi = np.zeros(np.shape(Lv)[0])

    for iL, LL in enumerate(Lv):
        AL[iL] = 1./l1integral(l1xv, l1yv, [LL], l2xv, l2yv, 
            uCls, tCls, Clphiphi = None, 
            l1min = l1min, l1max = l1max, l2min = l2min, l2max = l2max)
        Phi[iL] = l1integral(l1xv, l1yv, [LL], l2xv, l2yv, 
            uCls, tCls, Clphiphi = Clpp, 
            l1min = l1min, l1max = l1max, l2min = l2min, l2max = l2max)*AL[iL]

    return Lv, AL, Phi

def FFTest(map_in, ldT=3000, lmin=6000, lmax=10000, lbins=50, 
           uCls=None, lCls=None, Nls=None, Clpp=None, w=0., sg=0., 
           apply_bias=False, plots=None, show_horn=False):
    if uCls is None:
        uCls = np.ones(lmax)
    if lCls is None:
        lCls = np.ones(lmax)
    ell = np.arange(len(lCls), dtype=np.float64)
    if Nls is None:
        Nls = (w*np.pi/180./60.)**2. / np.exp(-ell*(ell+1)*(sg*np.pi/180./60.)**2)
    shape, wcs = map_in.shape, map_in.wcs
    lmap = map_in.modlmap()
    Tlp = WienerFilter(map_in, ell, uCls, lCls, Nls, lmin=0, lmax=ldT, grad=True)
    G = Tlp[0]**2 + Tlp[1]**2
    Thp = InvVarFilter(map_in, ell, lCls, Nls, lmin, lmax, grad=True)
    K = Thp[0]**2 + Thp[1]**2
    fc = FourierCalc(shape, wcs)
    p2d,_,_ = fc.power2d(G, K)
    B = 10000 // lbins + 1
    bins = np.arange(B) * lbins
    binner = bin2D(lmap,bins)
    cents, p1d = binner.bin(p2d)
    if apply_bias:
        c, AL, Phi = CalcBiasExp(shape, wcs, uCls, lCls+Nls, Clpp, lmin, lmax, 0, ldT, cents[cents<ldT])
        if plots == 'all':
            plt.figure(figsize=(8,6))
            plt.semilogy(c, AL)
            plt.xlabel(r'$L$')
            plt.ylabel(r'$A(L)$')
            plt.tight_layout()
            plt.show()
            plt.close()
            plt.figure(figsize=(8,6))
            plt.semilogy(c, Phi)
            plt.xlabel(r'$L$')
            plt.ylabel(r'$\Psi(L)$')
            plt.tight_layout()
            plt.show()
            plt.close()    
    #     abplot = False
    #     if plots == 'all':
    #         abplot = True
    #     c, AL, BL = CalcBias(shape, wcs, ell, ldT, lmin, lmax, bins[bins<=ldT], 
    #              lCls, Nls, w, sg, Clpp, plots=abplot)
    # if Clpp is not None:
    #     phiplot = False
    #     if plots == 'all':
    #         phiplot = True
    #     Phi = CalcExp(shape, wcs, lCls, w, sg, Clpp, ldT, lmin, lmax, bins[bins<=ldT], plots=phiplot)
    if plots:
        plt.figure(figsize=(8,6))
        plt.plot(cents, p1d)
        plt.axhline(0, c='k', ls=':')
        plt.xlabel(r"$L$")
        plt.ylabel(r"$C_L^{GK}$")
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(8,6))
        plt.plot(cents[cents<ldT], p1d[cents<ldT] * AL)
        plt.axhline(0, c='k', ls=':')
        if Clpp is not None:
            plt.plot(c, Phi, c='k', ls=':')
            if show_horn:
                plt.fill_between(cents[cents<ldT], ex-2*np.sqrt(AL), 
                                 ex+2*np.sqrt(AL), alpha=0.5)
        plt.xlabel(r"$L$")
        plt.ylabel(r"$C_L^{GK}$")
        plt.tight_layout()
        plt.show()
    if apply_bias and Clpp is not None:
        return cents, p1d, AL, Phi
    else:
        return cents, p1d