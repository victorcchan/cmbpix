import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap, utils as putils
from cmbpix.utils import *
from orphics.maps import binned_power, FourierCalc
from orphics.stats import bin2D
import symlens as s
from scipy.interpolate import CubicSpline

def ApplyFilter(map_in, Fl):
    return enmap.ifft(enmap.fft(map_in)*Fl).real

def WienerFilter(map_in, ells, lCls, Nls, lmin, lmax, grad=False):
    cs = CubicSpline(ells, lCls / (lCls + Nls))
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
    el = np.arange(0, 3000, 1, dtype=np.float64)
    prefactor = (s.l1*s.l1-s.Ldl1)*(s.l1*s.l1-s.Ldl1)
    spectra = s.e('uC_T_T_l1')*s.e('uC_T_T_l1')/s.e('tC_T_T_l1') * s.e('uC_T_T_l2')*s.e('uC_T_T_l2')/s.e('tC_T_T_l2')
    AL_integrand = prefactor*spectra
    feed_dict = {}
    feed_dict['uC_T_T'] = s.interp(el,lCls[:3000])(lmap)
    feed_dict['tC_T_T'] = s.interp(el,lCls[:3000])(lmap)+(w*np.pi/180./60.)**2./s.gauss_beam(lmap,sg)**2.
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

def CalcExp(shape, wcs, lCls, w, sg, Clpp, lmin, lmax, lbin, plots=False):
    # shape, wcs = map_in.shape, map_in.wcs
    lmap = enmap.modlmap(shape, wcs)
    ells = np.arange(0.,3000.,1, dtype=np.float64)
    ctt = lCls[:3000]
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

def FFTest(map_in, ldT=3000, lmin=6000, lmax=10000, lbins=50, 
           lCls=None, Nls=None, Clpp=None, w=0., sg=0., 
           apply_bias=False, plots=None, show_horn=False):
    if lCls is None:
        lCls = np.ones(lmax)
    ell = np.arange(len(lCls), dtype=np.float64)
    if Nls is None:
        Nls = (w*np.pi/180./60.)**2. / np.exp(-ell*(ell+1)*(sg*np.pi/180./60.)**2)
    shape, wcs = map_in.shape, map_in.wcs
    lmap = map_in.modlmap()
    Tlp = WienerFilter(map_in, ell, lCls, Nls, lmin=0, lmax=ldT, grad=True)
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
        abplot = False
        if plots == 'all':
            abplot = True
        c, AL, BL = CalcBias(shape, wcs, ell, ldT, lmin, lmax, bins[bins<=3000], 
                 lCls, Nls, w, sg, Clpp, plots=abplot)
    if Clpp is not None:
        phiplot = False
        if plots == 'all':
            phiplot = True
        Phi = CalcExp(shape, wcs, lCls, w, sg, Clpp, lmin, lmax, bins[bins<=3000], plots=phiplot)
    if plots:
        plt.figure(figsize=(8,6))
        plt.plot(cents, p1d)
        plt.axhline(0, c='k', ls=':')
        plt.xlabel(r"$L$")
        plt.ylabel(r"$C_L^{GK}$")
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(8,6))
        plt.plot(cents[cents<3000], p1d[cents<3000] * AL)
        plt.axhline(0, c='k', ls=':')
        if Clpp is not None:
            plt.plot(c, Phi, c='k', ls=':')
            if show_horn:
                plt.fill_between(cents[cents<3000], ex-2*np.sqrt(AL), 
                                 ex+2*np.sqrt(AL), alpha=0.5)
        plt.xlabel(r"$L$")
        plt.ylabel(r"$C_L^{GK}$")
        plt.tight_layout()
        plt.show()
    if apply_bias and Clpp is not None:
        return cents, p1d, AL, BL, Phi
    else:
        return cents, p1d