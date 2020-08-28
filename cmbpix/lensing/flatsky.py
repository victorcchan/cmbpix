import numpy as np
from pixell import enmap, utils
from cmbpix.utils import *
from cmbpix.lensing.estimator import LensingEstimator
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def _lin(x, *p):
    """A generic line function.

    Return y = mx + b using p[0] = b, p[1] = m.

    Parameters
    ----------
    x: real, array of reals
        Point(s) at which to evaluate function.
    p: array of size 2
        The linear coefficients of the function: p = [b, m].

    Returns
    -------
    y: real, array of reals
        The linear function output(s) evaluated at x.
    """
    return p[0] + p[1]*x

class FlatSkyLens(LensingEstimator):
    """Estimator for small scale lensing in a flat sky CMB map.

    An object for reducing a pixell CMB map of the cosmic microwave 
    background in (mostly) pixel-space to estimate the effects 
    of small angular scale weak lensing due to structure formation.

    Attributes
    ----------
    map_in: ndmap
        The input CMB map in pixell format.
    ldT: value, default=2000
        The maximum ell to consider for the large scale background 
        temperature gradient.
    lmin: value, default=3000
        The minimum ell to consider for the small scale temperature 
        fluctuations.
    lmax: value, default=None
        The maximum ell to consider for the small scale temperature 
        fluctuations. If None, there is no cutoff.

    """

    def __init__(self, cmbmap, ldT=2000, lmin=3000, lmax=np.inf, 
                    patchsize=40):
        """Initiate the estimator.

        Parameters
        ----------
        cmbmap: ndmap
            A CMB map.
        ldT: value, default=2000
            The maximum ell to consider for the large scale background 
            temperature gradient.
        lmin: value, default=3000
            The minimum ell to consider for the small scale temperature 
            fluctuations.
        lmax: value, default=inf
            The maximum ell to consider for the small scale temperature 
            fluctuations. If none given, there is no cutoff.
        patchsize: int, default=40
            The side length of patches (in number of pixels) for the map to be 
            broken up into to gather local lensing statistics.

        """
        self.map_in = cmbmap
        self.ldT = ldT
        self.lmin = lmin
        self.lmax = lmax
        self._p = patchsize
        # Derived attributes
        self._ly, self._lx = self.map_in.lmap()
        self._lmod = self.map_in.modlmap()

    def gather_patches(self):
        """Assemble patch statistics relevant to lensing at small scales.

        Compute the small scale (ell > lmin) temperature power at different 
        patches across the sky as well as the average amplitude of the 
        background temperature gradient (ell < ldT).

        """
        self._edge = 0 # Not throwing away edge pixels
        m_fft = enmap.fft(self.map_in)
        hp = np.zeros(self.map_in.shape)
        hp[np.where((self._lmod > self.lmin) & (self._lmod < self.lmax))] = 1.
        self._Tss = enmap.ifft(m_fft * hp)
        self._dTy, self._dTx = gradient_flat(self.map_in, self.ldT)
        self._dT = np.sqrt(self._dTx**2 + self._dTy**2)
        # Scale geometry for lower res map of patches
        pshp, pwcs = enmap.scale_geometry(self.map_in.shape, 
                                            self.map_in.wcs, 1./self._p)
        self._T2patch = enmap.zeros(pshp, pwcs)
        self._dTxpatch = enmap.zeros(pshp, pwcs)
        self._dTypatch = enmap.zeros(pshp, pwcs)
        Trs = self._Tss[:pshp[-2]*self._p,
                        :pshp[-1]*self._p].reshape([pshp[-2], self._p, 
                                                    pshp[-1], self._p])
        dTxrs = self._dTx[:pshp[-2]*self._p,
                            :pshp[-1]*self._p].reshape([pshp[-2], self._p, 
                                                        pshp[-1], self._p])
        dTyrs = self._dTy[:pshp[-2]*self._p,
                            :pshp[-1]*self._p].reshape([pshp[-2], self._p, 
                                                        pshp[-1], self._p])
        self._T2patch[:,:] = np.var(Trs, axis=(1,3))
        self._dTypatch[:,:] = np.mean(dTyrs, axis=(1,3))
        self._dTxpatch[:,:] = np.mean(dTxrs, axis=(1,3))
        self._dT2patch = self._dTxpatch**2 + self._dTypatch**2


    def gather_patches_cos(self):
        """Assemble patch statistics for small scale lensing with cos filter.
        
        Compute the small scale (ell > 3000) temperature power at different 
        patches across the sky as well as the average amplitude of the 
        background temperature gradient (ell < 2000). For the small scale 
        statistics, also apply a filter in Fourier space such that:

        .. math::
            f_\\ell = \\cos(\\hat{\\ell}\\cdot\\hat{\\nabla T})

        """
        self._edge = 5 # Edge pixels to throw away
        p = self._p
        m_fft = enmap.fft(self.map_in)
        hp = np.zeros(self.map_in.shape)
        hp[np.where((self._lmod > self.lmin) & (self._lmod < self.lmax))] = 1.
        self._Tss = enmap.ifft(m_fft * hp)
        self._dTy, self._dTx = gradient_flat(self.map_in, self.ldT)
        self._dT = np.sqrt(self._dTx**2 + self._dTy**2)
        # Scale geometry for lower res map of patches
        pshp, pwcs = enmap.scale_geometry(self.map_in.shape, 
                                            self.map_in.wcs, 1./self._p)
        self._T2patch = enmap.zeros(pshp, pwcs)
        self._dTxpatch = enmap.zeros(pshp, pwcs)
        self._dTypatch = enmap.zeros(pshp, pwcs)
        self._T_sub = np.zeros((pshp[-2], pshp[-1], p, p))
        for i in range(self._T2patch.shape[-2]):
            for j in range(self._T2patch.shape[-1]):
                self._dTypatch[i,j] = np.mean(self._dTy[i*p:(i+1)*p, 
                                                        j*p:(j+1)*p])
                self._dTxpatch[i,j] = np.mean(self._dTx[i*p:(i+1)*p, 
                                                        j*p:(j+1)*p])
                Tp = self._Tss[i*p:(i+1)*p,j*p:(j+1)*p]
                lsuby, lsubx = Tp.lmap()
                lsubmod = Tp.modlmap()
                lsubmod[0,0] = 1.
                fl = 1.j*(lsubx*self._dTxpatch[i,j] + \
                            lsuby*self._dTypatch[i,j]) / \
                        (lsubmod * np.sqrt(self._dTxpatch[i,j]**2 + \
                                            self._dTypatch[i,j]**2))
                fl[0,0] = 0.
                self._T_sub[i,j,:,:] = enmap.ifft(enmap.fft(Tp)*fl).real
                # Throw away pixels with edge effects
                self._T2patch[i,j] = np.var(self._T_sub[i,j,5:-5,5:-5])
        self._dT2patch = self._dTxpatch**2 + self._dTypatch**2

    def gather_patches_cos2(self):
        """Assemble patch statistics for small scale lensing with cos^2 filter.
        
        Compute the small scale (ell > 3000) temperature power at different 
        patches across the sky as well as the average amplitude of the 
        background temperature gradient (ell < 2000). For the small scale 
        statistics, also apply a filter in Fourier space such that:

        .. math::
            f_\\ell = \\cos^2(\\hat{\\ell}\\cdot\\hat{\\nabla T})

        """
        self._edge = 3 # Edge pixels to throw away
        p = self._p
        m_fft = enmap.fft(self.map_in)
        hp = np.zeros(self.map_in.shape)
        hp[np.where((self._lmod > self.lmin) & (self._lmod < self.lmax))] = 1.
        self._Tss = enmap.ifft(m_fft * hp)
        self._dTy, self._dTx = gradient_flat(self.map_in, self.ldT)
        self._dT = np.sqrt(self._dTx**2 + self._dTy**2)
        # Scale geometry for lower res map of patches
        pshp, pwcs = enmap.scale_geometry(self.map_in.shape, 
                                            self.map_in.wcs, 1./self._p)
        self._T2patch = enmap.zeros(pshp, pwcs)
        self._dTxpatch = enmap.zeros(pshp, pwcs)
        self._dTypatch = enmap.zeros(pshp, pwcs)
        self._T_sub = np.zeros((pshp[-2], pshp[-1], p, p))
        for i in range(self._T2patch.shape[-2]):
            for j in range(self._T2patch.shape[-1]):
                self._dTypatch[i,j] = np.mean(self._dTy[i*p:(i+1)*p, 
                                                        j*p:(j+1)*p])
                self._dTxpatch[i,j] = np.mean(self._dTx[i*p:(i+1)*p, 
                                                        j*p:(j+1)*p])
                Tp = self._Tss[i*p:(i+1)*p,j*p:(j+1)*p]
                lsuby, lsubx = Tp.lmap()
                lsubmod = Tp.modlmap()
                lsubmod[0,0] = 1. # Avoid divide by 0; set fl here to 0 later
                fl = (lsubx*self._dTxpatch[i,j] + \
                        lsuby*self._dTypatch[i,j])**2 / \
                    (lsubmod * np.sqrt(self._dTxpatch[i,j]**2 + \
                                        self._dTypatch[i,j]**2))**2
                fl[0,0] = 0.
                self._T_sub[i,j,:,:] = enmap.ifft(enmap.fft(Tp)*fl).real
                # Throw away pixels with edge effects
                self._T2patch[i,j] = np.var(self._T_sub[i,j,3:-3,3:-3])
        self._dT2patch = self._dTxpatch**2 + self._dTypatch**2

    def fit_binpts(self, bins=50, plot=False):
        """
        
        """
        g_ord = np.argsort(self._dT2patch.flatten())
        dT_ord = self._dT2patch.flatten()[g_ord]
        T_ord = self._T2patch.flatten()[g_ord]
        g_bin = np.exp(np.linspace(np.log(dT_ord[0]), 
                                    np.log(dT_ord[-1]), bins))
        bin_ranges = []
        nbin = []
        for b in range(bins-1):
            bin_ranges.append([g_bin[b], g_bin[b+1]])
            nbin.append(g_ord[np.where((dT_ord > g_bin[b]) & 
                                        (dT_ord < g_bin[b+1]))].size)
        dT2 = np.zeros((len(bin_ranges), 2))
        T2 = np.zeros((len(bin_ranges), 2))
        for b in range(len(bin_ranges)):
            bin_inds = np.where((dT_ord > bin_ranges[b][0]) & \
                                (dT_ord < bin_ranges[b][1]))
            n = dT_ord[bin_inds].size
            dT2[b,:] = [dT_ord[bin_inds].mean(), \
                        dT_ord[bin_inds].std() / np.sqrt(n), \
                       ]
            T2[b,:] = [T_ord[bin_inds].mean(), \
                       T_ord[bin_inds].std() / np.sqrt(n), \
                      ]
        #gb = np.where(np.isfinite(T2[:,0]))
        # Good bins that contain data points
        gb = np.where(np.array(nbin) > 5)
        popt, pcov = curve_fit(_lin, dT2[gb][1:-1,0], T2[gb][1:-1,0], 
                                [1, 1e-10], sigma=T2[gb][1 :-1,1], 
                                absolute_sigma=True)

        self.line = popt
        self.dline = pcov
        
        if plot:
            plt.figure(figsize=(12,8))
            plt.plot(dT_ord.flatten(), T_ord.flatten(), '.')
            plt.errorbar(dT2[gb][:,0], T2[gb][:,0], xerr=dT2[gb][:,1], 
                            yerr=T2[gb][:,1], fmt='.', c='r')
            plt.plot(g_bin, _lin(g_bin, *popt), c='C1')
            plt.xscale('log')
            plt.xlabel(r"$|\nabla T|^2~[\mu{\rm K}^2/rad^2$]")
            plt.ylabel(r"$\sigma_T^2~[\mu{\rm K}^2]$")
            plt.tight_layout()
            plt.show()
            plt.close()

    def fit_binerr(self, bins=50, edges=None, errs=None, plot=False, 
                    showerr=True):
        """
        
        """
        g_ord = np.argsort(self._dT2patch.flatten())
        dT_ord = self._dT2patch.flatten()[g_ord]
        T_ord = self._T2patch.flatten()[g_ord]
        T_err = np.zeros(T_ord.size)
        if errs is None and edges is None:
            N = T_ord.size // bins # Roughly equal number of points in bins
            self.bin_edges = dT_ord[np.arange(bins, dtype=int)*N]
            self.bin_edges[0] = 0 # Ensure we get everything
            self.errs = np.zeros(bins)
            for b in range(bins):
                self.errs[b] = T_ord[N*b:N*(b+1)].std()
                T_err[N*b:] = T_ord[N*b:N*(b+1)].std()
        else:
            self.bin_edges = edges
            self.bin_edges[0] = 0 # Ensure we get everything
            self.errs = errs
            for i, b in enumerate(self.bin_edges):
                T_err[np.where(dT_ord > b)] = self.errs[i]
        popt, pcov = curve_fit(_lin, dT_ord, T_ord, 
                                [1, 1e-10], sigma=T_err, 
                                absolute_sigma=True)
        self.line = popt
        self.dline = pcov
        
        if plot:
            plt.figure(figsize=(12,8))
            if showerr:
                plt.errorbar(dT_ord.flatten(), T_ord.flatten(), 
                                yerr=T_err, fmt='.')
            else:
                plt.plot(dT_ord.flatten(), T_ord.flatten(), '.')
            gs = np.linspace(dT_ord.min(), dT_ord.max(), 200)
            plt.plot(gs, _lin(gs, *popt), c='C1')
            plt.xscale('log')
            plt.xlabel(r"$|\nabla T|^2~[\mu{\rm K}^2/rad^2$]")
            plt.ylabel(r"$\sigma_T^2~[\mu{\rm K}^2]$")
            plt.tight_layout()
            plt.show()
            plt.close()

    def fit_pts(self, bins=50, plot=False, showerr=True):
        """
        
        """
        g_ord = np.argsort(self._dT2patch.flatten())
        dT_ord = self._dT2patch.flatten()[g_ord]
        T_ord = self._T2patch.flatten()[g_ord]
        T_err = np.ones(T_ord.size)*T_ord.std()
        popt, pcov = curve_fit(_lin, dT_ord, T_ord, 
                                [1, 1e-10], sigma=T_err, 
                                absolute_sigma=True)
        self.line = popt
        self.dline = pcov
        
        if plot:
            plt.figure(figsize=(12,8))
            if showerr:
                plt.errorbar(dT_ord.flatten(), T_ord.flatten(), 
                                yerr=T_err, fmt='.')
            else:
                plt.plot(dT_ord.flatten(), T_ord.flatten(), '.')
            gs = np.linspace(dT_ord.min(), dT_ord.max(), 200)
            plt.plot(gs, _lin(gs, *popt), c='C1')
            plt.xscale('log')
            plt.xlabel(r"$|\nabla T|^2~[\mu{\rm K}^2/rad^2$]")
            plt.ylabel(r"$\sigma_T^2~[\mu{\rm K}^2]$")
            plt.tight_layout()
            plt.show()
            plt.close()

    def fit_ptserr(self, bins=50, plot=False, showerr=True):
        """
        
        """
        n = (self._p-self._edge)**2 # Number of pixels/patch
        g_ord = np.argsort(self._dT2patch.flatten())
        dT_ord = self._dT2patch.flatten()[g_ord]
        T_ord = self._T2patch.flatten()[g_ord]
        T_err = T_ord * np.sqrt(2./(n-1))
        popt, pcov = curve_fit(_lin, dT_ord, T_ord, 
                                [1, 1e-10], sigma=T_err, 
                                absolute_sigma=True)
        self.line = popt
        self.dline = pcov
        
        if plot:
            plt.figure(figsize=(12,8))
            if showerr:
                plt.errorbar(dT_ord.flatten(), T_ord.flatten(), 
                                yerr=T_err, fmt='.')
            else:
                plt.plot(dT_ord.flatten(), T_ord.flatten(), '.')
            gs = np.linspace(dT_ord.min(), dT_ord.max(), 200)
            plt.plot(gs, _lin(gs, *popt), c='C1')
            plt.xscale('log')
            plt.xlabel(r"$|\nabla T|^2~[\mu{\rm K}^2/rad^2$]")
            plt.ylabel(r"$\sigma_T^2~[\mu{\rm K}^2]$")
            plt.tight_layout()
            plt.show()
            plt.close()