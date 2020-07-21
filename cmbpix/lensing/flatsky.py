import numpy as np
from pixell import enmap, utils
from cmbpix.utils import *
from cmbpix.lensing.estimator import LensingEstimator

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

    def gather_patches():
        """Assemble patch statistics relevant to lensing at small scales.

        Compute the small scale (ell > lmin) temperature power at different 
        patches across the sky as well as the average amplitude of the 
        background temperature gradient (ell < ldT).

        """
        m_fft = enmap.fft(self.map_in)
        hp = np.zeros(map_in.shape)
        hp[np.where((self._lmod > self.lmin) & (self._lmod < self.lmax))] = 1.
        self._Tss = enmap.ifft(m_fft * hp)
        self._dTy, self._dTx = gradient_flat(self.map_in, self.ldT)
        self._dT = np.sqrt(self._dTx**2 + self._dTy**2)
        # Scale geometry for lower res map of patches
        pshp, pwcs = enmap.scale_geometry(self.map_in.shape, 
                                            self.map_in.wcs, 1./self._p)
        self._Tpatch = enmap.zeros(pshp, pwcs)
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
        self._Tpatch[:,:] = np.var(Trs, axis=(1,3))
        self._dTypatch[:,:] = np.mean(dTyrs, axis=(1,3))
        self._dTxpatch[:,:] = np.mean(dTxrs, axis=(1,3))

    def gather_patches_cos():
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
        self._Tpatch = enmap.zeros(pshp, pwcs)
        self._dTxpatch = enmap.zeros(pshp, pwcs)
        self._dTypatch = enmap.zeros(pshp, pwcs)
        self._T_sub = np.zeros((pshp[-2], pshp[-1], p, p))
        for i in range(self._Tpatch.shape[-2]):
            for j in range(self._Tpatch.shape[-1]):
                self._dTypatch[i,j] = np.mean(self._dTy[i*p:(i+1)*p, 
                                                        j*p:(j+1)*p])
                self._dTxpatch[i,j] = np.mean(self._dTx[i*p:(i+1)*p, 
                                                        j*p:(j+1)*p])
                Tp = self._Tss[i*p:(i+1)*p,j*p:(j+1)*p]
                lsuby, lsubx = Tp.lmap()
                lsubmod = Tp.modlmap()
                fl = 1.j*(lsubx*self._dTxpatch[i,j] + \
                            lsuby*self._dTypatch[i,j]) / \
                        (lsubmod * np.sqrt(self._dTxpatch[i,j]**2 + \
                                            self._dTypatch[i,j]**2))
                fl[0,0] = 0.
                self._T_sub[i,j,:,:] = enmap.ifft(enmap.fft(Tp)*fl).real
                # Throw away pixels with edge effects
                self._Tpatch[i,j] = np.var(self._T_sub[i,j,5:-5,5:-5])

    def gather_patches_cos2():
        """Assemble patch statistics for small scale lensing with cos^2 filter.
        
        Compute the small scale (ell > 3000) temperature power at different 
        patches across the sky as well as the average amplitude of the 
        background temperature gradient (ell < 2000). For the small scale 
        statistics, also apply a filter in Fourier space such that:

        .. math::
            f_\\ell = \\cos^2(\\hat{\\ell}\\cdot\\hat{\\nabla T})

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
        self._Tpatch = enmap.zeros(pshp, pwcs)
        self._dTxpatch = enmap.zeros(pshp, pwcs)
        self._dTypatch = enmap.zeros(pshp, pwcs)
        self._T_sub = np.zeros((pshp[-2], pshp[-1], p, p))
        for i in range(self._Tpatch.shape[-2]):
            for j in range(self._Tpatch.shape[-1]):
                self._dTypatch[i,j] = np.mean(self._dTy[i*p:(i+1)*p, 
                                                        j*p:(j+1)*p])
                self._dTxpatch[i,j] = np.mean(self._dTx[i*p:(i+1)*p, 
                                                        j*p:(j+1)*p])
                Tp = self._Tss[i*p:(i+1)*p,j*p:(j+1)*p]
                lsuby, lsubx = Tp.lmap()
                lsubmod = Tp.modlmap()
                fl = (lsubx*self._dTxpatch[i,j] + \
                        lsuby*self._dTypatch[i,j])**2 / \
                    (lsubmod * np.sqrt(self._dTxpatch[i,j]**2 + \
                                        self._dTypatch[i,j]**2))**2
                fl[0,0] = 0.
                self._T_sub[i,j,:,:] = enmap.ifft(enmap.fft(Tp)*fl).real
                # Throw away pixels with edge effects
                self._Tpatch[i,j] = np.var(self._T_sub[i,j,5:-5,5:-5])