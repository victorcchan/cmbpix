import numpy as np
from pixell import enmap, utils
from cmbpix.utils import *
from cmbpix.lensing.estimator import LensingEstimator
from scipy.optimize import curve_fit, fsolve
from scipy.special import gamma
import matplotlib.pyplot as plt
from mpi4py import MPI
try:
    import cmasher as cmr
except(ModuleNotFoundError):
    pass

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

def _Pearson3(s2, N=1., sig2=1., *p):
    """Return the PDF of the Pearson Type III distribution evaluated at s2.
    
    Evaluate the PDF of the Pearson Type III distribution at s2.
    
    Parameters
    ----------
    s2: value(s)
        The value(s) at which to evaluate the Pearson III distribution.
    N: value(s), default=1.
        The number of independent samples used to estimate s2.
    sig2: values(s), default=1.
        The width of the distribution from which the samples of s2 were drawn.
    """
    if p:
        N = p[0]
        sig2 = p[1]
    ans = (N / (2 * sig2)) ** ((N-1)/2.)
    ans *= s2 ** ((N-3)/2.)
    ans *= np.exp(-N * s2 / (2 * sig2))
    return ans / gamma((N-1) / 2.)

def _PearsonWidth(vol, xs, N, sig2, *p):
    """Return the band of s2 where the integral of Pearson III is vol.
    
    Return a, b such that the integral of the Pearson III distribution is 
    vol, and p(a) = p(b).
    
    Parameters
    ----------
    vol: value between [0,1]
        The confidence level to find for the Pearson III distribution
    xs: 1d-array
        The range of xs to consider
    N: value
        Pearson III parameter corresponding to the number of indep samples
    sig2: value
        Pearson III parameter corresponding to the spread of sample dist
        
    Returns
    -------
    [a, b]: array of size 2
        The range of x such that the confidence interval of Pearson III is vol
    """
    if p:
        N = p[0]
        sig2 = p[1]
    Px = _Pearson3(xs, N, sig2)
    imode = np.argmax(Px)
    xmode = xs[imode]
    a0 = (xmode - xs.min())/2
    def findb(a):
        Pa = Pearson3(a, N, sig2)
        bguess = xs[imode:][np.argmin(np.abs(Px[imode:] - Pa))]
        minb = lambda b : _Pearson3(b, N, sig2) - Pa
        b = fsolve(minb, bguess)[0]
        return b
    def optP(a):
        b = findb(a)
        ab = np.linspace(a, b, 1001)
        intP = np.sum(_Pearson3(ab, N, sig2) * (ab[1]-ab[0]))
        return intP - vol
    a = fsolve(optP, a0)[0]
    b = findb(a)
    return [a, b]



def _bin_mids(b):
    return (b[1:] + b[:-1]) / 2

class MPIFlatSkyLens(LensingEstimator):
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
                    patchsize=40, comm=None):
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
        if comm == None:
            raise ValueError("This class requires an MPI communicator")
        self.comm = comm
        self.commrank = self.comm.Get_rank()
        self.commsize = self.comm.Get_size()
        if self.commrank == 0:
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

    def fit_binerr(self, bins=50, edges=None, errs=None, plot=False, 
                    showerr=True, showline=True, filename=None, scale='log'):
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
        self._T_err = T_err
        self._T_ord = T_ord
        self._dT_ord = dT_ord
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
            if showline:
                plt.plot(gs, _lin(gs, *popt), c='C1')
            plt.xscale(scale)
            plt.xlabel(r"$|\nabla T|^2~[\mu{\rm K}^2/rad^2$]")
            plt.ylabel(r"$\sigma_T^2~[\mu{\rm K}^2]$")
            plt.tight_layout()
            if filename is not None:
                plt.savefig(str(filename), transparent=True)
            plt.show()
            plt.close()

    def chi2line(self, p=None):
        """Determine the reduced chi2 statistic for the given line.

        Determine the reduced chi2 statistic for the given line. If no line 
        is given, then the fitted line is used.

        Parameters
        ----------
        p: array of size 2, default=None
            The parameters of the line: [Intercept, Slope]

        Returns
        -------
        red_chi2:
            The reduced chi2 statistic for the line
        """
        diff = self._T_ord - _lin(self._dT_ord, *p)
        chi2 = np.sum(diff**2 / self._T_err**2)
        return chi2 / (self._T_ord.size - 2)

    def chi2grid(self, plot=False):
        """Compute the grid of reduced chi2 values around the fitted space.

        Compute the grid of reduced chi2 values for the space +/- 5 sigma 
        around the parameters fitted for by one of the curve_fit methods. 
        Also saves the parameters/errors inferred from the chi2 statistics.

        Parameters
        ----------
        plot: bool, default=False
            If True, generate a 2D plot of the grid of reduced chi2 values.
        """

        bgrid = np.linspace(self.line[0] - 5*np.sqrt(self.dline[0][0]), 
                            self.line[0] + 5*np.sqrt(self.dline[0][0]), 
                            100
                           )
        mgrid = np.linspace(self.line[1] - 5*np.sqrt(self.dline[1][1]), 
                            self.line[1] + 5*np.sqrt(self.dline[1][1]), 
                            100
                           )
        db = bgrid[1] - bgrid[0]
        dm = mgrid[1] - mgrid[0]
        pgrid = np.meshgrid(bgrid, mgrid, indexing='ij')
        self.npgrid = pgrid
        self.cgrid = np.zeros((100,100))
        for i, b in enumerate(bgrid):
            for j, m in enumerate(mgrid):
                self.cgrid[i,j] = self.chi2line([b, m])
        npatch = self._dT2patch.flatten().size-2
        self.cgrid *= npatch
        Pgrid = np.exp(-(self.cgrid)/2)
        self.nPgrid = Pgrid
        norm = np.sum(Pgrid * db * dm)
        mI = np.sum(Pgrid * pgrid[0] * db * dm) / norm
        dI = np.sum(Pgrid * (pgrid[0] - mI)**2 * db * dm) / norm
        mS = np.sum(Pgrid * pgrid[1] * db * dm) / norm
        dS = np.sum(Pgrid * (pgrid[1] - mS)**2 * db * dm) / norm
        self.pc2 = np.array([mI, mS])
        self.dpc2 = np.array([dI, dS])
        nsigs = []
        for i in [3,2,1]:
            nsigs.append(np.exp(-npatch*
                                self.chi2line([mI, mS+np.sqrt(dS)*i])/2)/norm)
        self.nnorm = norm
        self.nsigs = nsigs
        if plot:
            plt.figure(figsize=(12,8))
            try:
                plt.pcolormesh(pgrid[0], pgrid[1], Pgrid/norm, cmap=cmr.ocean_r)
            except(NameError):
                plt.pcolormesh(pgrid[0], pgrid[1], Pgrid/norm)
            plt.colorbar(label=r"Likelihood")
            ncontours = plt.contour(pgrid[0], pgrid[1], Pgrid/norm, nsigs, 
                                    colors='red')
            fmt = {}
            strs = [r"$3\sigma$", r"$2\sigma$", r"$1\sigma$"]
            for l, s in zip(ncontours.levels, strs):
                fmt[l] = s
            plt.clabel(ncontours, fmt=fmt, inline=True, fontsize=20)
            plt.axvline(self.line[0], c='k')
            plt.axhline(self.line[1], c='k')
            plt.xlabel(r"Intercept [$\mu$K$^2$]")
            plt.ylabel(r"Slope [rad$^2$]")
            plt.show()
            plt.close()

    def PearsonLikelihood(self, plot=None, plotname=None):
        """Compute the model Pearson Type III model likelihood with patches.

        Compute the model likelihood for the map's patch statistics assuming 
        that the small scale temperature variance follows a Pearson Type III 
        distribution dictated by the background temperature gradient.

        Parameters
        ----------
        plot: str, default=None
            If plot is any of ['s', 'm'], then this method calls the 
            Sliced or Marginalized plotting methods associated with the 
            Pearson likelihood, respectively. Also works for 
            ['sliced', 'marginalized'], or ['slice', 'margin']. No plot 
            if None is given.
        plotname: str, default=None
            If given, then save the likelihood corner plot at this location
        """
        bgrid = np.linspace(self.line[0] - 5*np.sqrt(self.dline[0][0]), 
                            self.line[0] + 10*np.sqrt(self.dline[0][0]), 
                            200
                           )
        mgrid = np.linspace(self.line[1] - 5*np.sqrt(self.dline[1][1]), 
                            self.line[1] + 10*np.sqrt(self.dline[1][1]), 
                            200
                           )
        Ngrid = np.linspace(10, 20, 50)
        pgrid = np.meshgrid(bgrid, mgrid, Ngrid, indexing='ij')
        self.pgrid = pgrid
        Pgrid = np.zeros((200,200,50))
        db = bgrid[1] - bgrid[0]
        dm = mgrid[1] - mgrid[0]
        dN = Ngrid[1] - Ngrid[0]
        dT2p = self._dT2patch.flatten()
        T2p = self._T2patch.flatten()
        for k, N in enumerate(Ngrid):
            Pgrid[:,:,k] = np.sum(np.log(_Pearson3(T2p[None,None,...], N, 
                                                   _lin(dT2p, 
                                                   *[pgrid[0][:,:,k][...,None], 
                                                     pgrid[1][:,:,k][...,None]]
                                                        ))), axis=2)
        Pmean = np.mean(Pgrid)
        Pgrid = np.exp(Pgrid-Pmean)
        norm = np.sum(Pgrid * db * dm * dN)
        Pgrid /= norm
        self.Pgrid = Pgrid
        mI = np.sum(Pgrid * pgrid[0] * db * dm * dN)
        dI = np.sum(Pgrid * (pgrid[0] - mI)**2 * db * dm * dN)
        mS = np.sum(Pgrid * pgrid[1] * db * dm * dN)
        dS = np.sum(Pgrid * (pgrid[1] - mS)**2 * db * dm * dN)
        mN = np.sum(Pgrid * pgrid[2] * db * dm * dN)
        dN = np.sum(Pgrid * (pgrid[2] - mN)**2 * db * dm * dN)
        self.pP3 = np.array([mI, mS, mN])
        self.dpP3 = np.array([dI, dS, dN])
        sigs = []
        for i in [3,2,1]:
            cline = [mI+np.sqrt(dI)*i, mS]
            P = np.sum(np.log(_Pearson3(T2p, mN, _lin(dT2p, *cline))))
            sigs.append(np.exp(P - Pmean)/norm)
        self.sigs = sigs
        sls = [":", "--", "-"]
        if plot == "m" or plot == "marginalized" or plot == "margin":
            self.PearsonPlotMarginalized(plotname)
        if plot == "s" or plot == "sliced" or plot == "slice":
            self.PearsonPlotSliced(plotname)

    def PearsonPlotMarginalized(self, plotname=None):
        """Plot the corner plot for the computed Pearson Type III likelihood.

        Plot the corner plot for the computed Pearson Type III likelihood. 
        The 2D likelihood ellipses will be marginalized over the remaining 
        parameter.

        Parameters
        ----------
        plotname: str, default=None
            If given, then save the likelihood corner plot at this location
        """
        pgrid = self.pgrid
        Pgrid = self.Pgrid
        sigs = self.sigs
        f, axs = plt.subplots(3, 3, figsize=(10,10), 
                              gridspec_kw={'hspace':0.125, 
                                           'wspace':0.125})
        # 2D plots
        ## b, m
        axs[2,0].pcolormesh(pgrid[0][:,:,0], pgrid[1][:,:,0], 
                            np.sum(Pgrid, axis=2), cmap=cmr.ocean_r)
        iN = np.argmin(np.abs(pgrid[0,0,:] - self.pP3[2]))
        axs[2,0].contour(pgrid[0][:,:,iN], pgrid[1][:,:,iN], 
                         Pgrid[:,:,iN], sigs, linestyles=sls, colors='C1')
        axs[2,0].contour(self.npgrid[0], self.npgrid[1], 
                         self.nPgrid/self.nnorm, self.nsigs, 
                         linestyles=sls, colors='red')
        axs[2,0].set(xlabel=r"$b$ [$\mu$K$^2$]", ylabel=r"$m$ [rad$^2$]")
        ## N, m
        axs[2,1].pcolormesh(pgrid[2][0,:,:], pgrid[1][0,:,:], 
                            np.sum(Pgrid, axis=0), cmap=cmr.ocean_r)
        iI = np.argmin(np.abs(pgrid[:,0,0] - self.pP3[0]))
        axs[2,1].contour(pgrid[2][iI,:,:], pgrid[1][iI,:,:], 
                         Pgrid[iI,:,:], sigs, linestyles=sls, colors='C1')
        axs[2,1].set(yticklabels=[], xlabel=r"$N$")
        ## b, N
        axs[1,0].pcolormesh(pgrid[0][:,0,:], pgrid[2][:,0,:], 
                            np.sum(Pgrid, axis=1), cmap=cmr.ocean_r)
        iS = np.argmin(np.abs(self.pgrid[:,0,:] - self.pP3[1]))
        axs[1,0].contour(pgrid[0][:,iS,:], pgrid[2][:,iS,:], 
                         Pgrid[:,iS,:], sigs, linestyles=sls, colors='C1')
        axs[1,0].set(xticklabels=[], ylabel=r"$N$")
        # 1D histograms
        ## b
        axs[0,0].plot(pgrid[:,0,0], np.sum(Pgrid, axis=(1,2)))
        axs[0,0].set(xticklabels=[], yticks=[], 
            title=r"$b = {:.4f} \pm {:.4f}$".format(self.pP3[0], 
                                                    np.sqrt(self.dpP3[0])))
        ## N
        axs[1,1].plot(pgrid[0,0,:], np.sum(Pgrid, axis=(0,1)))
        axs[1,1].set(xticklabels=[], yticks=[], 
            title=r"$N = {:.2f} \pm {:.2f}$".format(self.pP3[2], 
                                                    np.sqrt(self.dpP3[2])))
        ## m
        axs[2,2].plot(self.pgrid[:,0,:], np.sum(Pgrid, axis=(0,2)))
        axs[2,2].set(yticks=[], xlabel=r"$m$ [rad$^2$]", 
            title=r"$m = {:.2e} \pm {:.2e}$".format(self.pP3[1], 
                                                    np.sqrt(self.dpP3[1])))
        # Hide unused axes
        axs[0,1].axis('off')
        axs[0,2].axis('off')
        axs[1,2].axis('off')
        plt.savefig(plotname)
        plt.show()

    def PearsonPlotSliced(self, plotname=None):
        """Plot the corner plot for the computed Pearson Type III likelihood.

        Plot the corner plot for the computed Pearson Type III likelihood. 
        The 2D likelihood ellipses will be slices at the estimated value of 
        the remaining parameter.

        Parameters
        ----------
        plotname: str, default=None
            If given, then save the likelihood corner plot at this location
        """
        pgrid = self.pgrid
        Pgrid = self.Pgrid
        sigs = self.sigs
        f, axs = plt.subplots(3, 3, figsize=(10,10), 
                              gridspec_kw={'hspace':0.125, 
                                           'wspace':0.125})
        # 2D plots
        ## b, m
        iN = np.argmin(np.abs(pgrid[0,0,:] - self.pP3[2]))
        axs[2,0].pcolormesh(pgrid[0][:,:,0], pgrid[1][:,:,0], 
                            Pgrid[:,:,iN], cmap=cmr.ocean_r)
        axs[2,0].contour(pgrid[0][:,:,iN], pgrid[1][:,:,iN], 
                         Pgrid[:,:,iN], sigs, linestyles=sls, colors='C1')
        axs[2,0].contour(self.npgrid[0], self.npgrid[1], 
                         self.nPgrid/self.nnorm, self.nsigs, 
                         linestyles=sls, colors='red')
        axs[2,0].set(xlabel=r"$b$ [$\mu$K$^2$]", ylabel=r"$m$ [rad$^2$]")
        ## N, m
        iI = np.argmin(np.abs(pgrid[:,0,0] - self.pP3[0]))
        axs[2,1].pcolormesh(pgrid[2][0,:,:], pgrid[1][0,:,:], 
                            Pgrid[iI,:,:], cmap=cmr.ocean_r)
        axs[2,1].contour(pgrid[2][iI,:,:], pgrid[1][iI,:,:], 
                         Pgrid[iI,:,:], sigs, linestyles=sls, colors='C1')
        axs[2,1].set(yticklabels=[], xlabel=r"$N$")
        ## b, N
        iS = np.argmin(np.abs(self.pgrid[:,0,:] - self.pP3[1]))
        axs[1,0].pcolormesh(pgrid[0][:,0,:], pgrid[2][:,0,:], 
                            Pgrid[:,iS,:], cmap=cmr.ocean_r)
        axs[1,0].contour(pgrid[0][:,iS,:], pgrid[2][:,iS,:], 
                         Pgrid[:,iS,:], sigs, linestyles=sls, colors='C1')
        axs[1,0].set(xticklabels=[], ylabel=r"$N$")
        # 1D histograms
        ## b
        axs[0,0].plot(pgrid[:,0,0], np.sum(Pgrid, axis=(1,2)))
        axs[0,0].set(xticklabels=[], yticks=[], 
            title=r"$b = {:.4f} \pm {:.4f}$".format(self.pP3[0], 
                                                    np.sqrt(self.dpP3[0])))
        ## N
        axs[1,1].plot(pgrid[0,0,:], np.sum(Pgrid, axis=(0,1)))
        axs[1,1].set(xticklabels=[], yticks=[], 
            title=r"$N = {:.2f} \pm {:.2f}$".format(self.pP3[2], 
                                                    np.sqrt(self.dpP3[2])))
        ## m
        axs[2,2].plot(self.pgrid[:,0,:], np.sum(Pgrid, axis=(0,2)))
        axs[2,2].set(yticks=[], xlabel=r"$m$ [rad$^2$]", 
            title=r"$m = {:.2e} \pm {:.2e}$".format(self.pP3[1], 
                                                    np.sqrt(self.dpP3[1])))
        # Hide unused axes
        axs[0,1].axis('off')
        axs[0,2].axis('off')
        axs[1,2].axis('off')
        plt.savefig(plotname)
        plt.show()

    def PearsonMPI(self):
        """
        """
        line = np.empty(2)
        dline = np.empty(2)
        if self.commrank == 0:
            line = self.line
            dline = np.sqrt(np.diag(self.dline))
        self.comm.Bcast(line, root=0)
        self.comm.Bcast(dline, root=0)
        bgrid = np.linspace(line[0] - 10*dline[0], 
                            line[0] + 10*dline[0], 
                            200
                            )
        mgrid = np.linspace(line[1] - 10*dline[1], 
                            line[1] + 10*dline[1], 
                            200
                            )
        Ngrid = np.linspace(10, 20, 40)
        pgrid = np.meshgrid(bgrid, mgrid, Ngrid, indexing='ij')
        if self.commrank == 0:
            self.pgrid = pgrid
        db = bgrid[1] - bgrid[0]
        dm = mgrid[1] - mgrid[0]
        dN = Ngrid[1] - Ngrid[0]
        npatch = np.empty(1, dtype=int)
        if self.commrank == 0:
            npatch[0] = self._dT2patch.size
        self.comm.Bcast(npatch, root=0)
        if self.commrank == 0:
            dT2p = self._dT2patch.flatten()
            T2p = self._T2patch.flatten()
        else:
            dT2p = np.empty(npatch)
            T2p = np.empty(npatch)
        self.comm.Bcast(dT2p, root=0)
        self.comm.Bcast(T2p, root=0)

        subsize = 40 // self.commsize
        subN = Ngrid[self.commrank*subsize:(self.commrank+1)*subsize]
        subP = np.zeros((subsize,200,200))
        Pgrid = np.zeros((self.commsize,subsize,200,200))

        for k, N in enumerate(subN):
            subP[k,:,:] = np.sum(np.log(_Pearson3(T2p[None,None,...], N, 
                                                  _lin(dT2p, 
                                                  *[pgrid[0][:,:,k][...,None], 
                                                    pgrid[1][:,:,k][...,None]]
                                                      ))), axis=2)
        self.comm.Gather(subP, Pgrid, root=0)
        if self.commrank == 0:
            Pgrid = np.concatenate(Pgrid, axis=0)
            Pgrid = Pgrid.swapaxes(0, 1).swapaxes(1, 2)
            Pmean = np.mean(Pgrid)
            Pgrid = np.exp(Pgrid-Pmean)
            norm = np.sum(Pgrid * db * dm * dN)
            Pgrid /= norm
            self.Pgrid = Pgrid
            mI = np.sum(Pgrid * pgrid[0] * db * dm * dN)
            dI = np.sum(Pgrid * (pgrid[0] - mI)**2 * db * dm * dN)
            mS = np.sum(Pgrid * pgrid[1] * db * dm * dN)
            dS = np.sum(Pgrid * (pgrid[1] - mS)**2 * db * dm * dN)
            mN = np.sum(Pgrid * pgrid[2] * db * dm * dN)
            dN = np.sum(Pgrid * (pgrid[2] - mN)**2 * db * dm * dN)
            self.pP3 = np.array([mI, mS, mN])
            self.dpP3 = np.array([dI, dS, dN])
            sigs = []
            for i in [3,2,1]:
                cline = [mI+np.sqrt(dI)*i, mS]
                P = np.sum(np.log(_Pearson3(T2p, mN, _lin(dT2p, *cline))))
                sigs.append(np.exp(P - Pmean)/norm)
            self.sigs = sigs