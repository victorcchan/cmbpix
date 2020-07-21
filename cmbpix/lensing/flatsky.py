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
	elldT: value, default=2000
		The maximum ell to consider for the large scale background 
		temperature gradient.
	ellmin: value, default=3000
		The minimum ell to consider for the small scale temperature 
		fluctuations.
	ellmax: value, default=None
		The maximum ell to consider for the small scale temperature 
		fluctuations. If None, there is no cutoff.

	"""

	def __init__(self, cmbmap, elldT=2000, ellmin=3000, ellmax=None, 
					patchsize=40):
		"""Initiate the estimator.

		Parameters
		----------
		cmbmap: ndmap
			A CMB map.
		elldT: value, default=2000
			The maximum ell to consider for the large scale background 
			temperature gradient.
		ellmin: value, default=3000
			The minimum ell to consider for the small scale temperature 
			fluctuations.
		ellmax: value, default=None
			The maximum ell to consider for the small scale temperature 
			fluctuations. If None, there is no cutoff.
		patchsize: int, default=40
			The side length of patches (in number of pixels) for the map to be 
			broken up into to gather local lensing statistics.
		"""
		self.map_in = cmbmap
		self.elldT = elldT
		self.ellmin = ellmin
		self.ellmax = ellmax
		self.patchsize = patchsize
		# Derived attributes
		self._ly, self._lx = self.map_in.lmap()
		self._lmod = self.map_in.modlmap()

	def gather_patches():
		"""Assemble patch statistics relevant to lensing at small scales.

		Return the small scale (ell > ellmin) temperature power at different 
		patches across the sky as well as the average amplitude of the 
		background temperature gradient (ell < elldT).

		Parameters
		----------
		map_in: ndmap, ndarray
		    The input map.
		p: int
		    The length of one side of a patch in number of pixels.

		Returns
		-------
		Trms: 2darray
		    The small scale temperature power across patches.
		dT: 2darray
		    The background temperature gradient.
		"""
		m_fft = enmap.fft(self.map_in)
		hp = np.zeros(map_in.shape)
		hp[np.where(lmod > self.ellmin)] = 1.
		self._Tss = enmap.ifft(m_fft * hp)
		self._dTy, self._dTx = gradient_flat(map_in, lmax)
		self._dT = np.sqrt(self._dTx**2 + self._dTy**2)
		# Scale geometry for lower res map of patches
		pshp, pwcs = enmap.scale_geometry(map_in.shape, map_in.wcs, 1./p)
		self._Tpatch = enmap.zeros(pshp, pwcs)
		self._dTxpatch = enmap.zeros(pshp, pwcs)
		self._dTypatch = enmap.zeros(pshp, pwcs)
		Trs = Tss[:pshp[-2]*p,:pshp[-1]*p].reshape([pshp[-2],p,pshp[-1],p])
		dTxrs = dTx[:pshp[-2]*p,:pshp[-1]*p].reshape([pshp[-2],p,pshp[-1],p])
		dTyrs = dTy[:pshp[-2]*p,:pshp[-1]*p].reshape([pshp[-2],p,pshp[-1],p])
		self._Tpatch[:,:] = np.var(Trs, axis=(1,3))
		self._dTypatch[:,:] = np.mean(dTyrs, axis=(1,3))
		self._dTxpatch[:,:] = np.mean(dTxrs, axis=(1,3))