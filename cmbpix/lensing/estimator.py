import numpy as np
import healpy as hp
from cmbpix.utils import *

class LensingEstimator():
	"""Estimator for small scale lensing in a full sky CMB map.

	An object for reducing a HEALPix map of the cosmic microwave 
	background in (mostly) pixel-space to estimate the effects 
	of small angular scale weak lensing due to structure formation.

	"""
	def __init__(self, cmbmap):
		"""Initiate the estimator.

		Parameters
		----------
		cmbmap: 1darray
			A CMB map in HEALPix format.

		Attributes
		----------
		map_in: 1d-array
			The input CMB map in HEALPix format.
		map_dtheta: 1d-array
			The gradient of the input map with respect to theta in 
			spherical coordinates. Created after executing the method 
			evaluate_gradient.
		map_dphi: 1d-array
			The gradient of the input map with respect to phi in 
			spherical coordinates. The map is already divided by 
			sin(theta). Created after executing the method 
			evaluate_gradient.

		"""
		self.map_in = cmbmap
		self._NSIDE_small = hp.npix2nside(self.map_in.size)
		self._NSIDE_large = 256

	def evaluate_gradient(self):
		"""Evaluate the 2D gradient of the CMB map.
		
		Evaluate the 2D gradient of the original input map across 
		the sky. This method uses the ``healpy`` function that returns 
		d/dtheta and d/dphi/sin(theta).

		"""
		alm = hp.map2alm(self.map_in)
		m, self.map_dtheta, self.map_dphi = hp.alm2map_der1(alm, self._NSIDE_small)
		del m, alm