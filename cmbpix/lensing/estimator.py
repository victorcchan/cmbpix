import numpy as np
import healpy as hp
from cmbpix.utils import *

class LensingEstimator():
	"""Estimator for small scale lensing in a full sky CMB map.
	"""
	def __init__(self, cmbmap):
		"""Initiate the LensingEstimator object.

		Create an object for reducting a HEALPix map of the cosmic 
		microwave background in (mostly) pixel-space to estimate the 
		effects of small angular scale weak lensing due to structure 
		formation.

		Parameters
		----------
		map: 1darray
			A CMB map in HEALPix format.

		Attributes
		----------
		map_in: 1darray
			The input CMB map in HEALPix format.
		NSIDE_in: int
			The NSIDE resolution of the input map.
		map_dtheta: 1darray
			The gradient of the input map with respect to theta in 
			spherical coordinates.
		map_dphi: 1darray
			The gradient of the input map with respect to phi in 
			spherical coordinates. The map is already divided by 
			sin(theta).

		"""
		self.map_in = cmbmap
		self.NSIDE_in = hp.npix2nside(self.map_in.size)

	def evaluate_gradient(self):
		"""Evaluate the 2D gradient of the CMB map.
		
		Evaluate the 2D gradient of the original input map across 
		the sky. This method uses the ``healpy`` function that returns 
		d/dtheta and d/dphi/sin(theta).

		"""
		alm = hp.map2alm(self.map_in)
		m, self.map_dtheta, self.map_dphi = hp.alm2map_der1(alm, self.NSIDE_in)
		del m, alm