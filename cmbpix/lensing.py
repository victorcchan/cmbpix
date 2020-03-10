import numpy as np
import healpy as hp

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

		History
		-------
		2020-03-10: First implementation.
		"""
		self.map_in = cmbmap
		self.NSIDE_in = hp.npix2nside(self.map_in.size)